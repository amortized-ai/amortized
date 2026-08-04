"""SDG job builder — Data Designer synthetic data generation."""

from __future__ import annotations

import json
import logging
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources, S3Download
from amortized.jobs.base import JobBuildResult
from amortized.jobs.common import fetch_document_chunks, set_mlflow_run_tag

logger = logging.getLogger("amortized.jobs.sdg")

IMAGE = "ghcr.io/amortized-ai/data-designer:latest"

_STALE_CONFIG_KEYS = (
    "model",
    "api_base",
    "api_key",
    "num_samples",
    "max_concurrency",
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "num_retries",
    "input_data",
    "input_documents",
    "strategy_params",
    "task_description",
    "document_id",
    "output_dir",
)


class SDGBuildError(Exception):
    """Raised when SDG config building fails and the job should be marked failed."""


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
    s3_downloads: list[S3Download],
) -> JobBuildResult:
    import yaml

    env: dict[str, str] = {"DD_API_KEY": "not-needed"}
    job_id = job["id"]

    for key in _STALE_CONFIG_KEYS:
        config.pop(key, None)

    document_ids = config.pop("document_ids", []) or config.pop("document_id", [])
    if isinstance(document_ids, str):
        document_ids = [document_ids]

    doc_setup_cmds: list[str] = []
    chunk_count = 0

    if document_ids and config_mod.settings.mlflow_tracking_uri:
        seed_config = config.get("seed_config", {})
        source = seed_config.get("source", {})
        stale_source_keys = (
            "chunk_size", "chunk_overlap", "tokenizer",
            "sentences_per_chunk", "min_text_length",
        )
        for key in stale_source_keys:
            source.pop(key, None)

        for doc_id in document_ids:
            try:
                chunks = await fetch_document_chunks(doc_id)
            except Exception:
                raise SDGBuildError(f"Failed to fetch chunks for document {doc_id}") from None
            for chunk_text in chunks:
                config_files[f"chunk_{chunk_count}.md"] = chunk_text
                chunk_count += 1

        if chunk_count:
            doc_setup_cmds = [
                "mkdir -p /tmp/chunks",
                *(f"cp /amortized/chunk_{i}.md /tmp/chunks/" for i in range(chunk_count)),
            ]
            source["seed_type"] = "file_contents"
            source["path"] = "/tmp/chunks"
            source.setdefault("encoding", "utf-8")
            seed_config["source"] = source
            config["seed_config"] = seed_config

            for col in config.get("columns", []):
                for field in ("prompt", "system_prompt"):
                    val = col.get(field, "")
                    if "{{ text }}" in val:
                        col[field] = val.replace("{{ text }}", "{{ content }}")

            logger.info(
                "Job %s: fetched %d pre-chunked chunks from %d document(s)",
                job_id, chunk_count, len(document_ids),
            )
        else:
            raise SDGBuildError(
                f"Job requires {len(document_ids)} document(s) but no chunks"
                " could be fetched from MLflow. Check that the document"
                " IDs are valid and MLflow is reachable."
            )

    for mc in config.get("model_configs", []):
        params = mc.setdefault("inference_parameters", {})
        params.setdefault("max_parallel_requests", 32)

    for col in config.get("columns", []):
        if "model_config_alias" in col:
            col.setdefault("model_alias", col.pop("model_config_alias"))

    num_records = config.pop("num_records", 100)
    config.pop("topic", None)

    dd_config = {"data_designer": config}
    config_files["config.yaml"] = yaml.dump(
        dd_config, default_flow_style=False, sort_keys=False
    )

    dd_cmd = (
        "data-designer create /amortized/config.yaml"
        f" --num-records {num_records}"
        " --artifact-path /amortized/work"
        " --no-tui"
    )
    processor_names = [p.get("name", "") for p in config.get("processors", [])]
    proc_dir = f"processors-files/{processor_names[-1]}" if processor_names else ""
    upload_cmd = (
        f"python3 /usr/local/bin/upload_to_mlflow.py /amortized/work/dataset {proc_dir}"
    )
    all_cmds = [*doc_setup_cmds, dd_cmd, upload_cmd]
    cmd = ["sh", "-c", " && ".join(all_cmds)]

    resolved_config = dict(config)
    resolved_config["num_records"] = num_records

    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        s3_downloads=s3_downloads,
        env=env,
        resources=Resources(gpus=0),
        image=IMAGE,
        resolved_config=resolved_config,
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    job_config = job.get("config", {})
    if isinstance(job_config, str):
        job_config = json.loads(job_config)

    nr = job_config.get("num_records", "")
    if nr:
        await set_mlflow_run_tag(mlflow_run_id, "num_samples", str(nr))

    mc = job_config.get("model_configs", [])
    if mc and isinstance(mc, list):
        model_name = mc[0].get("model", "")
        await set_mlflow_run_tag(mlflow_run_id, "teacher_model", model_name)

    topic = job_config.get("topic", "")
    if topic:
        await set_mlflow_run_tag(mlflow_run_id, "dataset_topic", topic)
