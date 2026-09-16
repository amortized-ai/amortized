"""SDG job builder — Data Designer synthetic data generation."""

from __future__ import annotations

import json
import logging
import os
import shlex
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.core.model_catalog import enabled_provider_defs
from amortized.jobs.base import JobBuildError, JobBuildResult
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


class SDGBuildError(JobBuildError):
    """Raised when SDG config building fails and the job should be marked failed."""


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
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
            "chunk_size",
            "chunk_overlap",
            "tokenizer",
            "sentences_per_chunk",
            "min_text_length",
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
                job_id,
                chunk_count,
                len(document_ids),
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
    mode = config.pop("mode", "create")
    topic = config.pop("topic", None)

    dd_config = {"data_designer": config}
    config_files["config.yaml"] = yaml.dump(dd_config, default_flow_style=False, sort_keys=False)

    # Direct-provider stopgap for the absent MLflow AI Gateway: the job image's
    # builtin default provider is `gateway` (a bundled MLflow gateway that does not
    # exist here), so write a model_providers.yaml with the providers enabled by
    # dropped-in keys and point DATA_DESIGNER_HOME at it.
    pre_commands: list[str] = []
    provider_defs = enabled_provider_defs()
    if provider_defs:
        dd_home = "/amortized/work/.data-designer"
        config_files["model_providers.yaml"] = yaml.dump(
            {"providers": provider_defs}, default_flow_style=False, sort_keys=False
        )
        env["DATA_DESIGNER_HOME"] = dd_home
        # Deliver each enabled provider's key into the job (injected as a per-job
        # Secret via spec.env) so the direct-provider path does not depend on the
        # operator also listing the key in forward_env. api_key is an env-var name
        # for builtin providers; literal keys are already inlined in the yaml above.
        for pdef in provider_defs:
            key_name = pdef.get("api_key", "")
            if key_name.isupper() and "_" in key_name and key_name in os.environ:
                env[key_name] = os.environ[key_name]
        pre_commands.append(
            f"mkdir -p {dd_home}"
            f" && cp /amortized/model_providers.yaml {dd_home}/model_providers.yaml"
        )

    records = min(num_records, 10) if mode == "preview" else num_records
    dd_cmd = (
        "data-designer create /amortized/config.yaml"
        f" --num-records {records}"
        " --artifact-path /amortized/work"
        " --no-tui"
    )
    processor_names = [p.get("name", "") for p in config.get("processors", [])]
    proc_dir = f"processors-files/{processor_names[-1]}" if processor_names else ""
    upload_dir = (
        f"/amortized/work/dataset/{proc_dir}"
        if proc_dir
        else "/amortized/work/dataset/parquet-files"
    )

    all_cmds = [*doc_setup_cmds, dd_cmd]
    cmd = ["sh", "-c", " && ".join(all_cmds)]

    post_cmd = (
        f"mlflow artifacts log-artifacts"
        f" -l {shlex.quote(upload_dir)}"
        f" -r $MLFLOW_RUN_ID"
        f" -a generated_data"
    )

    resolved_config = dict(config)
    resolved_config["num_records"] = records
    if mode != "create":
        resolved_config["mode"] = mode
    if topic:
        resolved_config["topic"] = topic

    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        env=env,
        pre_commands=pre_commands,
        post_commands=[post_cmd],
        resources=Resources(gpus=0),
        image=IMAGE,
        resolved_config=resolved_config,
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    await set_mlflow_run_tag(mlflow_run_id, "source", "sdg")

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

    try:
        tracking_uri = config_mod.settings.mlflow_tracking_uri
        if tracking_uri:
            from amortized.core.mlflow_client import MLflowClient

            client = MLflowClient(tracking_uri)
            run = await client.get_run(mlflow_run_id)
            run_name = run["info"].get("run_name", job["id"][:8])
            await set_mlflow_run_tag(mlflow_run_id, "dataset_name", f"ds-{run_name}")
    except Exception:
        logger.debug("Failed to set dataset display name", exc_info=True)
