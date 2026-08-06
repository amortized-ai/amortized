from __future__ import annotations

import json
import logging
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.jobs.base import JobBuildError, JobBuildResult
from amortized.jobs.common import set_mlflow_run_tag

logger = logging.getLogger("amortized.jobs.upload")


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    run_id = config.get("mlflow_upload_run_id")
    if not run_id:
        raise JobBuildError("mlflow_upload_run_id is required for upload jobs")

    filename = config.get("filename", "document")
    output_format = config.get("output_format", "md")
    chunker_type = config.get("chunker_type", "sentence")
    chunk_size = config.get("chunk_size", 2048)
    chunk_overlap = config.get("chunk_overlap", 200)

    artifact_path = config.get("artifact_path", "source")
    input_dir = "/amortized/work/input"
    pre_cmd = (
        f'python3 -c "'
        f"import mlflow; "
        f"mlflow.artifacts.download_artifacts("
        f"run_id='{run_id}', "
        f"artifact_path='{artifact_path}', "
        f"dst_path='{input_dir}')"
        f'"'
    )

    config_dict = {
        "input_path": f"{input_dir}/{artifact_path}/{filename}",
        "filename": filename,
        "output_format": output_format,
        "chunker_type": chunker_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "tokenizer": "cl100k_base",
    }

    config_files["config.json"] = json.dumps(config_dict)

    image = f"{config_mod.settings.image_registry}/document:latest"

    return JobBuildResult(
        command=["python3", "/app/process_document.py"],
        config_files=config_files,
        pre_commands=[pre_cmd],
        resources=Resources(gpus=0),
        image=image,
        resolved_config=config_dict,
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    await set_mlflow_run_tag(mlflow_run_id, "job_type", "document")

    job_config = job.get("config", {})
    if isinstance(job_config, str):
        job_config = json.loads(job_config)
    await set_mlflow_run_tag(mlflow_run_id, "filename", job_config.get("filename", ""))
