"""Upload job builder — document processing via containerized docling + chonkie."""

from __future__ import annotations

import json
import logging
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources, S3Download
from amortized.jobs.base import JobBuildError, JobBuildResult
from amortized.jobs.common import set_mlflow_run_tag

logger = logging.getLogger("amortized.jobs.upload")


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
    s3_downloads: list[S3Download],
) -> JobBuildResult:
    s3_uri = config.get("s3_uri")
    if not s3_uri:
        raise JobBuildError("s3_uri is required for upload jobs")

    filename = config.get("filename", "document")
    output_format = config.get("output_format", "md")
    chunker_type = config.get("chunker_type", "sentence")
    chunk_size = config.get("chunk_size", 2048)
    chunk_overlap = config.get("chunk_overlap", 200)

    s3_downloads.append(
        S3Download(
            s3_uri=s3_uri,
            local_path=f"/amortized/work/input/{filename}",
            is_directory=False,
        )
    )

    config_dict = {
        "input_path": f"/amortized/work/input/{filename}",
        "filename": filename,
        "output_format": output_format,
        "chunker_type": chunker_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "tokenizer": "cl100k_base",
    }

    config_files["config.json"] = json.dumps(config_dict)

    image = f"{config_mod.settings.image_registry}/document:latest"

    cmd = "mkdir -p /amortized/work/.cache && python3 /app/process_document.py"
    return JobBuildResult(
        command=["sh", "-c", cmd],
        config_files=config_files,
        s3_downloads=s3_downloads,
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
