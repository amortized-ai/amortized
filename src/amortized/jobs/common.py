"""Shared helpers used by multiple job builders."""

from __future__ import annotations

import json
import logging
from typing import Any

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient

logger = logging.getLogger("amortized.jobs")


async def set_mlflow_run_tag(mlflow_run_id: str, key: str, value: str) -> None:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return

    try:
        client = MLflowClient(tracking_uri)
        await client.set_tag(mlflow_run_id, key, value)
    except Exception:
        logger.warning(
            "Failed to set MLflow tag %s=%s on run %s",
            key,
            value,
            mlflow_run_id,
            exc_info=True,
        )


async def fetch_document_chunks(document_id: str) -> list[str]:
    """Fetch pre-chunked document content from MLflow artifact store."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not document_id:
        return []

    client = MLflowClient(tracking_uri)
    metadata_text = await client.get_artifact_text(document_id, "chunks/metadata.json")
    if not metadata_text:
        return []

    metadata = json.loads(metadata_text)
    chunks: list[str] = []
    for i in range(len(metadata)):
        text = await client.get_artifact_text(document_id, f"chunks/chunk_{i:03d}.md")
        if text:
            chunks.append(text)
    return chunks


async def resolve_parent_artifacts(
    job: dict[str, Any],
    config: dict[str, Any],
    s3_downloads: list,
) -> dict[str, Any]:
    """Resolve parent job artifacts and inject into config for chaining."""
    from amortized.backends import S3Download

    parent_job_id = job.get("parent_job_id", "") or config.get("parent_job_id", "")
    if not parent_job_id:
        return config

    from amortized.db.connection import _get_shared_db
    from amortized.db.repository import Repository

    db = await _get_shared_db()
    repo = Repository(db)
    parent = await repo.get_job(parent_job_id)

    if not parent:
        logger.warning("Parent job %s not found", parent_job_id)
        return config

    parent_run_id = parent.get("mlflow_run_id", "")
    if not parent_run_id:
        logger.warning("Parent job %s has no mlflow_run_id", parent_job_id)
        return config

    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri:
        return config

    try:
        client = MLflowClient(tracking_uri)
        run = await client.get_run(parent_run_id)
        artifact_uri: str = run["info"]["artifact_uri"]
    except Exception:
        logger.warning(
            "Could not resolve artifact URI for run %s", parent_run_id, exc_info=True
        )
        return config

    config = dict(config)
    from amortized.models import JobType

    if job["type"] == JobType.training.value and parent["type"] in ("sdg", "upload"):
        existing = config.get("data_path", "")
        if not existing or not existing.startswith("s3://"):
            s3_dir = f"{artifact_uri}/generated_data/"
            local_dir = "/amortized/work/data"
            s3_downloads.append(
                S3Download(
                    s3_uri=s3_dir,
                    local_path=local_dir,
                    is_directory=True,
                )
            )
            config["data_path"] = local_dir
            logger.info(
                "Injected SDG data from MLflow run %s: %s",
                parent_run_id,
                s3_dir,
            )

    return config
