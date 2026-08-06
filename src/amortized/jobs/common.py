"""Shared helpers used by multiple job builders."""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient

logger = logging.getLogger("amortized.jobs")

ArtifactResult = tuple[dict[str, Any], list[str]]


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
) -> ArtifactResult:
    """Resolve parent job artifacts and inject into config for chaining.

    Returns (config, pre_commands) where pre_commands are shell strings that
    download artifacts via ``mlflow.artifacts.download_artifacts()``.
    The caller must place pre_commands into ``JobBuildResult.pre_commands``.
    """
    parent_job_id = job.get("parent_job_id", "") or config.get("parent_job_id", "")
    if not parent_job_id:
        return config, []

    from amortized.db.connection import _get_shared_db
    from amortized.db.repository import Repository

    db = await _get_shared_db()
    repo = Repository(db)
    parent = await repo.get_job(parent_job_id)

    if not parent:
        logger.warning("Parent job %s not found", parent_job_id)
        return config, []

    parent_run_id = parent.get("mlflow_run_id", "")
    if not parent_run_id:
        logger.warning("Parent job %s has no mlflow_run_id", parent_job_id)
        return config, []

    pre_commands: list[str] = []
    config = dict(config)
    from amortized.models import JobType

    if job["type"] == JobType.training.value and parent["type"] in ("sdg", "upload"):
        existing = config.get("data_path", "")
        if not existing or not existing.startswith("s3://"):
            local_dir = "/amortized/work/data"
            pre_cmd = (
                f"mlflow artifacts download"
                f" -r {shlex.quote(parent_run_id)}"
                f" -a generated_data"
                f" -d {shlex.quote(local_dir)}"
            )
            pre_commands.append(pre_cmd)
            config["data_path"] = f"{local_dir}/generated_data"
            logger.info(
                "Will download artifacts from MLflow run %s to %s",
                parent_run_id,
                local_dir,
            )

    return config, pre_commands
