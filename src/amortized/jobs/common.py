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


async def _resolve_mlflow_run_by_job_id(job_id: str) -> str:
    """Search MLflow for a run tagged with this job_id and return its run_id."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri:
        return ""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.post(
                f"{tracking_uri}/api/2.0/mlflow/experiments/search",
                json={"max_results": 200},
            )
            resp.raise_for_status()
            exp_ids = [e["experiment_id"] for e in resp.json().get("experiments", [])]

        if not exp_ids:
            return ""
        client = MLflowClient(tracking_uri)
        runs = await client.search_runs(
            exp_ids,
            filter_string=f"tags.job_id = '{job_id}'",
            max_results=1,
        )
        if runs:
            return runs[0]["info"]["run_id"]
    except Exception:
        logger.warning("Failed to find MLflow run for job_id %s", job_id, exc_info=True)
    return ""


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

    if parent:
        parent_run_id = parent.get("mlflow_run_id", "")
        parent_type = parent.get("type", "")
    else:
        logger.info("Parent job %s not in local DB, searching MLflow", parent_job_id)
        parent_run_id = await _resolve_mlflow_run_by_job_id(parent_job_id)
        parent_type = "sdg"

    if not parent_run_id:
        logger.warning("Could not resolve MLflow run for parent job %s", parent_job_id)
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

    if job["type"] == JobType.training.value and parent_type in ("sdg", "upload"):
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
