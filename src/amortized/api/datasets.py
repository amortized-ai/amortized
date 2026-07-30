"""Dataset upload endpoint."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import aiosqlite
import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from amortized.config import settings
from amortized.core.jobs import create_job
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import Job, JobType

logger = logging.getLogger("amortized.api.datasets")

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

_ALLOWED_EXTENSIONS = (".jsonl", ".parquet")


def _tracking_uri() -> str:
    uri = settings.mlflow_tracking_uri
    if not uri:
        raise HTTPException(status_code=503, detail="MLflow tracking URI not configured")
    return uri


async def _store_dataset_in_mlflow(
    filename: str,
    file_bytes: bytes,
) -> tuple[str, str]:
    """Create an MLflow run and upload the file under generated_data/."""
    tracking_uri = _tracking_uri()
    async with httpx.AsyncClient(timeout=60.0) as client:
        experiment_name = "amortized/datasets"
        resp = await client.get(
            f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": experiment_name},
        )
        if resp.status_code == 404 or "RESOURCE_DOES_NOT_EXIST" in resp.text:
            create_resp = await client.post(
                f"{tracking_uri}/api/2.0/mlflow/experiments/create",
                json={"name": experiment_name},
            )
            if create_resp.status_code == 409:
                refetch = await client.get(
                    f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name",
                    params={"experiment_name": experiment_name},
                )
                refetch.raise_for_status()
                experiment_id = refetch.json()["experiment"]["experiment_id"]
            else:
                create_resp.raise_for_status()
                experiment_id = create_resp.json()["experiment_id"]
        else:
            resp.raise_for_status()
            exp = resp.json()["experiment"]
            experiment_id = exp["experiment_id"]
            if exp.get("lifecycle_stage") == "deleted":
                await client.post(
                    f"{tracking_uri}/api/2.0/mlflow/experiments/restore",
                    json={"experiment_id": experiment_id},
                )

        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        run_resp = await client.post(
            f"{tracking_uri}/api/2.0/mlflow/runs/create",
            json={
                "experiment_id": experiment_id,
                "run_name": filename,
                "start_time": now_ms,
                "tags": [
                    {"key": "job_type", "value": "upload"},
                    {"key": "dataset_name", "value": filename},
                    {"key": "source", "value": "upload"},
                ],
            },
        )
        run_resp.raise_for_status()
        run_id: str = run_resp.json()["run"]["info"]["run_id"]

        try:
            artifact_resp = await client.put(
                f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/generated_data/{filename}",
                params={"run_id": run_id},
                content=file_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
            artifact_resp.raise_for_status()

            await client.post(
                f"{tracking_uri}/api/2.0/mlflow/runs/update",
                json={
                    "run_id": run_id,
                    "status": "FINISHED",
                    "end_time": int(datetime.now(UTC).timestamp() * 1000),
                },
            )
        except Exception:
            try:
                await client.post(
                    f"{tracking_uri}/api/2.0/mlflow/runs/update",
                    json={"run_id": run_id, "status": "FAILED"},
                )
            except Exception:
                logger.debug("Could not mark MLflow run %s as FAILED", run_id)
            raise

        return run_id, experiment_id


@router.post("/upload", response_model=Job)
async def upload_dataset(
    file: UploadFile,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Any:
    """Upload a JSONL or Parquet file as a training dataset."""
    name = file.filename or "dataset"
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    run_id, experiment_id = await _store_dataset_in_mlflow(name, file_bytes)

    repo = Repository(db)
    row = await create_job(
        repo,
        job_type=JobType.upload,
        config={"source": "upload", "original_filename": name},
    )

    updated = await repo.update_job(
        row["id"],
        status="succeeded",
        mlflow_run_id=run_id,
        mlflow_experiment=experiment_id,
        completed_at=datetime.now(UTC).isoformat(),
    )
    assert updated is not None

    return updated
