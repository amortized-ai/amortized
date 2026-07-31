"""Dataset upload endpoint."""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite
import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from amortized.config import settings
from amortized.core.jobs import create_job
from amortized.core.mlflow_client import MLflowClient
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import Job, JobType

logger = logging.getLogger("amortized.api.datasets")

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

_ALLOWED_EXTENSIONS = (".jsonl", ".parquet")
_MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[/\\:\x00]", "_", name)
    if len(name) > 255:
        name = name[:255]
    return name or f"upload-{uuid.uuid4().hex[:8]}"


async def _store_dataset_in_mlflow(
    filename: str,
    file_bytes: bytes,
) -> tuple[str, str]:
    """Create an MLflow run and upload the file under generated_data/."""
    uri = settings.mlflow_tracking_uri
    if not uri:
        raise HTTPException(status_code=503, detail="MLflow tracking URI not configured")

    mlflow = MLflowClient(uri, timeout=60.0)
    experiment_id = await mlflow.ensure_experiment("amortized/datasets")
    run_id = await mlflow.create_run(
        experiment_id,
        name=filename,
        tags={
            "job_type": "upload",
            "dataset_name": filename,
            "source": "upload",
        },
    )

    try:
        await mlflow.upload_artifact(run_id, f"generated_data/{filename}", file_bytes)
        await mlflow.finish_run(run_id)
    except Exception:
        await mlflow.fail_run_quiet(run_id)
        raise

    return run_id, experiment_id


@router.post("/upload", response_model=Job)
async def upload_dataset(
    file: UploadFile,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Any:
    """Upload a JSONL or Parquet file as a training dataset."""
    name = _sanitize_filename(file.filename or "dataset")
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes, max {_MAX_UPLOAD_BYTES})",
        )

    try:
        run_id, experiment_id = await _store_dataset_in_mlflow(name, file_bytes)
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot connect to MLflow") from None
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="MLflow request timed out") from None
    except httpx.TransportError:
        raise HTTPException(status_code=502, detail="MLflow communication error") from None

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
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to update job record")

    return updated
