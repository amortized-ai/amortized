"""Dataset management — upload, list, inspect, and sample datasets."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from amortized.config import settings
from amortized.core.jobs import create_job
from amortized.core.mlflow_client import MLflowClient
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import Job, JobType

logger = logging.getLogger("amortized.api.datasets")

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

_upload_tasks: set[asyncio.Task[None]] = set()

_ALLOWED_EXTENSIONS = (".jsonl", ".parquet")
_MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[/\\:\x00]", "_", name)
    if len(name) > 255:
        name = name[:255]
    return name or f"upload-{uuid.uuid4().hex[:8]}"


def _topic_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return " ".join(stem.replace("_", " ").replace("-", " ").split())


async def _store_dataset_in_mlflow(
    filename: str,
    file_bytes: bytes,
) -> tuple[str, str]:
    """Create an MLflow run and upload the file under generated_data/."""
    uri = settings.mlflow_tracking_uri
    if not uri:
        raise HTTPException(status_code=503, detail="MLflow tracking URI not configured")

    tags: dict[str, str] = {
        "job_type": "upload",
        "dataset_name": filename,
        "source": "dataset",
    }
    topic = _topic_from_filename(filename)
    if topic:
        tags["dataset_topic"] = topic

    mlflow = MLflowClient(uri, timeout=60.0)
    experiment_id = await mlflow.ensure_experiment("amortized/datasets")
    run_id = await mlflow.create_run(
        experiment_id,
        name=filename,
        tags=tags,
    )

    try:
        await mlflow.upload_artifact(run_id, f"generated_data/{filename}", file_bytes)
        await mlflow.finish_run(run_id)
    except Exception:
        await mlflow.fail_run_quiet(run_id)
        raise

    return run_id, experiment_id


async def _process_dataset_upload(
    job_id: str,
    filename: str,
    file_bytes: bytes,
) -> None:
    from amortized.db.connection import get_pool

    try:
        run_id, experiment_id = await _store_dataset_in_mlflow(
            filename, file_bytes,
        )
    except Exception as exc:
        logger.warning("Dataset upload failed for job %s: %s", job_id, exc)
        async with get_pool().acquire() as conn:
            await Repository(conn).update_job(
                job_id,
                status="failed",
                completed_at=datetime.now(UTC),
                error=str(exc),
            )
        return

    async with get_pool().acquire() as conn:
        await Repository(conn).update_job(
            job_id,
            status="succeeded",
            mlflow_run_id=run_id,
            mlflow_experiment=experiment_id,
            completed_at=datetime.now(UTC),
        )


@router.post("/upload", response_model=Job, status_code=202)
async def upload_dataset(
    file: UploadFile,
    db: asyncpg.Connection = Depends(_get_db),
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

    repo = Repository(db)
    row = await create_job(
        repo,
        job_type=JobType.upload,
        config={"source": "dataset", "original_filename": name},
    )

    task = asyncio.create_task(_process_dataset_upload(row["id"], name, file_bytes))
    _upload_tasks.add(task)
    task.add_done_callback(_upload_tasks.discard)

    return row


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _mlflow_client() -> MLflowClient:
    uri = settings.mlflow_tracking_uri
    if not uri:
        raise HTTPException(status_code=503, detail="MLflow tracking URI not configured")
    return MLflowClient(uri)


def _run_to_summary(run: dict[str, Any]) -> dict[str, Any]:
    tags: dict[str, str] = {}
    for t in run.get("data", {}).get("tags", []):
        tags[t["key"]] = t["value"]
    params: dict[str, str] = {}
    for p in run.get("data", {}).get("params", []):
        params[p["key"]] = p["value"]
    metrics: dict[str, float] = {}
    for m in run.get("data", {}).get("metrics", []):
        metrics[m["key"]] = m["value"]
    info = run.get("info", {})

    samples = tags.get("num_samples", "")
    if not samples and "num_samples_generated" in metrics:
        samples = str(int(metrics["num_samples_generated"]))

    teacher = tags.get("teacher_model", "")
    if not teacher:
        teacher = params.get("model", "")

    return {
        "run_id": info.get("run_id", ""),
        "name": tags.get("dataset_name", info.get("run_name", "")),
        "topic": tags.get("dataset_topic", ""),
        "source": tags.get("source", "sdg"),
        "samples": samples,
        "teacher_model": teacher,
        "job_id": tags.get("job_id", ""),
        "experiment_id": info.get("experiment_id", ""),
        "created_at": info.get("start_time"),
    }


async def _get_all_experiment_ids(mlflow: MLflowClient) -> list[str]:
    return await mlflow.list_experiment_ids()


# ---------------------------------------------------------------------------
# List / search datasets
# ---------------------------------------------------------------------------


@router.get(
    "",
    operation_id="list_datasets",
    summary="List datasets, optionally filtered by name or topic.",
)
async def list_datasets(
    search: str = Query("", description="Filter by name or topic (substring)"),
) -> list[dict[str, Any]]:
    mlflow = _mlflow_client()
    exp_ids = await _get_all_experiment_ids(mlflow)
    if not exp_ids:
        return []
    sdg_runs = await mlflow.search_runs(
        exp_ids,
        filter_string="tags.job_type = 'sdg' AND attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=200,
    )
    upload_runs = await mlflow.search_runs(
        exp_ids,
        filter_string="tags.source = 'dataset' AND attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=200,
    )
    seen: set[str] = set()
    runs: list[dict[str, Any]] = []
    for r in sdg_runs + upload_runs:
        rid = r.get("info", {}).get("run_id", "")
        if rid and rid not in seen:
            seen.add(rid)
            runs.append(r)
    runs.sort(key=lambda r: r.get("info", {}).get("start_time", 0), reverse=True)
    results = [_run_to_summary(r) for r in runs]
    if search:
        q = search.lower()
        results = [d for d in results if q in d["name"].lower() or q in d["topic"].lower()]
    return results


# ---------------------------------------------------------------------------
# Get dataset detail
# ---------------------------------------------------------------------------


@router.get(
    "/{run_id}",
    operation_id="get_dataset",
    summary="Get full metadata and artifact list for a dataset.",
)
async def get_dataset(run_id: str) -> dict[str, Any]:
    mlflow = _mlflow_client()
    try:
        run = await mlflow.get_run(run_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Dataset not found") from None
        raise
    summary = _run_to_summary(run)
    artifacts = await mlflow.list_artifacts(run_id, "generated_data")
    summary["artifacts"] = [
        {"path": a.get("path", ""), "file_size": a.get("file_size", 0)} for a in artifacts
    ]
    return summary


# ---------------------------------------------------------------------------
# Get dataset samples
# ---------------------------------------------------------------------------


@router.get(
    "/{run_id}/samples",
    operation_id="get_dataset_samples",
    summary="Preview rows from a dataset (parquet or JSONL).",
)
async def get_dataset_samples(
    run_id: str,
    limit: int = Query(5, ge=1, le=50, description="Max rows to return"),
) -> list[dict[str, Any]]:
    mlflow = _mlflow_client()
    artifacts = await mlflow.list_artifacts(run_id, "generated_data")
    if not artifacts:
        raise HTTPException(status_code=404, detail="No artifacts found")

    parquet = next((a for a in artifacts if a.get("path", "").endswith(".parquet")), None)
    jsonl = next((a for a in artifacts if a.get("path", "").endswith(".jsonl")), None)
    target = parquet or jsonl
    if not target:
        raise HTTPException(status_code=404, detail="No parquet or JSONL artifact found")

    path = target["path"]

    try:
        data = await mlflow.get_artifact(run_id, path)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {path}") from None
        logger.warning("MLflow get_artifact failed for run %s path %s: %s", run_id, path, exc)
        raise HTTPException(
            status_code=502, detail=f"Failed to read artifact from MLflow: {exc}"
        ) from None
    except httpx.ConnectError as exc:
        logger.warning("MLflow connection failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Cannot connect to MLflow: {exc}") from None
    except httpx.TimeoutException as exc:
        logger.warning("MLflow request timed out: %s", exc)
        raise HTTPException(status_code=504, detail=f"MLflow request timed out: {exc}") from None

    try:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            table = pq.read_table(io.BytesIO(data))  # type: ignore[no-untyped-call]
            records: list[dict[str, Any]] = table.slice(0, limit).to_pylist()
        else:
            lines = data.decode("utf-8").strip().split("\n")
            records = [json.loads(line) for line in lines[:limit]]
    except Exception as exc:
        logger.warning("Failed to parse artifact %s: %s", path, exc)
        raise HTTPException(
            status_code=422, detail=f"Failed to parse dataset artifact: {exc}"
        ) from None

    return records
