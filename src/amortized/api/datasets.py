"""Dataset management — upload, list, inspect, and sample datasets."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite
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
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{mlflow._base}/api/2.0/mlflow/experiments/search",
            json={"max_results": 200},
        )
        resp.raise_for_status()
        return [e["experiment_id"] for e in resp.json().get("experiments", [])]


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
        filter_string="tags.job_type = 'sdg'",
        order_by=["start_time DESC"],
        max_results=200,
    )
    upload_runs = await mlflow.search_runs(
        exp_ids,
        filter_string="tags.job_type = 'upload'",
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
        results = [
            d for d in results
            if q in d["name"].lower() or q in d["topic"].lower()
        ]
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
        {"path": a.get("path", ""), "file_size": a.get("file_size", 0)}
        for a in artifacts
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

    run = await mlflow.get_run(run_id)
    artifact_uri = run.get("info", {}).get("artifact_uri", "")
    if not artifact_uri or not artifact_uri.startswith("s3://"):
        raise HTTPException(status_code=502, detail="Cannot resolve artifact storage")

    from amortized.api.artifacts import _get_s3_client

    parts = artifact_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    s3_key = f"{prefix}/{path}"

    try:
        s3 = _get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        data = obj["Body"].read()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}") from exc

    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(io.BytesIO(data))
        records: list[dict[str, Any]] = table.slice(0, limit).to_pylist()
    else:
        lines = data.decode("utf-8").strip().split("\n")
        records = [json.loads(line) for line in lines[:limit]]

    return records
