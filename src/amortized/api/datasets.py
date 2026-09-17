"""Dataset management — upload, list, inspect, and sample datasets."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
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
from amortized.models import DatasetSplitRequest, Job, JobType

logger = logging.getLogger("amortized.api.datasets")

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

_upload_tasks: set[asyncio.Task[None]] = set()
_upload_semaphore = asyncio.Semaphore(3)

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


def _count_samples(filename: str, file_bytes: bytes) -> int | None:
    try:
        if filename.endswith(".jsonl"):
            return sum(1 for line in file_bytes.split(b"\n") if line.strip())
        if filename.endswith(".parquet"):
            import pyarrow.parquet as pq

            return int(pq.read_metadata(io.BytesIO(file_bytes)).num_rows)  # type: ignore[no-untyped-call]
    except Exception:
        logger.warning("Could not count samples in %s", filename)
    return None


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
        "source": "upload",
    }
    topic = _topic_from_filename(filename)
    if topic:
        tags["dataset_topic"] = topic
    sample_count = _count_samples(filename, file_bytes)
    if sample_count is not None:
        tags["num_samples"] = str(sample_count)

    mlflow = MLflowClient(uri, timeout=60.0)
    experiment_id = await mlflow.ensure_experiment("amortized/datasets")
    run_id = await mlflow.create_run(
        experiment_id,
        name=filename,
        tags=tags,
    )

    try:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jsonl"
        normalized = f"data.{ext}"
        await mlflow.upload_artifact(run_id, f"generated_data/{normalized}", file_bytes)
        await mlflow.finish_run(run_id)
    except Exception:
        await mlflow.fail_run_quiet(run_id)
        raise

    return run_id, experiment_id


def _sanitize_upload_error(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "Cannot connect to MLflow"
    if isinstance(exc, httpx.TimeoutException):
        return "MLflow request timed out"
    if isinstance(exc, httpx.TransportError):
        return "MLflow communication error"
    return str(exc)


async def _process_dataset_upload(
    job_id: str,
    filename: str,
    file_bytes: bytes,
) -> None:
    from amortized.db.connection import get_pool

    async with _upload_semaphore:
        try:
            async with get_pool().acquire() as conn:
                await Repository(conn).update_job(
                    job_id,
                    status="running",
                    started_at=datetime.now(UTC),
                )

            run_id, experiment_id = await _store_dataset_in_mlflow(
                filename,
                file_bytes,
            )

            async with get_pool().acquire() as conn:
                await Repository(conn).update_job(
                    job_id,
                    status="succeeded",
                    mlflow_run_id=run_id,
                    mlflow_experiment=experiment_id,
                    completed_at=datetime.now(UTC),
                )
        except Exception as exc:
            logger.warning("Dataset upload failed for job %s: %s", job_id, exc)
            try:
                async with get_pool().acquire() as conn:
                    await Repository(conn).update_job(
                        job_id,
                        status="failed",
                        completed_at=datetime.now(UTC),
                        error=_sanitize_upload_error(exc),
                    )
            except Exception:
                logger.exception("Failed to mark job %s as failed", job_id)


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
        config={"source": "upload", "original_filename": name},
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
        # The metric set this dataset evaluates with (if any eval has run
        # on it) — reused by the eval agent so scores stay comparable
        # across models and sessions.
        "eval_metric_set": _parse_metric_set(tags.get("eval_metric_set", "")),
    }


def _parse_metric_set(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass
    return None


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
        filter_string=(
            "(tags.source = 'upload' OR tags.source = 'split') AND attributes.status = 'FINISHED'"
        ),
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
    summary=(
        "Preview sample rows from a dataset. Use after SDG job succeeds to "
        "verify data quality before training. Pass the job's mlflow_run_id as run_id."
    ),
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


# ---------------------------------------------------------------------------
# Dataset split — materialize a portion (and complement) of an existing run
# ---------------------------------------------------------------------------


async def _find_dataset_artifact(mlflow: MLflowClient, run_id: str) -> str:
    """The dataset's data artifact path (parquet preferred, else jsonl)."""
    artifacts = await mlflow.list_artifacts(run_id, "generated_data")
    parquet = next((a for a in artifacts if a.get("path", "").endswith(".parquet")), None)
    jsonl = next((a for a in artifacts if a.get("path", "").endswith(".jsonl")), None)
    target = parquet or jsonl
    if not target:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset run {run_id} has no parquet or JSONL artifact",
        )
    return str(target["path"])


def _parse_records(path: str, data: bytes) -> list[dict[str, Any]]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(io.BytesIO(data))  # type: ignore[no-untyped-call]
        return [dict(r) for r in table.to_pylist()]
    lines = data.decode("utf-8").strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def _serialize_records(path: str, records: list[dict[str, Any]]) -> bytes:
    if path.endswith(".parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(records)
        buf = io.BytesIO()
        pq.write_table(table, buf)  # type: ignore[no-untyped-call]
        return buf.getvalue()
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8")


def _split_indices(total: int, request: DatasetSplitRequest) -> list[int]:
    """Row indices for the portion, deterministic for a given request."""
    if request.count and request.fraction:
        raise HTTPException(status_code=422, detail="provide either count or fraction, not both")
    if not request.count and not request.fraction:
        raise HTTPException(status_code=422, detail="provide count or fraction")
    n = request.count or round(total * request.fraction)
    n = max(0, min(n, total))

    if request.strategy == "head":
        return list(range(n))
    if request.strategy == "tail":
        return list(range(total - n, total))
    rng = random.Random(request.seed)
    return sorted(rng.sample(range(total), n))


async def _store_split_run(
    mlflow: MLflowClient,
    name: str,
    source_run: dict[str, Any],
    source_path: str,
    records: list[dict[str, Any]],
    split: dict[str, Any],
) -> str:
    """Materialize records as a new dataset run with lineage tags."""
    source_tags = {t["key"]: t["value"] for t in source_run.get("data", {}).get("tags", [])}
    tags: dict[str, str] = {
        "job_type": "upload",
        "dataset_name": name,
        "source": "split",
        "source_run_id": source_run.get("info", {}).get("run_id", ""),
        "num_samples": str(len(records)),
        "split": json.dumps(split, sort_keys=True),
    }
    if source_tags.get("dataset_topic"):
        tags["dataset_topic"] = source_tags["dataset_topic"]

    experiment_id = await mlflow.ensure_experiment("amortized/datasets")
    run_id = await mlflow.create_run(experiment_id, name=name, tags=tags)
    try:
        ext = source_path.rsplit(".", 1)[-1].lower()
        await mlflow.upload_artifact(
            run_id, f"generated_data/data.{ext}", _serialize_records(source_path, records)
        )
        await mlflow.finish_run(run_id)
    except Exception:
        await mlflow.fail_run_quiet(run_id)
        raise
    return run_id


async def _process_dataset_split(
    job_id: str, source_run_id: str, request: DatasetSplitRequest
) -> None:
    from amortized.db.connection import get_pool

    async with _upload_semaphore:
        try:
            async with get_pool().acquire() as conn:
                await Repository(conn).update_job(
                    job_id, status="running", started_at=datetime.now(UTC)
                )

            mlflow = MLflowClient(settings.mlflow_tracking_uri, timeout=60.0)
            source_run = await mlflow.get_run(source_run_id)
            source_path = await _find_dataset_artifact(mlflow, source_run_id)
            data = await mlflow.get_artifact(source_run_id, source_path)
            records = _parse_records(source_path, data)

            portion_idx = _split_indices(len(records), request)
            complement_idx = [i for i in range(len(records)) if i not in set(portion_idx)]
            portion = [records[i] for i in portion_idx]
            complement = [records[i] for i in complement_idx]

            source_name = (
                {t["key"]: t["value"] for t in source_run.get("data", {}).get("tags", [])}.get(
                    "dataset_name", ""
                )
                or source_run.get("info", {}).get("run_name", "")
                or source_run_id[:8]
            )
            portion_name = request.name or f"{source_name} (split)"
            complement_name = request.complement_name or f"{source_name} (complement)"

            split_meta = {
                "source_run_id": source_run_id,
                "strategy": request.strategy,
                "seed": request.seed,
                "count": request.count,
                "fraction": request.fraction,
            }
            portion_run_id = await _store_split_run(
                mlflow, portion_name, source_run, source_path, portion, split_meta
            )
            complement_run_id = ""
            if request.create_complement:
                complement_run_id = await _store_split_run(
                    mlflow, complement_name, source_run, source_path, complement, split_meta
                )

            async with get_pool().acquire() as conn:
                await Repository(conn).update_job(
                    job_id,
                    status="succeeded",
                    mlflow_run_id=portion_run_id,
                    mlflow_experiment="amortized/datasets",
                    completed_at=datetime.now(UTC),
                    config={
                        "source": "split",
                        "source_run_id": source_run_id,
                        "split_run_id": portion_run_id,
                        "complement_run_id": complement_run_id,
                        "num_portion": len(portion),
                        "num_complement": len(complement),
                    },
                )
        except Exception as exc:
            logger.warning("Dataset split failed for job %s: %s", job_id, exc)
            try:
                async with get_pool().acquire() as conn:
                    await Repository(conn).update_job(
                        job_id,
                        status="failed",
                        completed_at=datetime.now(UTC),
                        error=_sanitize_upload_error(exc),
                    )
            except Exception:
                logger.exception("Failed to mark job %s as failed", job_id)


@router.post("/{run_id}/split", response_model=Job, status_code=202)
async def split_dataset(
    run_id: str,
    request: DatasetSplitRequest,
    db: asyncpg.Connection = Depends(_get_db),
) -> Any:
    """Materialize a portion of a dataset (and optionally its complement).

    Creates a new dataset run holding the selected rows — usable anywhere
    a dataset run id works (eval_data_run_id for evals, data_run_id for
    training). With create_complement (default) this is the two-way
    train/eval split: the portion is typically the eval set and the
    complement the training set. Read both run ids from the finished
    job's config (split_run_id / complement_run_id).
    """
    mlflow = _mlflow_client()
    try:
        await mlflow.get_run(run_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Dataset run {run_id} not found") from None
        raise
    # Fail fast on bad sizes before creating the job.
    await _find_dataset_artifact(mlflow, run_id)
    _split_indices(2, request)  # validates count/fraction exclusivity

    repo = Repository(db)
    row = await create_job(
        repo,
        job_type=JobType.upload,
        config={
            "source": "split",
            "source_run_id": run_id,
            "strategy": request.strategy,
            "seed": request.seed,
            "count": request.count,
            "fraction": request.fraction,
            "create_complement": request.create_complement,
        },
    )
    task = asyncio.create_task(_process_dataset_split(row["id"], run_id, request))
    _upload_tasks.add(task)
    task.add_done_callback(_upload_tasks.discard)

    return row
