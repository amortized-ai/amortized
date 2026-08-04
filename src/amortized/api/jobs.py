"""Job management endpoints — unified CRUD."""

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from amortized.api.costs import _load_pricing_data
from amortized.core.compute import get_backend
from amortized.core.jobs import (
    InvalidJobStateError,
    JobNotFoundError,
    deserialize_handle,
)
from amortized.core.jobs import (
    cancel_job as core_cancel_job,
)
from amortized.core.jobs import (
    create_job as core_create_job,
)
from amortized.core.jobs import (
    delete_job as core_delete_job,
)
from amortized.core.jobs import (
    get_job as core_get_job,
)
from amortized.core.jobs import (
    list_jobs as core_list_jobs,
)
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import (
    DryRunResponse,
    Job,
    JobRequest,
    JobStatus,
    JobType,
    TrainingJobConfig,
)
from amortized.worker import _resolve_mlflow_artifact_uri

logger = logging.getLogger("amortized.api.jobs")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def _job_response(row: dict[str, Any]) -> Job:
    return Job(**row)


_KNOWN_COLUMN_TYPES = frozenset(
    {
        "sampler",
        "llm-text",
        "llm-code",
        "llm-structured",
        "llm-judge",
        "validation",
        "expression",
        "custom",
        "seed-dataset",
        "embedding",
        "image",
    }
)

_LLM_COLUMN_TYPES = frozenset({"llm-text", "llm-code", "llm-structured", "llm-judge"})


def _validate_sdg_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    columns = config.get("columns")
    if columns is None:
        return ["columns: Data Designer config requires a 'columns' field"]
    if not isinstance(columns, list) or len(columns) == 0:
        return ["columns: must be a non-empty list"]

    model_aliases_needed: list[str] = []

    for i, col in enumerate(columns):
        prefix = f"columns[{i}]"
        col_type = col.get("column_type")
        if not col_type:
            errors.append(f"{prefix}.column_type: required")
            continue
        if col_type not in _KNOWN_COLUMN_TYPES:
            errors.append(
                f"{prefix}.column_type: unknown type '{col_type}'"
                f" (valid: {', '.join(sorted(_KNOWN_COLUMN_TYPES))})"
            )
            continue
        if not col.get("name"):
            errors.append(f"{prefix}.name: required")

        if col_type == "sampler":
            if not col.get("sampler_type"):
                errors.append(f"{prefix}.sampler_type: required for sampler columns")
            sampler_type = col.get("sampler_type", "")
            if sampler_type in ("category", "subcategory"):
                params = col.get("params", {})
                values = params.get("values") if isinstance(params, dict) else None
                if not values or not isinstance(values, list) or len(values) == 0:
                    errors.append(f"{prefix}.params.values: required for {sampler_type} samplers")
        elif col_type in _LLM_COLUMN_TYPES:
            alias = col.get("model_alias")
            if not alias:
                errors.append(f"{prefix}.model_alias: required for {col_type} columns")
            else:
                model_aliases_needed.append(alias)
            if not col.get("system_prompt"):
                errors.append(f"{prefix}.system_prompt: required for {col_type} columns")
            if not col.get("prompt"):
                errors.append(f"{prefix}.prompt: required for {col_type} columns")

    if model_aliases_needed:
        model_configs = config.get("model_configs")
        if not model_configs or not isinstance(model_configs, list):
            errors.append("model_configs: required when columns use llm-text")
        else:
            for j, mc in enumerate(model_configs):
                if not mc.get("alias"):
                    errors.append(f"model_configs[{j}].alias: required")
                if not mc.get("model"):
                    errors.append(f"model_configs[{j}].model: required")
            defined_aliases = {mc.get("alias") for mc in model_configs if mc.get("alias")}
            for alias in model_aliases_needed:
                if alias not in defined_aliases:
                    errors.append(
                        f"model_alias: '{alias}' not found in model_configs "
                        f"(available: {', '.join(sorted(defined_aliases)) or 'none'})"
                    )

    return errors


def _validate_config(job_type: JobType, config: dict[str, Any]) -> list[str]:
    try:
        if job_type == JobType.training:
            TrainingJobConfig(**config)
        elif job_type == JobType.sdg:
            return _validate_sdg_config(config)
    except ValidationError as exc:
        return [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
    return []


@router.post("", status_code=201, response_model=Job, operation_id="create_job")
async def create_job(
    request: JobRequest,
    http_request: Request,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job | JSONResponse:
    job_type = request.type

    errors = _validate_config(job_type, request.config)

    if request.dry_run:
        dry_resp = DryRunResponse(
            valid=not errors,
            errors=errors,
            warnings=[],
            type=job_type.value,
            config=request.config,
        )
        return JSONResponse(content=dry_resp.model_dump(), status_code=200)

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    user_id = http_request.headers.get("X-Forwarded-User", "")

    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=job_type,
            config=request.config,
            recipe=request.recipe,
            parent_job_id=request.parent_job_id,
            user_id=user_id,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _job_response(row)


@router.get("", response_model=list[Job], operation_id="list_jobs")
async def get_jobs(
    status: JobStatus | None = None,
    type: JobType | None = None,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Job]:
    repo = Repository(db)
    rows = await core_list_jobs(repo, status=status, job_type=type)
    return [_job_response(row) for row in rows]


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

_GPU_RATE_DEFAULT = 2.50
_TOKENS_PER_SAMPLE = 2000
_FALLBACK_TOKEN_RATE = 3.0  # $ per 1M tokens


class DailyCost(BaseModel):
    date: str
    training: float
    sdg: float


class CostBreakdownResponse(BaseModel):
    total_cost: float
    training_cost: float
    sdg_cost: float
    daily: list[DailyCost]
    gpu_rate_per_hour: float
    currency: str


def _estimate_training_cost(
    config: dict[str, Any],
    started_at: str,
    completed_at: str,
    gpu_rate: float,
) -> float:
    try:
        t_start = datetime.fromisoformat(started_at)
        t_end = datetime.fromisoformat(completed_at)
        duration_hours = max((t_end - t_start).total_seconds() / 3600, 0)
    except (ValueError, TypeError):
        return 0.0

    compute = config.get("compute", {})
    gpu_count = compute.get("gpus", 1) if isinstance(compute, dict) else 1
    return duration_hours * gpu_count * gpu_rate


def _estimate_sdg_cost(
    config: dict[str, Any],
    pricing_data: list[dict[str, Any]],
) -> float:
    model_name = config.get("model") or config.get("teacher_model") or ""
    num_samples = int(config.get("num_samples", 100))
    total_tokens = num_samples * _TOKENS_PER_SAMPLE

    prompt_rate = _FALLBACK_TOKEN_RATE
    completion_rate = _FALLBACK_TOKEN_RATE

    if model_name:
        model_lower = model_name.lower()
        for entry in pricing_data:
            entry_id = entry.get("id", "").lower()
            entry_name = entry.get("name", "").lower()
            if model_lower in entry_id or model_lower in entry_name:
                prompt_rate = float(entry.get("prompt_cost_per_1m", _FALLBACK_TOKEN_RATE))
                completion_rate = float(entry.get("completion_cost_per_1m", _FALLBACK_TOKEN_RATE))
                break

    return total_tokens * (prompt_rate + completion_rate) / 2_000_000


@router.get(
    "/cost-breakdown",
    response_model=CostBreakdownResponse,
    operation_id="get_job_cost_breakdown",
)
async def get_job_cost_breakdown(
    time_range: str = Query("30d", alias="range", description="Time window: 7d, 30d, or 90d"),
    db: aiosqlite.Connection = Depends(_get_db),
) -> CostBreakdownResponse:
    num_days = {"7d": 7, "30d": 30, "90d": 90}.get(time_range, 30)
    cutoff = (datetime.now(tz=UTC) - timedelta(days=num_days)).isoformat()
    gpu_rate = float(os.environ.get("AMORTIZED_GPU_RATE", _GPU_RATE_DEFAULT))

    cursor = await db.execute(
        "SELECT type, config, started_at, completed_at "
        "FROM jobs WHERE status = 'succeeded' AND completed_at > ? "
        "ORDER BY completed_at",
        (cutoff,),
    )
    rows = await cursor.fetchall()

    pricing_data = _load_pricing_data()

    daily_map: dict[str, dict[str, float]] = {}
    training_total = 0.0
    sdg_total = 0.0

    for row in rows:
        job_type = row[0]
        try:
            config = json.loads(row[1]) if isinstance(row[1], str) else row[1]
        except (json.JSONDecodeError, TypeError):
            config = {}
        started_at = row[2] or ""
        completed_at = row[3] or ""

        day = completed_at[:10] if completed_at else ""
        if not day:
            continue

        if day not in daily_map:
            daily_map[day] = {"training": 0.0, "sdg": 0.0}

        if job_type == "training":
            cost = _estimate_training_cost(config, started_at, completed_at, gpu_rate)
            daily_map[day]["training"] += cost
            training_total += cost
        elif job_type == "sdg":
            cost = _estimate_sdg_cost(config, pricing_data)
            daily_map[day]["sdg"] += cost
            sdg_total += cost

    today = datetime.now(tz=UTC).date()
    daily: list[DailyCost] = []
    for i in range(num_days - 1, -1, -1):
        d = today - timedelta(days=i)
        date_str = d.isoformat()
        entry = daily_map.get(date_str, {"training": 0.0, "sdg": 0.0})
        daily.append(
            DailyCost(
                date=date_str,
                training=round(entry["training"], 4),
                sdg=round(entry["sdg"], 4),
            )
        )

    return CostBreakdownResponse(
        total_cost=round(training_total + sdg_total, 4),
        training_cost=round(training_total, 4),
        sdg_cost=round(sdg_total, 4),
        daily=daily,
        gpu_rate_per_hour=gpu_rate,
        currency="USD",
    )


@router.get("/{job_id}", response_model=Job, operation_id="get_job")
async def get_job_detail(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_response(row)


@router.delete("/{job_id}", response_model=Job, operation_id="cancel_job")
async def cancel_job(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    repo = Repository(db)
    try:
        row = await core_cancel_job(repo, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from exc
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _job_response(row)


@router.get("/{job_id}/logs", operation_id="get_job_logs")
async def get_job_logs(
    job_id: str,
    tail: int = 100,
    db: aiosqlite.Connection = Depends(_get_db),
) -> dict[str, Any]:
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    handle = deserialize_handle(row.get("backend_handle"))
    if handle is None:
        msg = "No backend handle — job may not have started"
        return {"job_id": job_id, "logs": [], "message": msg}

    try:
        backend = get_backend(handle.backend_name)
    except KeyError:
        msg = f"Backend {handle.backend_name!r} not available"
        return {"job_id": job_id, "logs": [], "message": msg}

    lines: list[str] = []
    try:
        async for line in backend.logs(handle):
            lines.append(line)
            if len(lines) > tail:
                lines = lines[-tail:]
    except Exception as exc:
        logger.warning("Failed to fetch logs for job %s: %s", job_id, exc)
        return {"job_id": job_id, "logs": [], "message": str(exc)}

    return {"job_id": job_id, "logs": lines}


@router.get("/{job_id}/artifacts", operation_id="get_job_artifacts")
async def get_job_artifacts(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Return MLflow artifact URI for a completed job."""
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    mlflow_run_id = row.get("mlflow_run_id", "")
    if not mlflow_run_id:
        return {
            "job_id": job_id,
            "artifact_uri": "",
            "message": "No MLflow run ID — job may not have completed",
        }

    artifact_uri = await _resolve_mlflow_artifact_uri(mlflow_run_id)
    return {
        "job_id": job_id,
        "mlflow_run_id": mlflow_run_id,
        "artifact_uri": artifact_uri,
    }


@router.post("/{job_id}/delete", status_code=204, operation_id="delete_job")
async def delete_job(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> None:
    repo = Repository(db)
    try:
        await core_delete_job(repo, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from exc
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
