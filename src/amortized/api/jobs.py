"""Job management endpoints — unified CRUD."""

import logging
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from amortized.core.compute import get_backend
from amortized.core.jobs import (
    InvalidJobStateError,
    JobNotFoundError,
    _deserialize_handle,
)
from amortized.core.jobs import (
    cancel_job as core_cancel_job,
)
from amortized.core.jobs import (
    create_job as core_create_job,
)
from amortized.core.jobs import (
    get_job as core_get_job,
)
from amortized.core.jobs import (
    list_jobs as core_list_jobs,
)
from amortized.core.redact import redact_config
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import (
    DryRunResponse,
    EvalJobConfig,
    Job,
    JobRequest,
    JobStatus,
    JobType,
    SynthJobConfig,
    TrainingJobConfig,
)
from amortized.worker import _resolve_mlflow_artifact_uri

logger = logging.getLogger("amortized.api.jobs")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

_SENSITIVE_CONFIG_KEYS = frozenset({"api_key", "api_secret", "token", "password"})


def _job_response(row: dict[str, Any]) -> Job:
    row["config"] = redact_config(row["config"])
    return Job(**row)


def _validate_config(job_type: JobType, config: dict[str, Any]) -> list[str]:
    try:
        if job_type == JobType.training:
            TrainingJobConfig(**config)
        elif job_type == JobType.sdg:
            SynthJobConfig(**config)
        elif job_type == JobType.eval:
            EvalJobConfig(**config)
    except ValidationError as exc:
        return [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
    return []


def _strip_secrets(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Remove sensitive keys from config, return (clean_config, secrets)."""
    clean = {}
    secrets: dict[str, str] = {}
    for k, v in config.items():
        if k in _SENSITIVE_CONFIG_KEYS and isinstance(v, str) and v:
            secrets[k] = v
        else:
            clean[k] = v
    return clean, secrets


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
            config=redact_config(request.config),
        )
        return JSONResponse(content=dry_resp.model_dump(), status_code=200)

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    clean_config, secrets = _strip_secrets(request.config)

    user_id = http_request.headers.get("X-Forwarded-User", "")

    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=job_type,
            config=clean_config,
            recipe=request.recipe,
            parent_job_id=request.parent_job_id,
            user_id=user_id,
            secrets=secrets,
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

    handle = _deserialize_handle(row.get("backend_handle"))
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
