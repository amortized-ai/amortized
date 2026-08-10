"""Job management endpoints."""

import logging
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

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
    Job,
    JobStatus,
    JobType,
    SDGJobRequest,
    TrainingJobRequest,
)
from amortized.worker import _resolve_mlflow_artifact_uri

logger = logging.getLogger("amortized.api.jobs")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def _job_response(row: dict[str, Any]) -> Job:
    return Job(**row)


# ---------------------------------------------------------------------------
# SDG error simplification
# ---------------------------------------------------------------------------

_COLUMN_TYPE_TO_CLASS = {
    "sampler": "SamplerColumnConfig",
    "llm-text": "LLMTextColumnConfig",
    "llm-code": "LLMCodeColumnConfig",
    "llm-judge": "LLMJudgeColumnConfig",
    "llm-structured": "LLMStructuredColumnConfig",
    "expression": "ExpressionColumnConfig",
    "validation": "ValidationColumnConfig",
    "seed-dataset": "SeedDatasetColumnConfig",
    "embedding": "EmbeddingColumnConfig",
    "image": "ImageColumnConfig",
    "custom": "CustomColumnConfig",
}


def _simplify_sdg_errors(
    exc: ValidationError | RequestValidationError,
    body: dict[str, Any],
) -> list[dict[str, str]]:
    """Filter Pydantic union errors to only the matching column type."""
    columns = body.get("columns", [])
    valid_types = sorted(_COLUMN_TYPE_TO_CLASS.keys())
    simplified: list[dict[str, str]] = []

    bad_col_types: list[tuple[int, str]] = []
    if isinstance(columns, list):
        for i, col in enumerate(columns):
            if isinstance(col, dict):
                ct = col.get("column_type", "")
                if ct and ct not in _COLUMN_TYPE_TO_CLASS:
                    bad_col_types.append((i, ct))
    if bad_col_types:
        for idx, ct in bad_col_types:
            simplified.append(
                {
                    "field": f"columns[{idx}].column_type",
                    "error": f"'{ct}' is not valid. Valid types: {valid_types}",
                }
            )
        return simplified

    for err in exc.errors():
        loc = tuple(err.get("loc", ()))
        if loc and loc[0] == "body":
            loc = loc[1:]

        if len(loc) >= 2 and loc[0] == "columns" and isinstance(loc[1], int):
            idx = loc[1]
            col = columns[idx] if idx < len(columns) else {}
            col_type = col.get("column_type", "")
            expected_cls = _COLUMN_TYPE_TO_CLASS.get(col_type, "")
            loc_str = str(loc[2]) if len(loc) > 2 else ""
            if expected_cls and expected_cls not in loc_str:
                continue
            field = str(loc[-1]) if len(loc) > 2 else ""
            path = f"columns[{idx}].{field}" if field else f"columns[{idx}]"
            simplified.append({"field": path, "error": err["msg"]})
        else:
            path = ".".join(str(part) for part in loc)
            simplified.append({"field": path, "error": err["msg"]})

    if not simplified:
        for err in exc.errors()[:5]:
            loc = tuple(err.get("loc", ()))
            if loc and loc[0] == "body":
                loc = loc[1:]
            path = ".".join(str(part) for part in loc)
            simplified.append({"field": path, "error": err["msg"]})

    return simplified


# ---------------------------------------------------------------------------
# Training data validation
# ---------------------------------------------------------------------------


async def _validate_training_data(
    config: dict[str, Any],
    parent_job_id: str,
    db: asyncpg.Connection,
) -> list[str]:
    """Validate that training data is available (via parent job or data_path)."""
    errors: list[str] = []
    data_path = config.get("data_path", "")

    if not parent_job_id and not data_path:
        errors.append(
            "training jobs require either parent_job_id (to chain from an"
            " SDG job) or data_path (direct path to training data)"
        )
        return errors

    if parent_job_id and not data_path:
        repo = Repository(db)
        parent = await repo.get_job(parent_job_id)
        if parent is None:
            errors.append(f"parent_job_id: job '{parent_job_id}' not found")
        elif parent.get("status") != "succeeded":
            errors.append(
                f"parent_job_id: job '{parent_job_id}' has status"
                f" '{parent.get('status')}' (must be 'succeeded')"
            )
        elif not parent.get("mlflow_run_id"):
            errors.append(
                f"parent_job_id: job '{parent_job_id}' has no MLflow"
                " artifacts — the dataset may not have been uploaded"
            )

    return errors


# ---------------------------------------------------------------------------
# Job creation endpoints (one per job type)
# ---------------------------------------------------------------------------


@router.post("/sdg", status_code=201, response_model=Job, operation_id="create_sdg_job")
async def create_sdg_job(
    request: SDGJobRequest,
    http_request: Request,
    db: asyncpg.Connection = Depends(_get_db),
) -> Job:
    """Create a synthetic data generation job using Data Designer."""
    config = request.model_dump(exclude_none=True)
    parent_job_id = config.pop("parent_job_id", "")

    user_id = http_request.headers.get("X-Forwarded-User", "")

    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=JobType.sdg,
            config=config,
            parent_job_id=parent_job_id,
            user_id=user_id,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _job_response(row)


@router.post(
    "/training",
    status_code=201,
    response_model=Job,
    operation_id="create_training_job",
)
async def create_training_job(
    request: TrainingJobRequest,
    http_request: Request,
    db: asyncpg.Connection = Depends(_get_db),
) -> Job:
    """Create a model training job."""
    config = request.model_dump(exclude_none=True, exclude_unset=True)
    parent_job_id = config.pop("parent_job_id", "")

    errors = await _validate_training_data(config, parent_job_id, db)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    user_id = http_request.headers.get("X-Forwarded-User", "")

    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=JobType.training,
            config=config,
            parent_job_id=parent_job_id,
            user_id=user_id,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _job_response(row)


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------


@router.get("", response_model=list[Job], operation_id="list_jobs")
async def get_jobs(
    status: JobStatus | None = None,
    type: JobType | None = None,
    db: asyncpg.Connection = Depends(_get_db),
) -> list[Job]:
    from amortized.config import settings as _settings

    repo = Repository(db)
    rows = await core_list_jobs(
        repo, status=status, job_type=type, k8s_namespace=_settings.compute_namespace,
    )
    return [_job_response(row) for row in rows]


@router.get("/{job_id}", response_model=Job, operation_id="get_job")
async def get_job_detail(
    job_id: str,
    db: asyncpg.Connection = Depends(_get_db),
) -> Job:
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_response(row)


@router.delete("/{job_id}", response_model=Job, operation_id="cancel_job")
async def cancel_job(
    job_id: str,
    db: asyncpg.Connection = Depends(_get_db),
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
    db: asyncpg.Connection = Depends(_get_db),
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
    db: asyncpg.Connection = Depends(_get_db),
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
    db: asyncpg.Connection = Depends(_get_db),
) -> None:
    repo = Repository(db)
    try:
        await core_delete_job(repo, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from exc
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
