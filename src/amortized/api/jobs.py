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
    SDGJobRequest,
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
        elif job_type == JobType.upload:
            errors: list[str] = []
            if not config.get("s3_uri"):
                errors.append("s3_uri is required for upload jobs")
            if not config.get("filename"):
                errors.append("filename is required for upload jobs")
            return errors
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
    exc: ValidationError,
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
        loc = err.get("loc", ())
        if len(loc) >= 3 and loc[0] == "columns" and isinstance(loc[1], int):
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
            path = ".".join(str(part) for part in err.get("loc", ()))
            simplified.append({"field": path, "error": err["msg"]})

    return simplified


@router.post("/sdg", status_code=201, response_model=Job, operation_id="create_sdg_job")
async def create_sdg_job(
    http_request: Request,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    """Create a synthetic data generation job using Data Designer."""
    body = await http_request.json()
    try:
        request = SDGJobRequest(**body)
    except ValidationError as exc:
        errors = _simplify_sdg_errors(exc, body)
        raise HTTPException(status_code=422, detail=errors) from exc

    config = request.model_dump(exclude_none=True)

    mode = config.pop("mode", "create")
    document_ids = config.pop("document_ids", [])
    parent_job_id = config.pop("parent_job_id", "")
    topic = config.get("topic", "")

    config["document_ids"] = document_ids
    if topic:
        config["topic"] = topic
    if mode != "create":
        config["mode"] = mode

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
