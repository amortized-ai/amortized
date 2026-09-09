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
    EvalJobRequest,
    Job,
    JobStatus,
    JobType,
    SDGJobRequest,
    ServeJobRequest,
    TrainingJobRequest,
    ValidatedJobConfig,
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


def _strip_eval_api_keys(job: Job) -> None:
    """Remove endpoint API keys from a job response before returning it."""
    if not isinstance(job.config, dict):
        return
    for key in ("endpoint", "endpoint_base", "endpoint_tuned", "judge"):
        endpoint = job.config.get(key)
        if isinstance(endpoint, dict):
            endpoint.pop("api_key", None)


async def _validate_eval_data(
    config: dict[str, Any],
    parent_job_id: str,
    db: asyncpg.Connection,
) -> list[str]:
    """Validate that eval data is available (via parent job or MLflow run)."""
    errors: list[str] = []
    data_run_id = config.get("eval_data_run_id", "")

    if not parent_job_id and not data_run_id:
        errors.append(
            "eval jobs require either parent_job_id (a completed SDG or upload"
            " job holding the dataset) or eval_data_run_id (an MLflow run ID,"
            " e.g. from a dataset uploaded via /api/v1/datasets)"
        )
        return errors

    if parent_job_id:
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


async def _validate_serve_model(config: dict[str, Any]) -> list[str]:
    """Validate that a servable model source is configured."""
    errors: list[str] = []
    training_job_id = str(config.get("training_job_id", "")).strip()

    if not training_job_id and not str(config.get("model_name_or_path", "")).strip():
        errors.append(
            "serve jobs require either training_job_id (a succeeded training"
            " job whose tuned model to serve) or model_name_or_path"
            " (an HF model id or local path)"
        )
    return errors


# ---------------------------------------------------------------------------
# Job creation endpoints (one per job type)
# ---------------------------------------------------------------------------


@router.post(
    "/sdg",
    status_code=201,
    response_model=Job,
    operation_id="create_sdg_job",
    summary="Create and submit an SDG job. Called by the frontend on user confirmation.",
)
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
    summary="Create and submit a training job. Called by the frontend on user confirmation.",
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

    # Serve pods pin to the user's GPU instead of requesting
    # nvidia.com/gpu, so the quota only reflects training. Keep the
    # one-GPU budget honest: training must wait until the user's serve
    # deployments are stopped.
    from amortized import config as config_module
    from amortized.models import JobStatus, JobType

    if config_module.settings.compute_backend == "kubernetes":
        running_serves = await db.fetch(
            """SELECT id FROM jobs
               WHERE type = $1 AND status = $2 AND k8s_namespace = $3""",
            JobType.serve.value,
            JobStatus.running.value,
            config_module.settings.compute_namespace,
        )
        if running_serves:
            raise HTTPException(
                status_code=409,
                detail=(
                    "You have {} running serve job(s) sharing your GPU budget"
                    " ({}). Stop them from the Jobs page before training."
                    " Serve jobs don't count against the GPU quota, so this"
                    " check keeps your budget at one GPU.".format(
                        len(running_serves),
                        ", ".join(sorted(r["id"][:8] for r in running_serves)),
                    )
                ),
            )

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


@router.post(
    "/eval",
    status_code=201,
    response_model=Job,
    operation_id="create_eval_job",
    summary=(
        "Create and submit an eval job. Compares a base model endpoint against"
        " a tuned model endpoint on an eval dataset."
    ),
)
async def create_eval_job(
    request: EvalJobRequest,
    http_request: Request,
    db: asyncpg.Connection = Depends(_get_db),
) -> Job:
    """Create a model evaluation job."""
    config = request.model_dump(exclude_none=True, exclude_unset=True)
    parent_job_id = config.pop("parent_job_id", "")

    errors = await _validate_eval_data(config, parent_job_id, db)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    user_id = http_request.headers.get("X-Forwarded-User", "")

    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=JobType.eval,
            config=config,
            parent_job_id=parent_job_id,
            user_id=user_id,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = _job_response(row)
    _strip_eval_api_keys(response)
    return response


@router.post(
    "/serve",
    status_code=201,
    response_model=Job,
    operation_id="create_serve_job",
    summary=(
        "Create and submit a serve job — brings up a persistent vLLM inference"
        " endpoint for a tuned model (from a training job) or any HF model."
        " Runs until cancelled."
    ),
)
async def create_serve_job(
    request: ServeJobRequest,
    http_request: Request,
    db: asyncpg.Connection = Depends(_get_db),
) -> Job:
    """Create a model serving job."""
    config = request.model_dump(exclude_none=True, exclude_unset=True)
    parent_job_id = config.pop("parent_job_id", "")
    if parent_job_id and not config.get("training_job_id"):
        config["training_job_id"] = parent_job_id
    # Lineage: serve jobs are children of the training job they serve
    parent_job_id = str(config.get("training_job_id", ""))

    errors = await _validate_serve_model(config)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    user_id = http_request.headers.get("X-Forwarded-User", "")

    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=JobType.serve,
            config=config,
            parent_job_id=parent_job_id,
            user_id=user_id,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _job_response(row)


# ---------------------------------------------------------------------------
# Job validation endpoints (MCP-facing, no DB insert)
# ---------------------------------------------------------------------------


@router.post(
    "/serve/validate",
    response_model=ValidatedJobConfig,
    operation_id="validate_serve_job",
    summary=(
        "Validate a serve job config and present it for user confirmation."
        " Set training_job_id to serve a completed training job's tuned model,"
        " or model_name_or_path to serve any HF model."
    ),
)
async def validate_serve_job(request: ServeJobRequest) -> ValidatedJobConfig:
    """Validate a serve job config without creating it."""
    config = request.model_dump(exclude_none=True, exclude_unset=True)
    parent_job_id = config.pop("parent_job_id", "")
    if parent_job_id and not config.get("training_job_id"):
        config["training_job_id"] = parent_job_id
        parent_job_id = ""

    errors = await _validate_serve_model(config)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    return ValidatedJobConfig(
        job_type=JobType.serve,
        config=config,
        parent_job_id="",
    )


@router.post(
    "/sdg/validate",
    response_model=ValidatedJobConfig,
    operation_id="validate_sdg_job",
    summary=(
        "Validate an SDG job config and present it for user confirmation. "
        "The UI renders a confirmation card — the user clicks confirm to submit. "
        "Use mode 'preview' for a ~10 sample test run first, then 'create' for the full run."
    ),
)
async def validate_sdg_job(request: SDGJobRequest) -> ValidatedJobConfig:
    config = request.model_dump(exclude_none=True)
    parent_job_id = config.pop("parent_job_id", "")
    return ValidatedJobConfig(
        job_type=JobType.sdg,
        config=config,
        parent_job_id=parent_job_id,
    )


@router.post(
    "/training/validate",
    response_model=ValidatedJobConfig,
    operation_id="validate_training_job",
    summary=(
        "Validate a training job config and present it for user confirmation. "
        "The UI renders a confirmation card with ALL parameters — do NOT output "
        "a markdown table, parameter list, or config summary before or after this "
        "tool call. Write ONE short sentence, then call this tool. "
        "Set parent_job_id to chain from a completed SDG job."
    ),
)
async def validate_training_job(
    request: TrainingJobRequest,
    db: asyncpg.Connection = Depends(_get_db),
) -> ValidatedJobConfig:
    """Validate a training job config without creating it."""
    config = request.model_dump(exclude_none=True, exclude_unset=True)
    parent_job_id = config.pop("parent_job_id", "")

    errors = await _validate_training_data(config, parent_job_id, db)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    return ValidatedJobConfig(
        job_type=JobType.training,
        config=config,
        parent_job_id=parent_job_id,
    )


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------


@router.post(
    "/eval/validate",
    response_model=ValidatedJobConfig,
    operation_id="validate_eval_job",
    summary=(
        "Validate an eval job config and present it for user confirmation."
        " Set parent_job_id to chain from a completed SDG job, or"
        " eval_data_run_id to use an uploaded dataset."
    ),
)
async def validate_eval_job(
    request: EvalJobRequest,
    db: asyncpg.Connection = Depends(_get_db),
) -> ValidatedJobConfig:
    """Validate an eval job config without creating it."""
    config = request.model_dump(exclude_none=True, exclude_unset=True)
    parent_job_id = config.pop("parent_job_id", "")

    errors = await _validate_eval_data(config, parent_job_id, db)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    return ValidatedJobConfig(
        job_type=JobType.eval,
        config=config,
        parent_job_id=parent_job_id,
    )


@router.get(
    "",
    response_model=list[Job],
    operation_id="list_jobs",
    summary="List all jobs, optionally filtered by status or type (sdg, training).",
)
async def get_jobs(
    status: JobStatus | None = None,
    type: JobType | None = None,
    db: asyncpg.Connection = Depends(_get_db),
) -> list[Job]:
    from amortized.config import settings as _settings

    repo = Repository(db)
    rows = await core_list_jobs(
        repo,
        status=status,
        job_type=type,
        k8s_namespace=_settings.compute_namespace,
    )
    return [_job_response(row) for row in rows]


@router.get(
    "/{job_id}",
    response_model=Job,
    operation_id="get_job",
    summary=(
        "Get full job details including status, config, timestamps, and MLflow "
        "run ID. Use to check job status or inspect configuration."
    ),
)
async def get_job_detail(
    job_id: str,
    db: asyncpg.Connection = Depends(_get_db),
) -> Job:
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_response(row)


@router.delete(
    "/{job_id}",
    response_model=Job,
    operation_id="cancel_job",
    summary="Cancel a running job. Only works on jobs with status 'pending' or 'running'.",
)
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


@router.get(
    "/{job_id}/logs",
    operation_id="get_job_logs",
    summary=(
        "Get container logs for a job. Use to diagnose failed jobs. "
        "Returns the last N lines (default 100)."
    ),
)
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


@router.get(
    "/{job_id}/artifacts",
    operation_id="get_job_artifacts",
    summary=(
        "Get the MLflow artifact URI for a completed job. Use to locate "
        "training outputs or SDG datasets in the artifact store."
    ),
)
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


@router.get(
    "/{job_id}/eval-results",
    operation_id="get_eval_results",
    summary=(
        "Get aggregate eval metrics for a completed eval job: per-model"
        " exact_match/format_validity/error rates and the judge win-rate."
    ),
)
async def get_eval_results(
    job_id: str,
    db: asyncpg.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Return the parsed eval_results/metrics.json for an eval job."""
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if row.get("type") != "eval":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not an eval job")

    mlflow_run_id = row.get("mlflow_run_id", "")
    if row.get("status") != "succeeded" or not mlflow_run_id:
        return {
            "job_id": job_id,
            "results": None,
            "message": f"Eval results not available yet (status: {row.get('status')})",
        }

    from amortized.config import settings as _settings

    if not _settings.mlflow_tracking_uri:
        raise HTTPException(status_code=503, detail="MLflow tracking URI not configured")

    import json as _json

    from amortized.core.mlflow_client import MLflowClient

    client = MLflowClient(_settings.mlflow_tracking_uri)
    metrics_text = await client.get_artifact_text(mlflow_run_id, "eval_results/metrics.json")
    if not metrics_text:
        return {
            "job_id": job_id,
            "results": None,
            "message": "No eval_results/metrics.json artifact on the MLflow run",
        }
    try:
        results = _json.loads(metrics_text).get("results", {})
    except ValueError:
        raise HTTPException(
            status_code=502, detail="Corrupt metrics.json artifact on the MLflow run"
        ) from None

    return {"job_id": job_id, "mlflow_run_id": mlflow_run_id, "results": results}


@router.post(
    "/{job_id}/delete",
    status_code=204,
    operation_id="delete_job",
    summary="Permanently delete a job record. Only works on cancelled or failed jobs.",
)
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
