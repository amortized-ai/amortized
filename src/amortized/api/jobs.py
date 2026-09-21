"""Job management endpoints."""

import json
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
    """Validate that training data is available (via parent job, run, or path)."""
    errors: list[str] = []
    data_path = config.get("data_path", "")
    data_run_id = str(config.get("data_run_id", "") or "")

    if not parent_job_id and not data_path and not data_run_id:
        errors.append(
            "training jobs require parent_job_id (to chain from an SDG"
            " job), data_run_id (an MLflow run with a generated_data"
            " artifact, e.g. an uploaded dataset or split), or data_path"
            " (direct path to training data)"
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


def _validate_eval_rubric_judge(config: dict[str, Any]) -> list[str]:
    """Fail at the boundary when a rubric has no judge.

    A rubric is scored by an LLM judge, which the caller must set
    explicitly — there is no default. Reject a rubric with no ``judge`` as
    a 422 at validate/create time, not a dispatch-time failure.
    """
    rubric = [c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")]
    if rubric and not config.get("judge"):
        return [
            "a judge endpoint is required to score the rubric — set `judge`"
            " explicitly (there is no default judge)"
        ]
    return []


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


async def _persist_metric_set(
    config: dict[str, Any], parent_job_id: str, db: asyncpg.Connection
) -> None:
    """Tag the dataset run with the eval's metric set (idempotent).

    The tag is the source of truth for "which metrics does this dataset
    evaluate on" — the eval agent reads it in later sessions instead of
    re-designing metrics, keeping comparisons across models consistent.
    """
    dataset_run_id = str(config.get("eval_data_run_id") or "")
    if not dataset_run_id and parent_job_id:
        repo = Repository(db)
        parent = await repo.get_job(parent_job_id)
        if parent:
            dataset_run_id = parent.get("mlflow_run_id", "")
    if not dataset_run_id:
        return

    metric_set = {
        "metrics": [m for m in (config.get("metrics") or []) if m],
        "rubric": [
            {"name": c.get("name"), "description": c.get("description", "")}
            for c in (config.get("rubric") or [])
            if isinstance(c, dict) and c.get("name")
        ],
    }
    if not metric_set["metrics"] and not metric_set["rubric"]:
        return

    import json as _json

    # A soft-deleted dataset run rejects tag writes, which would silently
    # drop the metric set (and later sessions would re-design metrics).
    # The dataset is clearly still in use — the eval references it — so
    # restore the run and keep it visible in the Datasets tab.
    from amortized.config import settings as _settings
    from amortized.core.mlflow_client import MLflowClient
    from amortized.jobs.common import set_mlflow_run_tag

    if _settings.mlflow_tracking_uri:
        client = MLflowClient(_settings.mlflow_tracking_uri)
        try:
            run = await client.get_run(dataset_run_id)
            if run.get("info", {}).get("lifecycle_stage") == "deleted":
                await client.restore_run(dataset_run_id)
                logger.info(
                    "Restored soft-deleted dataset run %s to persist its"
                    " eval metric set", dataset_run_id[:8],
                )
        except Exception:
            logger.warning(
                "Could not check/restore dataset run %s before tagging",
                dataset_run_id[:8],
                exc_info=True,
            )

    await set_mlflow_run_tag(
        dataset_run_id, "eval_metric_set", _json.dumps(metric_set)
    )


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
    errors.extend(_validate_eval_rubric_judge(config))
    if not [c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")]:
        errors.append(
            "eval jobs require at least one rubric criterion (custom"
            " judge-scored metrics — the built-in structural metrics"
            " exact_match/format_validity were removed)"
        )
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    user_id = http_request.headers.get("X-Forwarded-User", "")

    # Persist the metric set on the dataset's MLflow run so every later
    # eval on the same dataset (any session) reuses the same metrics.
    await _persist_metric_set(config, parent_job_id, db)

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


# Runtime-injected keys the eval builder recomputes on every dispatch —
# stripped when a legacy job (no request_config snapshot) is retried.
_RETRY_STRIP_KEYS = ("eval_data_path", "port", "served_model_name")


@router.post(
    "/{job_id}/retry",
    status_code=201,
    response_model=Job,
    operation_id="retry_job",
    summary=(
        "Retry a FAILED eval job by cloning its original request config"
        " verbatim (rubric text included — the new scores land in the same"
        " Evaluation tab comparison group). Only use this to re-run an"
        " eval UNCHANGED; to change anything, assemble a new config"
        " instead. Note: external-endpoint API keys are not retained, so"
        " keyed endpoints need resubmission."
    ),
)
async def retry_job(
    job_id: str,
    http_request: Request,
    db: asyncpg.Connection = Depends(_get_db),
) -> Job:
    """Clone a failed eval job's original config into a new job."""
    repo = Repository(db)
    job = await repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.get("type") != JobType.eval.value:
        raise HTTPException(status_code=422, detail="only eval jobs can be retried")
    if job.get("status") not in (JobStatus.failed.value, JobStatus.cancelled.value):
        raise HTTPException(
            status_code=422,
            detail=(
                f"job status is '{job.get('status')}' — only failed or"
                " cancelled jobs can be retried"
            ),
        )

    # The pre-dispatch snapshot is authoritative; legacy rows (created
    # before the snapshot existed, or snapshotted from a worker-resolved
    # config by the migration) carry runtime-injected keys, so strip
    # them in either path — the builder recomputes all of them.
    snapshot = job.get("request_config")
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except ValueError:
            snapshot = None
    if isinstance(snapshot, dict) and snapshot.get("rubric"):
        config = dict(snapshot)
    else:
        config = dict(job.get("config") or {})
    for key in _RETRY_STRIP_KEYS:
        config.pop(key, None)
    config.pop("parent_job_id", None)

    parent_job_id = job.get("parent_job_id", "")
    errors = await _validate_eval_data(config, parent_job_id, db)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    user_id = http_request.headers.get("X-Forwarded-User", "") or job.get("user_id", "")

    await _persist_metric_set(config, parent_job_id, db)

    try:
        row = await core_create_job(
            repo,
            job_type=JobType.eval,
            config=config,
            recipe=job.get("recipe", ""),
            parent_job_id=parent_job_id,
            user_id=user_id,
            retry_of=job_id,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info("Retrying eval job %s as %s", job_id[:8], row["id"][:8])
    response = _job_response(row)
    _strip_eval_api_keys(response)
    return response


# ---------------------------------------------------------------------------
# Job validation endpoints (MCP-facing, no DB insert)
# ---------------------------------------------------------------------------


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
# Duration stats
# ---------------------------------------------------------------------------


@router.get(
    "/stats/duration",
    operation_id="get_job_duration_stats",
    summary="Average duration in seconds for completed jobs, grouped by type.",
)
async def get_job_duration_stats(
    db: asyncpg.Connection = Depends(_get_db),
) -> dict[str, float]:
    rows = await db.fetch(
        """
        SELECT type,
               EXTRACT(EPOCH FROM AVG(completed_at - started_at)) AS avg_seconds
          FROM jobs
         WHERE status = 'succeeded'
           AND started_at IS NOT NULL
           AND completed_at IS NOT NULL
         GROUP BY type
        """
    )
    return {row["type"]: round(row["avg_seconds"], 1) for row in rows}


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
    errors.extend(_validate_eval_rubric_judge(config))
    if not [c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")]:
        errors.append(
            "eval jobs require at least one rubric criterion (custom"
            " judge-scored metrics — the built-in structural metrics"
            " exact_match/format_validity were removed)"
        )
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    # Paraphrase guardrail: comparison groups key on the exact rubric
    # text, so a re-typed (even slightly reworded) criterion set would
    # fork the dataset's row in the Evaluation tab. Warn — do not
    # block — when the submitted criterion names match an earlier eval
    # on this dataset but the descriptions differ.
    warnings = await _rubric_drift_warning(config, parent_job_id, db)

    return ValidatedJobConfig(
        job_type=JobType.eval,
        config=config,
        parent_job_id=parent_job_id,
        warnings=warnings,
    )


async def _rubric_drift_warning(
    config: dict[str, Any], parent_job_id: str, db: asyncpg.Connection
) -> list[str]:
    """Warn when submitted criteria shadow an earlier eval's rubric.

    Looks at existing eval jobs on the same dataset (same parent job
    or eval_data_run_id) and compares criterion sets: same names with
    different descriptions means the judge will score something
    slightly different, and the new eval will NOT share a comparison
    group with the old ones.
    """
    inline = [c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")]
    if not inline:
        return []

    data_run_id = str(config.get("eval_data_run_id") or "")
    repo = Repository(db)
    candidates = await repo.list_jobs(job_type=JobType.eval)
    submitted = {str(c["name"]): str(c.get("description", "")) for c in inline}
    for job in candidates:
        cfg = job.get("config", {})
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except ValueError:
                continue
        same_dataset = (
            data_run_id
            and str(cfg.get("eval_data_run_id") or "") == data_run_id
        ) or (parent_job_id and job.get("parent_job_id") == parent_job_id)
        if not same_dataset:
            continue
        existing = {
            str(c["name"]): str(c.get("description", ""))
            for c in (cfg.get("rubric") or [])
            if isinstance(c, dict) and c.get("name")
        }
        if not existing or set(existing) != set(submitted):
            continue
        if existing == submitted:
            return []  # exact reuse — nothing to warn about
        changed = sorted(
            n for n in existing if existing[n] != submitted.get(n, existing[n])
        )
        return [
            "this dataset already has evals with these rubric criteria"
            f" but different descriptions (changed: {', '.join(changed)})."
            " Scores will NOT be comparable across the two, and the new"
            " eval gets its own row in the Evaluation tab. Reuse the"
            " earlier criteria verbatim (copy them from the previous"
            " eval's config) unless the user explicitly wants different"
            " criteria."
        ]
    return []


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


# Upper bound on how many log lines a single request may ask for, so a caller
# cannot force an unbounded tail slice / response.
_MAX_LOG_TAIL = 5000


async def _mlflow_log_fallback(mlflow_run_id: str | None, tail: int) -> list[str] | None:
    """Return the tail of the console log persisted to a job's MLflow run
    (``logs/amortized-job.log``, uploaded by worker._wrap_job_logging), or None if
    unavailable. Used when the live pod is gone (K8s TTL cleanup)."""
    if not mlflow_run_id:
        return None
    from amortized.config import settings as _settings
    from amortized.core.mlflow_client import MLflowClient

    if not _settings.mlflow_tracking_uri:
        return None
    try:
        client = MLflowClient(_settings.mlflow_tracking_uri)
        return await client.read_artifact_tail(mlflow_run_id, "logs/amortized-job.log", tail)
    except Exception as exc:
        logger.warning("MLflow log fallback failed for run %s: %s", mlflow_run_id, exc)
        return None


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
    tail = max(1, min(tail, _MAX_LOG_TAIL))
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    lines: list[str] = []
    live_msg: str | None = None
    handle = deserialize_handle(row.get("backend_handle"))
    if handle is None:
        live_msg = "No backend handle — job may not have started"
    else:
        try:
            backend = get_backend(handle.backend_name)
        except KeyError:
            live_msg = f"Backend {handle.backend_name!r} not available"
        else:
            try:
                async for line in backend.logs(handle):
                    lines.append(line)
                    if len(lines) > tail:
                        lines = lines[-tail:]
            except Exception as exc:
                logger.warning("Failed to fetch live logs for job %s: %s", job_id, exc)
                live_msg = str(exc)

    if lines:
        return {"job_id": job_id, "logs": lines}

    # No live logs (the pod was cleaned up after the job finished) — fall back to the
    # console log persisted to the job's MLflow run (worker._wrap_job_logging).
    persisted = await _mlflow_log_fallback(row.get("mlflow_run_id"), tail)
    if persisted is not None:
        return {"job_id": job_id, "logs": persisted, "source": "mlflow"}

    return {"job_id": job_id, "logs": [], "message": live_msg or "No logs available"}


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
        " rubric criterion scores and job-health diagnostics (error and"
        " empty rates)."
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
