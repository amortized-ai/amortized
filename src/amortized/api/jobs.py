"""Job management endpoints."""

import hashlib
import json
import logging
import re
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from amortized.core import mirror
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
    PromptView,
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


async def _require_dependency_succeeded(
    repo: Repository,
    job_id: str,
    label: str,
    *,
    require_mlflow: bool = True,
) -> list[str]:
    """A referenced upstream job must exist, be ``succeeded``, and (optionally)
    have MLflow artifacts. Returns the error(s) for that reference, empty when OK.

    This is the fail-closed half of the dependency invariant: a downstream job
    cannot be created until every upstream job it references is terminal
    ``succeeded``. Shared by training and eval validation so every dependency
    (SDG dataset parent, the model under eval) is gated identically.
    """
    job = await repo.get_job(job_id)
    if job is None:
        return [f"{label}: job '{job_id}' not found"]
    status = job.get("status")
    if status != "succeeded":
        return [
            f"{label}: job '{job_id}' has status '{status}' — it must finish"
            " ('succeeded') before it can be used"
        ]
    if require_mlflow and not job.get("mlflow_run_id"):
        return [
            f"{label}: job '{job_id}' has no MLflow artifacts — it may not"
            " have finished producing its output"
        ]
    return []


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
        errors += await _require_dependency_succeeded(repo, parent_job_id, "parent_job_id")

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
    training_job_id = str(config.get("training_job_id", "") or "")

    if not parent_job_id and not data_run_id:
        errors.append(
            "eval jobs require either parent_job_id (a completed SDG or upload"
            " job holding the dataset) or eval_data_run_id (an MLflow run ID,"
            " e.g. from a dataset uploaded via /api/v1/datasets)"
        )
        return errors

    repo = Repository(db)
    if parent_job_id:
        errors += await _require_dependency_succeeded(repo, parent_job_id, "parent_job_id")

    # The model under eval must itself have finished training — otherwise there is
    # no adapter to score. This is the training -> eval edge of the dependency
    # invariant (the SDG -> {training,eval} edge is parent_job_id above).
    if training_job_id:
        errors += await _require_dependency_succeeded(repo, training_job_id, "training_job_id")

    return errors


_EMBEDDING_ALGORITHMS = {"embedding_sft", "classifier", "embedding"}


async def _apply_classification_eval_mode(
    config: dict[str, Any],
    db: asyncpg.Connection,
) -> list[str]:
    """Keep eval_mode consistent with the model being evaluated.

    An embedding model (trained with embedding_sft) can only be evaluated as a
    classifier, and the classification eval only works on an embedding model.
    When a tuned model is referenced, default/validate eval_mode against the
    training job's algorithm so users don't hit a confusing dispatch-time failure.
    Mutates ``config`` (may set eval_mode) and returns any errors.
    """
    training_job_id = str(config.get("training_job_id", "") or "").strip()
    if not training_job_id:
        return []
    parent = await Repository(db).get_job(training_job_id)
    if not parent or parent.get("type") != "training":
        return []
    parent_config = parent.get("config") or {}
    if isinstance(parent_config, str):
        try:
            parent_config = json.loads(parent_config)
        except (ValueError, TypeError):
            parent_config = {}
    algorithm = str(parent_config.get("algorithm", "")).replace("-", "_")
    is_embedding = algorithm in _EMBEDDING_ALGORITHMS
    eval_mode = config.get("eval_mode")

    if is_embedding:
        if not eval_mode:
            config["eval_mode"] = "classification"  # the only mode that fits
        elif eval_mode != "classification":
            return [
                "training_job_id is an embedding model (embedding_sft); its eval_mode"
                " must be 'classification' (a generative/judge eval cannot score an"
                " embedding model)"
            ]
    elif eval_mode == "classification":
        return [
            "eval_mode='classification' requires an embedding model — train with"
            " algorithm='embedding_sft' (or 'classifier') first"
        ]
    return []


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
    errors.extend(await _apply_classification_eval_mode(config, db))
    errors.extend(_validate_eval_rubric_judge(config))
    if config.get("eval_mode") != "classification" and not [
        c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")
    ]:
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


@router.post(
    "/{job_id}/retry",
    status_code=201,
    response_model=Job,
    operation_id="retry_job",
    summary=(
        "Retry/rerun a FAILED or cancelled job (SDG, training, or eval) by"
        " cloning its original request config. With no overrides it re-runs"
        " the job UNCHANGED (for eval the new scores land in the same"
        " Evaluation tab comparison group); pass `overrides` to change"
        " specific knobs (e.g. num_train_epochs, num_records). The caller"
        " confirms before this creates the job — it is not a silent dispatch."
        " Note: external-endpoint API keys are not retained, so keyed"
        " endpoints need resubmission."
    ),
)
async def retry_job(
    job_id: str,
    http_request: Request,
    db: asyncpg.Connection = Depends(_get_db),
    overrides: dict[str, Any] | None = None,
) -> Job:
    """Clone a failed/cancelled job's original config into a new job of the same type.

    Generalized from the former eval-only path: the shared mirror.clone drops each
    type's runtime/lineage fields from the pre-dispatch snapshot and applies any
    caller `overrides` (the rerun-with-a-tweak case). The frontend gates this behind
    a confirmation step, so re-running an expensive training/SDG job is never a
    single misclick."""
    repo = Repository(db)
    job = await repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job_type = str(job.get("type") or "")
    spec = mirror.MIRROR_SPECS.get(job_type)
    if spec is None:
        raise HTTPException(
            status_code=422, detail=f"job type '{job_type}' cannot be retried"
        )
    if job.get("status") not in (JobStatus.failed.value, JobStatus.cancelled.value):
        raise HTTPException(
            status_code=422,
            detail=(
                f"job status is '{job.get('status')}' — only failed or"
                " cancelled jobs can be retried"
            ),
        )

    # The pre-dispatch snapshot is authoritative; legacy rows (created before the
    # snapshot existed, or snapshotted from a worker-resolved config by the
    # migration) carry runtime-injected keys. mirror.clone strips each type's
    # runtime + lineage fields from whichever source we have — the builder
    # recomputes all of them — and applies the caller's overrides on top.
    snapshot = job.get("request_config")
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except ValueError:
            snapshot = None
    source = snapshot if isinstance(snapshot, dict) and snapshot else job.get("config")
    config = mirror.clone(source, spec, overrides=overrides)

    parent_job_id = job.get("parent_job_id", "")

    # Per-type validation mirrors each create endpoint (SDG has none at this layer).
    if job_type == JobType.training.value:
        errors = await _validate_training_data(config, parent_job_id, db)
        if errors:
            raise HTTPException(status_code=422, detail=errors)
    elif job_type == JobType.eval.value:
        errors = await _validate_eval_data(config, parent_job_id, db)
        if errors:
            raise HTTPException(status_code=422, detail=errors)
        await _persist_metric_set(config, parent_job_id, db)

    user_id = http_request.headers.get("X-Forwarded-User", "") or job.get("user_id", "")

    try:
        row = await core_create_job(
            repo,
            job_type=JobType(job_type),
            config=config,
            recipe=job.get("recipe", ""),
            parent_job_id=parent_job_id,
            user_id=user_id,
            retry_of=job_id,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info("Retrying %s job %s as %s", job_type, job_id[:8], row["id"][:8])
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
    prompts = _recipe_prompts(config)
    assessor = next((p.text for p in prompts if p.role == "assessor"), None)
    return ValidatedJobConfig(
        job_type=JobType.sdg,
        config=config,
        parent_job_id=parent_job_id,
        assessor_prompt=assessor,
        prompts=prompts,
    )


def _assessor_column_name(config: dict[str, Any]) -> str | None:
    """Name of the column whose `system_prompt` is the assessor prompt — the column
    whose output becomes the assistant turn in an SFT processor template (referenced
    via `{{column}}`). Returns None when it can't be identified confidently (no blind
    "last column with a system_prompt" fallback — that risked treating a sampler's
    prompt as the assessor prompt)."""
    columns = config.get("columns")
    if not isinstance(columns, list):
        return None
    by_name = {c.get("name"): c for c in columns if isinstance(c, dict)}
    for proc in config.get("processors") or []:
        if not isinstance(proc, dict):
            continue
        template = proc.get("template")
        messages = template.get("messages") if isinstance(template, dict) else None
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            raw_content = msg.get("content")
            content = raw_content if isinstance(raw_content, str) else ""
            ref = re.search(r"\{\{\s*(\w+)\s*\}\}", content)
            if not ref:
                continue
            col = by_name.get(ref.group(1))
            prompt = col.get("system_prompt") if isinstance(col, dict) else None
            if isinstance(prompt, str) and prompt.strip():
                return ref.group(1)
    return None


def _assessor_prompt(config: dict[str, Any]) -> str | None:
    """The assessor/system prompt the teacher follows, resolved authoritatively from
    the config. Thin wrapper over `_recipe_prompts` kept for existing callers."""
    return next((p.text for p in _recipe_prompts(config) if p.role == "assessor"), None)


def _prompt_label(column: str) -> str:
    """Human card heading for an input-generator column (e.g. 'support_ticket' ->
    'Support ticket prompt')."""
    words = column.replace("_", " ").strip()
    return f"{words[:1].upper()}{words[1:]} prompt" if words else "Input prompt"


def _recipe_prompts(config: dict[str, Any]) -> list[PromptView]:
    """Every reviewable system prompt in an SDG recipe, in generation order (input
    generators first, assessor last), each tagged with its column and role. This is
    what the confirmation card renders, so the user reviews ALL prompts the recipe
    carries — a ticket-generation prompt and an assessor prompt, not just one."""
    columns = config.get("columns")
    if not isinstance(columns, list):
        return []
    assessor = _assessor_column_name(config)
    inputs: list[PromptView] = []
    assessor_view: PromptView | None = None
    for col in columns:
        if not isinstance(col, dict):
            continue
        name = col.get("name")
        prompt = col.get("system_prompt")
        if not isinstance(name, str) or not isinstance(prompt, str) or not prompt.strip():
            continue
        text = prompt.strip()
        if name == assessor:
            assessor_view = PromptView(
                role="assessor", label="Assessor system prompt", column=name, text=text
            )
        else:
            inputs.append(
                PromptView(role="input", label=_prompt_label(name), column=name, text=text)
            )
    return inputs + ([assessor_view] if assessor_view else [])


class CloneSdgForEvalRequest(BaseModel):
    training_job_id: str = Field(
        ...,
        description=(
            "The completed training job the eval set is for. Its parent SDG"
            " recipe (teacher model, prompts, SFT format) is mirrored exactly."
        ),
    )
    num_records: int = Field(
        ...,
        gt=0,
        description="Number of held-out eval records the user asked for.",
    )


class ClonedSdgConfig(BaseModel):
    config: dict[str, Any] = Field(
        description="A ready-to-validate SDG config mirroring the training SDG recipe."
    )
    train_sdg_job_id: str = Field(
        description="The SDG job this config was cloned from (the model's training SDG)."
    )
    note: str


def _coerce_config(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


@router.post(
    "/sdg/clone-for-eval",
    response_model=ClonedSdgConfig,
    operation_id="clone_sdg_config_for_eval",
    summary=(
        "Build the eval-set SDG config for a trained model by mirroring its"
        " training SDG recipe verbatim (teacher model, prompts, SFT format);"
        " only num_records changes. For a purely synthetic task, re-running"
        " generates fresh held-out inputs. For a document-grounded task it"
        " re-seeds from the SAME source documents, so the eval is freshly"
        " regenerated but NOT a disjoint held-out split — the platform's"
        " input-overlap check runs at validate time either way. Pass the"
        " returned `config` straight to validate_sdg_job. Use this instead"
        " of hand-assembling an eval SDG."
    ),
)
async def clone_sdg_config_for_eval(
    request: CloneSdgForEvalRequest,
    db: asyncpg.Connection = Depends(_get_db),
) -> ClonedSdgConfig:
    """Deterministically mirror a model's training SDG recipe for its eval set.

    Replaces the error-prone manual clone: resolving the training job's parent
    SDG and copying its teacher/prompts/format by hand is exactly where a weak
    driver drifts (wrong teacher, reworded prompt) or loops (polling get_job to
    'verify' overlap). This returns the mirrored config in one call."""
    repo = Repository(db)
    training = await repo.get_job(request.training_job_id)
    if not training:
        raise HTTPException(
            status_code=404,
            detail=f"training job {request.training_job_id[:8]} not found",
        )
    train_sdg_id = str(training.get("parent_job_id") or "")
    if not train_sdg_id:
        raise HTTPException(
            status_code=422,
            detail=(
                "training job has no parent SDG job, so its teacher/prompt/format"
                " cannot be cloned automatically — build the eval SDG from the"
                " training config manually"
            ),
        )
    sdg = await repo.get_job(train_sdg_id)
    if not sdg:
        raise HTTPException(
            status_code=404,
            detail=f"parent SDG job {train_sdg_id[:8]} not found",
        )
    # Mirror the WHOLE generation recipe; only the sample count changes. The shared
    # mirror.clone carries every field except SDG's non-recipe ones (sample count +
    # lineage + mode) so recipe-defining fields carry over automatically — an
    # allowlist silently dropped constraints/tool_configs, giving the eval different
    # generation behavior despite the "only num_records changes" promise.
    cloned = mirror.clone(
        sdg.get("config"),
        mirror.SDG_MIRROR,
        overrides={"num_records": request.num_records},
    )
    if not cloned.get("columns"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"parent SDG job {train_sdg_id[:8]} has no columns to mirror —"
                " its config is empty or in an unexpected shape"
            ),
        )
    # Document-grounded recipes re-seed from the SAME source chunks as training
    # (document_ids is mirrored), so re-running regenerates fresh text but is NOT a
    # disjoint held-out split. Purely synthetic recipes do get fresh held-out draws.
    # Either way the input-overlap check still runs at validate time.
    held_out_note = (
        " It re-seeds from the SAME source documents as training, so the eval is"
        " freshly regenerated but NOT a disjoint held-out split; the platform"
        " flags input overlap when you validate the eval job."
        if cloned.get("document_ids")
        else " Fresh generation yields fresh held-out inputs; the platform still"
        " flags any input overlap when you validate the eval job."
    )
    return ClonedSdgConfig(
        config=cloned,
        train_sdg_job_id=train_sdg_id,
        note=(
            "Mirrors the training SDG recipe (teacher, prompts, SFT format)"
            " verbatim; only num_records changed. Pass `config` to"
            " validate_sdg_job." + held_out_note
        ),
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

    repo = Repository(db)
    # Resolve the run the job will ACTUALLY train on: an explicit data_run_id
    # (e.g. a split complement) takes precedence over the parent SDG's full
    # output, matching _resolve_training_data_run and the overlap check. Resolving
    # parent-first here showed a split-trained job the full SDG size on the card.
    data_run_id = str(config.get("data_run_id") or "")
    data_run = data_run_id or await _resolve_data_run(repo, parent_job_id, "")
    record_count = await _dataset_record_count(data_run)

    return ValidatedJobConfig(
        job_type=JobType.training,
        config=config,
        parent_job_id=parent_job_id,
        data_record_count=record_count,
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
    errors.extend(await _apply_classification_eval_mode(config, db))
    errors.extend(_validate_eval_rubric_judge(config))
    if config.get("eval_mode") != "classification" and not [
        c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")
    ]:
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
    warnings.extend(await _training_sdg_mirror_warning(config, parent_job_id, db))
    warnings.extend(await _eval_overlap_warning(config, parent_job_id, db))

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
        cfg = _coerce_config(job.get("config"))
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


_SdgSignature = tuple[
    tuple[str, ...], tuple[str, ...], str, tuple[str, ...], tuple[str, ...]
]


def _sdg_signature(cfg: Any) -> _SdgSignature | None:
    """What defines an SDG pipeline's task AND its input distribution.

    Teacher models + system prompts alone are not enough: `clone_sdg_config_for_eval`
    treats `topic`, `document_ids`, and the sampler columns as recipe-defining too,
    so two configs with the same teacher+assessor prompt but a different document set
    or topic generate a DIFFERENT task. Folding those in stops a different-input eval
    set from being flagged `recipe_match` / passing the mirror check. Returns sorted
    tuples so ordering doesn't matter; `None` when unparseable or carrying no signal."""
    cfg = _coerce_config(cfg)
    models = tuple(
        sorted(
            str(m["model"])
            for m in (cfg.get("model_configs") or [])
            if isinstance(m, dict) and m.get("model")
        )
    )
    prompts = tuple(
        sorted(
            str(c["system_prompt"])
            for c in (cfg.get("columns") or [])
            if isinstance(c, dict) and c.get("system_prompt")
        )
    )
    topic = str(cfg.get("topic") or "")
    document_ids = tuple(sorted(str(d) for d in (cfg.get("document_ids") or []) if d))
    # Column identity (name/type), so a different sampler set is a different task
    # even when the assessor system prompt is unchanged.
    columns = tuple(
        sorted(
            str(c.get("name") or c.get("column_type") or "")
            for c in (cfg.get("columns") or [])
            if isinstance(c, dict) and (c.get("name") or c.get("column_type"))
        )
    )
    if not models and not prompts and not topic and not document_ids and not columns:
        return None
    return models, prompts, topic, document_ids, columns


async def _training_sdg_mirror_warning(
    config: dict[str, Any], parent_job_id: str, db: asyncpg.Connection
) -> list[str]:
    """Warn when an eval set for a trained model was NOT generated by the same
    SDG pipeline that produced the model's training data.

    A tuned model can only be scored fairly on the task it was trained for, so
    the eval-data SDG must reuse the training-data SDG's teacher model and
    assessor system prompt (regenerating only fresh inputs for held-out
    isolation). We compare the two SDG configs' (teacher models, system prompts)
    signatures and warn — do not block — on divergence.

    Best-effort: silent when the lineage can't be resolved from job rows — the
    eval data is an uploaded dataset / split rather than a chained SDG job
    (no `parent_job_id`), or the training data did not chain from an SDG job
    (trained from a split/upload, or the SDG ran in an earlier, absent job).
    Those gaps are why this warns rather than blocks."""
    training_job_id = str(config.get("training_job_id") or "")
    if not training_job_id or not parent_job_id:
        return []  # not an eval-for-trained-model chained from an SDG job

    repo = Repository(db)
    training = await repo.get_job(training_job_id)
    if not training:
        return []
    train_sdg_id = str(training.get("parent_job_id") or "")
    if not train_sdg_id:
        return []  # training data not chained from an SDG job -> nothing to mirror

    train_sdg = await repo.get_job(train_sdg_id)
    eval_sdg = await repo.get_job(str(parent_job_id))
    if not train_sdg or not eval_sdg:
        return []

    train_sig = _sdg_signature(train_sdg.get("config"))
    eval_sig = _sdg_signature(eval_sdg.get("config"))
    if not train_sig or not eval_sig or train_sig == eval_sig:
        return []

    # Explain every signature component that diverged — the signature folds in
    # topic/document_ids/columns too, so building diffs for only teacher+prompt
    # left a topic/doc/column-only change showing an empty "differs in: )".
    diffs: list[str] = []
    if train_sig[0] != eval_sig[0]:
        diffs.append(
            f"teacher model (training SDG: {', '.join(train_sig[0]) or 'none'};"
            f" eval SDG: {', '.join(eval_sig[0]) or 'none'})"
        )
    if train_sig[1] != eval_sig[1]:
        diffs.append("assessor/system prompt")
    if train_sig[2] != eval_sig[2]:
        diffs.append(
            f"topic (training SDG: {train_sig[2] or 'none'};"
            f" eval SDG: {eval_sig[2] or 'none'})"
        )
    if train_sig[3] != eval_sig[3]:
        diffs.append("source documents")
    if train_sig[4] != eval_sig[4]:
        diffs.append("columns / sampler set")
    return [
        "this eval set for a trained model was NOT generated by the same SDG"
        f" pipeline as the model's training data (differs in: {'; '.join(diffs)})."
        " A tuned model should be scored on the task it was trained for — reuse"
        f" the training SDG config (job {train_sdg_id}) verbatim: same teacher"
        " model and assessor system prompt, regenerating only fresh inputs."
    ]


def _record_input_signature(rec: dict[str, Any]) -> str:
    """A stable hash of a record's INPUT content, for overlap detection.

    SDG datasets are the SFT `messages` shape, so the input the model saw is the
    user turn(s); two records with the same user content are the same example
    regardless of the generated answer. Falls back to the whole record when there
    is no `messages` column, so it still works on other dataset shapes."""
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        user = "\n".join(
            str(m.get("content", ""))
            for m in msgs
            if isinstance(m, dict) and m.get("role") == "user"
        ).strip()
        if user:
            return hashlib.sha256(user.encode("utf-8")).hexdigest()
    blob = json.dumps(rec, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


async def _input_signatures(run_id: str) -> set[str]:
    """Input signatures of every record in an MLflow dataset run."""
    from amortized.api.datasets import (
        _find_dataset_artifacts,
        _mlflow_client,
        _parse_records,
    )

    mlflow = _mlflow_client()
    paths = await _find_dataset_artifacts(mlflow, run_id)
    sigs: set[str] = set()
    for path in paths:
        for rec in _parse_records(path, await mlflow.get_artifact(run_id, path)):
            sigs.add(_record_input_signature(rec))
    return sigs


async def _dataset_record_count(run_id: str) -> int | None:
    """Number of records in an MLflow dataset run, or None if it can't be loaded.

    Best-effort: a transient MLflow error must never block an otherwise-valid
    confirmation card, so the count is simply omitted on failure."""
    if not run_id:
        return None
    from amortized.api.datasets import (
        _find_dataset_artifacts,
        _mlflow_client,
        _parse_records,
    )

    try:
        mlflow = _mlflow_client()
        paths = await _find_dataset_artifacts(mlflow, run_id)
        total = 0
        for path in paths:
            total += sum(
                1 for _ in _parse_records(path, await mlflow.get_artifact(run_id, path))
            )
        return total if paths else None
    except Exception:
        logger.warning("record count: failed to load dataset %s", run_id[:8], exc_info=True)
        return None


async def _resolve_data_run(
    repo: Repository, parent_job_id: str, data_run_id: str
) -> str:
    """MLflow run holding a dataset, from either a parent job or a direct run id."""
    if parent_job_id:
        job = await repo.get_job(parent_job_id)
        if job:
            run = str(job.get("mlflow_run_id") or "")
            if run:
                return run
            # Parent row exists but its run id isn't populated — fall through to the
            # explicit data_run_id rather than returning "" (which silently disabled
            # the downstream leakage/overlap check).
    return data_run_id


async def _resolve_training_data_run(repo: Repository, training_job_id: str) -> str:
    """MLflow run holding the data a training job actually trained on.

    Prefer the exact `data_run_id` the job used (e.g. a split complement), else
    the parent SDG job's MLflow run."""
    training = await repo.get_job(training_job_id)
    if not training:
        return ""
    cfg = training.get("config") or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except ValueError:
            cfg = {}
    data_run_id = str((cfg or {}).get("data_run_id") or "")
    if data_run_id:
        return data_run_id
    return await _resolve_data_run(repo, str(training.get("parent_job_id") or ""), "")


async def _eval_overlap_warning(
    config: dict[str, Any], parent_job_id: str, db: asyncpg.Connection
) -> list[str]:
    """Warn when an eval set for a trained model reuses the model's training inputs.

    Resolves the eval dataset and the model's training dataset to their MLflow
    runs, loads both, and intersects per-record input signatures. Any overlap
    means the model would be scored on inputs it already trained on (leakage).
    Warn — do not block.

    Best-effort: silent when either dataset can't be resolved or loaded (the same
    lineage gaps as the mirror check), so a transient MLflow error never blocks
    an otherwise-valid eval."""
    training_job_id = str(config.get("training_job_id") or "")
    if not training_job_id:
        return []  # not an eval for a trained model

    repo = Repository(db)
    eval_run = await _resolve_data_run(
        repo, parent_job_id, str(config.get("eval_data_run_id") or "")
    )
    train_run = await _resolve_training_data_run(repo, training_job_id)
    if not eval_run or not train_run:
        return []

    if eval_run == train_run:
        return [
            "the eval dataset is the model's training dataset — the model would be"
            " scored entirely on data it trained on. Use a held-out eval set."
        ]

    try:
        eval_sigs = await _input_signatures(eval_run)
        train_sigs = await _input_signatures(train_run)
    except Exception:
        logger.warning("eval overlap check: failed to load datasets", exc_info=True)
        return []
    if not eval_sigs or not train_sigs:
        return []

    overlap = eval_sigs & train_sigs
    if not overlap:
        return []
    return [
        f"{len(overlap)} of {len(eval_sigs)} eval records reuse inputs the model"
        " already saw in training — the eval would be scored partly on its own"
        " training data (leakage). Regenerate the held-out set with fresh inputs"
        " before evaluating."
    ]


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
