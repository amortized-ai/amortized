"""Job admission: the shared path for validating and creating jobs from configs.

Every create route (and retry) goes through ``admit_job``, which checks the LLM endpoints the
job will call before inserting it. ``core.jobs.create_job`` refuses a job type that declares
endpoints unless admission ran, so a future job type gets enforcement just by implementing its
builder's ``model_checks`` hook.

Database connections are scoped to the queries that need them: the model check can take up to
its full time budget, and holding a pooled connection across it would let a handful of slow
validations starve the pool.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException

from amortized.core.jobs import (
    InvalidJobStateError,
    create_job,
    gpu_budget_error,
    inline_credential_error,
)
from amortized.core.model_check_client import check_job_models
from amortized.core.model_endpoints import FAILED, UNVERIFIED, CheckResult
from amortized.db.connection import get_pool
from amortized.db.repository import Repository
from amortized.models import JobType

logger = logging.getLogger("amortized.api.job_admission")

Prechecks = Callable[[Repository], Awaitable[list[str]]]
BeforeInsert = Callable[[Repository], Awaitable[None]]


class ModelCheckFailedError(Exception):
    """A definite model-check failure or config error. Rendered as a 422 with code
    ``model_check_failed`` so clients can tell it apart from other validation errors."""

    def __init__(self, message: str, details: list[dict[str, str]]) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


# What each definite failure means and what to change, in words. The message is relayed by the
# agent and shown on the "Not submitted" card, so it says what's wrong — not the raw HTTP error,
# which reads as a broken endpoint when the usual cause is a wrong model name (a 404 from a
# gateway that serves no such model). The raw detail stays in the per-endpoint `details`.
_FAILURE_HINTS = {
    "not_found": "isn't served there (HTTP 404). Check the model name; for a gateway endpoint,"
    " use the endpoint name",
    "auth_required": "requires an API key (HTTP 401/403). Add the endpoint's key in Settings",
    "auth_failed": "rejected the API key (HTTP 401/403). Check the key in Settings",
    "unreachable": "couldn't be reached. Check the URL",
    "deadline": "didn't answer before the check's deadline. The endpoint may be overloaded or"
    " the URL wrong",
    "bad_request": "rejected a test request. Check the model name and endpoint",
    "server_error": "returned a server error. The endpoint may be down",
    "bad_response": "didn't answer like a chat model. Check that the URL is its chat API",
}


def _failure_line(result: CheckResult) -> str:
    hint = _FAILURE_HINTS.get(result.reason, f"failed a test request ({result.reason})")
    return f"{result.label} model {result.model!r} at {result.base_url} {hint}."


async def check_models_or_raise(job_type: str, config: dict[str, Any]) -> list[CheckResult]:
    """Check the job's LLM endpoints. Raises ModelCheckFailedError on any config error or
    definite failure; returns the remaining (passed / unverified) results."""
    report = await check_job_models(job_type, config)
    failed = [r for r in report.results if r.outcome == FAILED]
    if report.config_errors or failed:
        lines = [e.message for e in report.config_errors] + [_failure_line(r) for r in failed]
        details = [{"field": e.label, "error": e.message} for e in report.config_errors] + [
            {
                "field": r.label,
                "error": f"{r.model} at {r.base_url}: {r.reason}"
                + (f": {r.detail}" if r.detail else ""),
            }
            for r in failed
        ]
        # One problem per line (each is a sentence; the card renders line breaks).
        raise ModelCheckFailedError("\n".join(lines), details)
    for r in report.results:
        if r.outcome == UNVERIFIED:
            logger.warning(
                "%s job: %s model %r at %s not verified (%s)",
                job_type,
                r.label,
                r.model,
                r.base_url,
                r.reason,
            )
    return report.results


async def admit_job(
    job_type: JobType,
    config: dict[str, Any],
    *,
    parent_job_id: str = "",
    user_id: str = "",
    recipe: str = "",
    retry_of: str = "",
    prechecks: Prechecks | None = None,
    before_insert: BeforeInsert | None = None,
) -> tuple[dict[str, Any], list[CheckResult]]:
    """Validate, model-check, and create a job.

    1. ``prechecks`` (DB-backed validation) in a scoped connection; errors → 422. Then the GPU
       budget, on the config as the prechecks normalized it.
    2. The model check, with no connection held.
    3. In a fresh connection: ``prechecks`` again if the check probed anything (upstream state
       may have changed while it ran), ``before_insert`` side effects, then the insert with the
       check report."""
    # A config carrying credentials must never be probed (a user:password@ base_url would
    # otherwise be sent before create_job refuses it), so this comes before anything else.
    credential_error = inline_credential_error(config)
    if credential_error:
        raise HTTPException(status_code=422, detail=[credential_error])

    if prechecks is not None:
        async with get_pool().acquire() as conn:
            errors = await prechecks(Repository(conn))
        if errors:
            raise HTTPException(status_code=422, detail=errors)

    # After the prechecks: they can set fields the GPU estimate reads (an eval of a classifier
    # gets eval_mode / eval_on_gpu stamped). Still before the probe, so a job over budget
    # doesn't wait out a check it can never use.
    budget_error = gpu_budget_error(job_type, config)
    if budget_error:
        raise HTTPException(status_code=422, detail=[budget_error])

    results = await check_models_or_raise(job_type.value, config)

    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        # Nothing probed means no time passed since the prechecks; re-running them would only
        # repeat their work (a training precheck downloads the dataset from MLflow).
        if prechecks is not None and results:
            errors = await prechecks(repo)
            if errors:
                raise HTTPException(status_code=422, detail=errors)
        if before_insert is not None:
            await before_insert(repo)
        try:
            row = await create_job(
                repo,
                job_type=job_type,
                config=config,
                recipe=recipe,
                parent_job_id=parent_job_id,
                user_id=user_id,
                retry_of=retry_of,
                models_checked=True,
                model_check_report=[r.to_dict() for r in results],
            )
        except InvalidJobStateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return row, results
