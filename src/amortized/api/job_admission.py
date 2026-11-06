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

from amortized.core.jobs import InvalidJobStateError, create_job
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


def _failure_line(result: CheckResult) -> str:
    detail = f": {result.detail}" if result.detail else ""
    return (
        f"{result.label} model {result.model!r} at {result.base_url!r} did not answer a test"
        f" request ({result.reason}{detail})"
    )


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
        raise ModelCheckFailedError("; ".join(lines), details)
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

    1. ``prechecks`` (DB-backed validation) in a scoped connection; errors → 422.
    2. The model check, with no connection held.
    3. In a fresh connection: ``prechecks`` again (upstream state may have changed during the
       check), ``before_insert`` side effects, then the insert with the check report."""
    if prechecks is not None:
        async with get_pool().acquire() as conn:
            errors = await prechecks(Repository(conn))
        if errors:
            raise HTTPException(status_code=422, detail=errors)

    results = await check_models_or_raise(job_type.value, config)

    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        if prechecks is not None:
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
