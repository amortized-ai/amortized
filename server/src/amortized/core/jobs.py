"""Job lifecycle domain logic — zero HTTP imports."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from amortized.core.events import emit_event
from amortized.core.job_types import UnknownJobTypeError, validate_config, validate_semantic
from amortized.models import JobStatus, JobType

if TYPE_CHECKING:
    from amortized.db.repository import Repository

logger = logging.getLogger("amortized.core.jobs")

_VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.validating: {JobStatus.queued, JobStatus.failed},
    JobStatus.queued: {JobStatus.provisioning, JobStatus.failed, JobStatus.cancelled},
    JobStatus.provisioning: {JobStatus.running, JobStatus.failed, JobStatus.cancelled},
    JobStatus.running: {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled},
}


async def transition_job(
    repo: Repository,
    job_id: str,
    new_status: JobStatus,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate and perform a job state transition."""
    job = await repo.get_job(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    current = JobStatus(job["status"])
    allowed = _VALID_TRANSITIONS.get(current, set())
    if new_status not in allowed:
        raise InvalidJobStateError(
            f"Cannot transition from {current.value} to {new_status.value}"
        )
    now = datetime.now(UTC).isoformat()
    updated = await repo.update_job_status(job_id, status=new_status, updated_at=now, **kwargs)
    await emit_event(repo, job_id, "state_change", {"status": new_status.value})
    assert updated is not None
    return updated


async def create_job(
    repo: Repository,
    *,
    job_type: JobType,
    config: dict[str, Any],
    output_dir: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    row = await repo.create_job(
        job_id=job_id,
        job_type=job_type,
        config=config,
        created_at=now,
        output_dir=output_dir,
        metadata=metadata,
    )

    try:
        schema_errors = validate_config(job_type.value, config)
    except UnknownJobTypeError:
        schema_errors = []
    if schema_errors:
        await transition_job(repo, job_id, JobStatus.failed, error=str(schema_errors))
        raise InvalidJobStateError(f"Schema validation failed: {schema_errors}")

    semantic_errors = await validate_semantic(job_type.value, config)
    if semantic_errors:
        await transition_job(repo, job_id, JobStatus.failed, error=str(semantic_errors))
        raise InvalidJobStateError(f"Pre-flight validation failed: {semantic_errors}")

    await transition_job(repo, job_id, JobStatus.queued)
    logger.info("Created %s job %s", job_type.value, job_id)
    return await repo.get_job(job_id) or row


async def validate_job(
    *,
    job_type: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate a job config without creating it. Returns dry-run result."""
    try:
        schema_errors = validate_config(job_type, config)
    except UnknownJobTypeError as exc:
        raise exc
    semantic_errors = await validate_semantic(job_type, config)
    all_errors = schema_errors + semantic_errors
    return {
        "valid": not all_errors,
        "errors": all_errors,
    }


async def get_job(repo: Repository, job_id: str) -> dict[str, Any] | None:
    return await repo.get_job(job_id)


async def list_jobs(
    repo: Repository,
    *,
    status: JobStatus | None = None,
    job_type: JobType | None = None,
) -> list[dict[str, Any]]:
    return await repo.list_jobs(status=status, job_type=job_type)


async def cancel_job(repo: Repository, job_id: str) -> dict[str, Any]:
    row = await repo.get_job(job_id)
    if row is None:
        raise JobNotFoundError(job_id)

    current_status = row["status"]
    if current_status in (JobStatus.succeeded.value, JobStatus.failed.value):
        raise InvalidJobStateError(
            f"Cannot cancel job with status '{current_status}'"
        )
    if current_status == JobStatus.cancelled.value:
        return row

    now = datetime.now(UTC).isoformat()

    if current_status == JobStatus.running.value:
        from amortized.worker import cancel_job_via_backend, kill_job_process

        handle_json = row.get("backend_handle")
        cancelled = await cancel_job_via_backend(job_id, handle_json)
        if not cancelled:
            pid = row.get("pid")
            if pid is not None:
                await kill_job_process(pid)
                logger.info("Killed process %d for job %s", pid, job_id)

    updated = await repo.update_job_status(
        job_id,
        status=JobStatus.cancelled,
        updated_at=now,
        completed_at=now,
    )
    await emit_event(repo, job_id, "state_change", {"status": JobStatus.cancelled.value})
    logger.info("Cancelled job %s", job_id)
    assert updated is not None, f"Job {job_id} vanished during cancel"
    return updated


class JobNotFoundError(Exception):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job {job_id} not found")


class InvalidJobStateError(Exception):
    pass
