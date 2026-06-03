"""Job lifecycle domain logic — zero HTTP imports."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from amortized_runtime.core.events import emit_event
from amortized_runtime.models import JobStatus, JobType

if TYPE_CHECKING:
    from amortized_runtime.db.repository import Repository

logger = logging.getLogger("amortized_runtime.core.jobs")


async def create_job(
    repo: Repository,
    *,
    job_type: JobType,
    config: dict[str, Any],
    output_dir: str | None = None,
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    row = await repo.create_job(
        job_id=job_id,
        job_type=job_type,
        config=config,
        created_at=now,
        output_dir=output_dir,
    )

    await emit_event(repo, job_id, "state_change", {"status": JobStatus.pending.value})
    logger.info("Created %s job %s", job_type.value, job_id)
    return row


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
    if current_status in (JobStatus.completed.value, JobStatus.failed.value):
        raise InvalidJobStateError(
            f"Cannot cancel job with status '{current_status}'"
        )
    if current_status == JobStatus.cancelled.value:
        return row

    now = datetime.now(UTC).isoformat()

    pid = row.get("pid")
    if current_status == JobStatus.running.value and pid is not None:
        from amortized_runtime.worker import kill_job_process

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
