"""Job lifecycle domain logic — zero HTTP imports."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from amortized.backends import BackendHandle
from amortized.core.compute import get_backend
from amortized.models import JobStatus, JobType

if TYPE_CHECKING:
    from amortized.db.repository import Repository

logger = logging.getLogger("amortized.core.jobs")


async def create_job(
    repo: Repository,
    *,
    job_type: JobType,
    config: dict[str, Any],
    recipe: str = "",
    parent_job_id: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    if not parent_job_id:
        parent_job_id = config.get("parent_job_id", "")

    job_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    stored_config = dict(config)
    if "parent_job_id" in stored_config:
        stored_config.pop("parent_job_id")

    row = await repo.create_job(
        job_id=job_id,
        job_type=job_type,
        config=stored_config,
        created_at=now,
        recipe=recipe,
        parent_job_id=parent_job_id,
        user_id=user_id,
    )

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
    if current_status in (JobStatus.succeeded.value, JobStatus.failed.value):
        raise InvalidJobStateError(f"Cannot cancel job with status '{current_status}'")
    if current_status == JobStatus.cancelled.value:
        return row

    now = datetime.now(UTC).isoformat()

    if current_status == JobStatus.running.value:
        handle_json = row.get("backend_handle")
        cancelled = await cancel_job_via_backend(job_id, handle_json)
        if not cancelled:
            logger.warning("Could not cancel job %s via backend", job_id)

    updated = await repo.update_job(
        job_id,
        status=JobStatus.cancelled.value,
        completed_at=now,
    )
    logger.info("Cancelled job %s", job_id)
    assert updated is not None
    return updated


def deserialize_handle(raw: str | None) -> BackendHandle | None:
    if not raw:
        return None
    d = json.loads(raw)
    raw_secrets = d.get("secret_names")
    secret_names = [tuple(s) for s in raw_secrets] if raw_secrets else None
    return BackendHandle(
        backend_name=d["backend_name"],
        job_id=d["job_id"],
        remote_pid=d.get("remote_pid"),
        remote_dir=d.get("remote_dir"),
        container_id=d.get("container_id"),
        scheduler_id=d.get("scheduler_id"),
        secret_names=secret_names,
    )


async def cancel_job_via_backend(job_id: str, handle_json: str | None) -> bool:
    handle = deserialize_handle(handle_json)
    if handle is None:
        return False
    try:
        backend = get_backend(handle.backend_name)
        await backend.cancel(handle)
        return True
    except (KeyError, OSError):
        return False


class JobNotFoundError(Exception):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job {job_id} not found")


class InvalidJobStateError(Exception):
    pass


_TERMINAL_STATUSES = frozenset({
    JobStatus.succeeded.value,
    JobStatus.failed.value,
    JobStatus.cancelled.value,
})


async def delete_job(repo: Repository, job_id: str) -> None:
    row = await repo.get_job(job_id)
    if row is None:
        raise JobNotFoundError(job_id)
    if row["status"] not in _TERMINAL_STATUSES:
        raise InvalidJobStateError(
            f"Cannot delete job in state '{row['status']}' — cancel it first"
        )
    await repo.delete_job(job_id)
    logger.info("Deleted job %s", job_id)
