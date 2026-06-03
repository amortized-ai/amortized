"""Domain event model — zero HTTP imports."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from amortized_runtime.db.repository import Repository


@dataclass
class Event:
    job_id: str
    type: str
    data: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


async def emit_event(
    repo: Repository,
    job_id: str,
    event_type: str,
    data: dict[str, Any] | None = None,
) -> Event:
    event = Event(job_id=job_id, type=event_type, data=data)
    await repo.create_event(
        event_id=event.id,
        job_id=event.job_id,
        event_type=event.type,
        timestamp=event.timestamp,
        data=event.data,
    )
    return event


async def list_events(
    repo: Repository,
    job_id: str,
    *,
    since: str | None = None,
    types: list[str] | None = None,
) -> list[dict[str, Any]]:
    return await repo.list_events(job_id, since=since, types=types)
