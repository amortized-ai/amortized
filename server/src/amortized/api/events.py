"""Event streaming and listing endpoints — HTTP layer over core.events."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

import amortized.config as config_mod
from amortized.core.events import list_events as core_list_events
from amortized.core.jobs import get_job as core_get_job
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository

logger = logging.getLogger("amortized.api.events")

router = APIRouter(prefix="/api/v1/jobs", tags=["events"])

_SSE_RETRY_MS = 3000
_KEEPALIVE_INTERVAL = 15
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _parse_types(types: str | None) -> list[str] | None:
    if not types:
        return None
    return [t.strip() for t in types.split(",") if t.strip()]


@router.get("/{job_id}/events")
async def get_job_events(
    request: Request,
    job_id: str,
    since: str | None = Query(None, description="ISO timestamp — return events after this time"),
    types: str | None = Query(None, description="Comma-separated event types to filter"),
    stream: bool = Query(False, description="Enable SSE streaming"),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    db: aiosqlite.Connection = Depends(_get_db),
) -> Any:
    repo = Repository(db)

    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    type_list = _parse_types(types)

    if not stream:
        events = await core_list_events(repo, job_id, since=since, types=type_list)
        return events

    cursor = last_event_id or since

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        nonlocal cursor
        db_path = str(config_mod.settings.db_path)

        yield {"event": "retry", "data": "", "retry": _SSE_RETRY_MS}  # type: ignore[dict-item]

        ticks_since_event = 0
        while True:
            if await request.is_disconnected():
                break

            db_inner = await aiosqlite.connect(db_path)
            db_inner.row_factory = aiosqlite.Row
            try:
                inner_repo = Repository(db_inner)
                events = await core_list_events(
                    inner_repo,
                    job_id,
                    since=cursor,
                    types=type_list,
                )
                job = await core_get_job(inner_repo, job_id)
            finally:
                await db_inner.close()

            if events:
                ticks_since_event = 0
                for event in events:
                    cursor = event["timestamp"]
                    yield {
                        "id": event["timestamp"],
                        "event": event["type"],
                        "data": json.dumps(
                            {
                                "job_id": job_id,
                                "timestamp": event["timestamp"],
                                "type": event["type"],
                                "data": event.get("data", {}),
                            }
                        ),
                    }
            else:
                ticks_since_event += 1
                if ticks_since_event * 2 >= _KEEPALIVE_INTERVAL:
                    yield {"comment": "keepalive"}
                    ticks_since_event = 0

            if job and job["status"] in _TERMINAL_STATUSES:
                yield {"event": "done", "data": json.dumps({"status": job["status"]})}
                return

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


@router.get("/{job_id}/logs")
async def stream_job_logs(
    request: Request,
    job_id: str,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    db: aiosqlite.Connection = Depends(_get_db),
) -> EventSourceResponse:
    repo = Repository(db)

    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    cursor = last_event_id

    async def log_generator() -> AsyncGenerator[dict[str, str], None]:
        nonlocal cursor
        db_path = str(config_mod.settings.db_path)

        yield {"event": "retry", "data": "", "retry": _SSE_RETRY_MS}  # type: ignore[dict-item]

        while True:
            if await request.is_disconnected():
                break

            db_inner = await aiosqlite.connect(db_path)
            db_inner.row_factory = aiosqlite.Row
            try:
                inner_repo = Repository(db_inner)
                events = await core_list_events(
                    inner_repo,
                    job_id,
                    since=cursor,
                    types=["log", "progress"],
                )
                job = await core_get_job(inner_repo, job_id)
            finally:
                await db_inner.close()

            for event in events:
                cursor = event["timestamp"]
                message = (event.get("data") or {}).get("message", "")
                yield {
                    "id": event["timestamp"],
                    "event": "log",
                    "data": message,
                }

            if job and job["status"] in _TERMINAL_STATUSES:
                yield {"event": "done", "data": job["status"]}
                return

            if not events:
                yield {"comment": "keepalive"}

            await asyncio.sleep(1)

    return EventSourceResponse(log_generator())
