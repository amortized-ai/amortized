"""Event streaming and listing endpoints — HTTP layer over core.events."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

import amortized.config as config_mod
from amortized.core.events import list_events as core_list_events
from amortized.core.jobs import get_job as core_get_job
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository

logger = logging.getLogger("amortized.api.events")

router = APIRouter(prefix="/api/v1/jobs", tags=["events"])


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

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        cursor = since
        db_path = str(config_mod.settings.db_path)
        while True:
            if await request.is_disconnected():
                break

            db_inner = await aiosqlite.connect(db_path)
            db_inner.row_factory = aiosqlite.Row
            try:
                inner_repo = Repository(db_inner)
                events = await core_list_events(
                    inner_repo, job_id, since=cursor, types=type_list,
                )
            finally:
                await db_inner.close()

            for event in events:
                cursor = event["timestamp"]
                yield {
                    "event": event["type"],
                    "data": json.dumps(event),
                }

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())
