"""Event ingest endpoint — containers POST events back to the control plane."""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from amortized.core.events import emit_event
from amortized.core.jobs import get_job as core_get_job
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository

logger = logging.getLogger("amortized.api.event_ingest")

router = APIRouter(prefix="/api/v1/events", tags=["events"])


class EventIngestRequest(BaseModel):
    job_id: str = Field(..., description="Job that produced this event")
    type: str = Field(..., description="Event type (e.g. progress, state_change, heartbeat)")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload")


@router.post("/ingest")
async def ingest_event(
    req: EventIngestRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Receive events from container runners via HTTP POST."""
    repo = Repository(db)

    job = await core_get_job(repo, req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    event = await emit_event(repo, req.job_id, req.type, req.data)
    logger.debug("Ingested %s event for job %s", req.type, req.job_id)

    if req.type == "artifact" and isinstance(req.data, dict):
        name = req.data.get("name", "")
        path = req.data.get("path", "")
        if name:
            existing = await repo.list_artifacts(req.job_id)
            already_registered = any(a.get("name") == name for a in existing)
            if not already_registered:
                await repo.create_artifact(
                    artifact_id=str(uuid.uuid4()),
                    job_id=req.job_id,
                    artifact_type="generated_data",
                    path=path,
                    size=0,
                    created_at=datetime.now(UTC).isoformat(),
                    name=name,
                    location=path,
                )

    return {"id": event.id, "job_id": req.job_id, "type": req.type, "timestamp": event.timestamp}
