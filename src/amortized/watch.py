"""In-memory job watch registry and event delivery to the agent server."""

from __future__ import annotations

import logging
from typing import Any

import httpx

import amortized.config as config_mod

logger = logging.getLogger("amortized.watch")

_job_watchers: dict[str, str] = {}

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def register_watch(job_id: str, session_id: str) -> None:
    _job_watchers[job_id] = session_id
    logger.info("Watching job %s for session %s", job_id, session_id)


def unregister_watch(job_id: str) -> None:
    removed = _job_watchers.pop(job_id, None)
    if removed:
        logger.info("Unwatched job %s", job_id)


def get_watcher(job_id: str) -> str | None:
    return _job_watchers.get(job_id)


async def emit_job_event(
    job_id: str, status: str, job: dict[str, Any]
) -> None:
    session_id = _job_watchers.get(job_id)
    if not session_id:
        return

    agent_url = config_mod.settings.agent_server_url
    if not agent_url:
        return

    event = {
        "type": "job_status_change",
        "job_id": job_id,
        "status": status,
        "job_type": job.get("type", ""),
        "error": job.get("error"),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{agent_url}/session/{session_id}/event",
                json=event,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Agent event delivery failed: %s %s", resp.status_code, resp.text
                )
    except Exception:
        logger.warning("Failed to deliver event to agent server", exc_info=True)

    if status in TERMINAL_STATUSES:
        unregister_watch(job_id)
