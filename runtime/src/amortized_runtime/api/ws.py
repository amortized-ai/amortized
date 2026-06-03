"""WebSocket endpoint for real-time job metrics streaming."""

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from amortized_runtime.db import get_job
from amortized_runtime.models import JobStatus, JobType

logger = logging.getLogger("amortized_runtime.api.ws")

router = APIRouter(prefix="/api/v1/jobs", tags=["websocket"])


async def _get_db_connection() -> aiosqlite.Connection:
    """Get a database connection for WebSocket handlers."""
    import aiosqlite

    import amortized_runtime.config as config_mod

    config_mod.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(config_mod.settings.db_path))
    db.row_factory = aiosqlite.Row
    return db


async def _stream_training_metrics(
    websocket: WebSocket, output_dir: str, job_id: str
) -> None:
    """Tail training_metrics.jsonl and push new lines to the client."""
    metrics_path = Path(output_dir) / "training_metrics.jsonl"
    lines_sent = 0

    while True:
        # Check job status
        db = await _get_db_connection()
        try:
            job = await get_job(db, job_id)
        finally:
            await db.close()

        if job is None:
            await websocket.send_json({"type": "error", "data": {"message": "Job not found"}})
            return

        status = job["status"]

        # Read new metrics lines
        if metrics_path.exists():
            lines = metrics_path.read_text().strip().splitlines()
            new_lines = lines[lines_sent:]
            for line in new_lines:
                if line.strip():
                    try:
                        data = json.loads(line)
                        await websocket.send_json({"type": "metric", "data": data})
                    except json.JSONDecodeError:
                        continue
            lines_sent = len(lines)

        # Send status update
        await websocket.send_json({"type": "status", "data": {"status": status}})

        # Stop if job is done
        if status in (
            JobStatus.completed.value,
            JobStatus.failed.value,
            JobStatus.cancelled.value,
        ):
            return

        await asyncio.sleep(1.0)


async def _stream_sdg_progress(
    websocket: WebSocket, output_dir: str, job_id: str
) -> None:
    """Poll SDG checkpoint progress and push updates to the client."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    metadata_path = checkpoint_dir / "flow_metadata.json"

    while True:
        # Check job status
        db = await _get_db_connection()
        try:
            job = await get_job(db, job_id)
        finally:
            await db.close()

        if job is None:
            await websocket.send_json({"type": "error", "data": {"message": "Job not found"}})
            return

        status = job["status"]

        # Read progress from metadata
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                await websocket.send_json({"type": "metric", "data": metadata})
            except (json.JSONDecodeError, OSError):
                pass

        # Send status update
        await websocket.send_json({"type": "status", "data": {"status": status}})

        # Stop if job is done
        if status in (
            JobStatus.completed.value,
            JobStatus.failed.value,
            JobStatus.cancelled.value,
        ):
            return

        await asyncio.sleep(1.0)


@router.websocket("/{job_id}/stream")
async def stream_job_metrics(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for streaming job metrics in real-time."""
    await websocket.accept()
    logger.info("WebSocket connected for job %s", job_id)

    try:
        # Look up the job
        db = await _get_db_connection()
        try:
            job = await get_job(db, job_id)
        finally:
            await db.close()

        if job is None:
            await websocket.send_json({"type": "error", "data": {"message": "Job not found"}})
            await websocket.close()
            return

        output_dir = job.get("output_dir")
        if not output_dir:
            await websocket.send_json(
                {"type": "error", "data": {"message": "No output directory for job"}}
            )
            await websocket.close()
            return

        # Stream based on job type
        if job["type"] == JobType.training.value:
            await _stream_training_metrics(websocket, output_dir, job_id)
        else:
            await _stream_sdg_progress(websocket, output_dir, job_id)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)
    except Exception:
        logger.exception("WebSocket error for job %s", job_id)
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
