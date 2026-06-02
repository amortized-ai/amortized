"""FastAPI application entry point."""

import asyncio
import contextlib
import logging
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI

from amortized_runtime.db import init_db
from amortized_runtime.routers import agent, estimate, flows, jobs, ws
from amortized_runtime.worker import cleanup_orphaned_jobs, worker_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("amortized_runtime")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize database and start background worker on startup."""
    await init_db()
    await cleanup_orphaned_jobs()
    logger.info("Amortized runtime started")

    # Start background worker
    worker_task = asyncio.create_task(worker_loop())

    yield

    # Shutdown worker
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task
    logger.info("Amortized runtime shutting down")


app = FastAPI(
    title="Amortized Runtime",
    description="AI model customization runtime API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(jobs.router)
app.include_router(flows.router)
app.include_router(estimate.router)
app.include_router(ws.router)
app.include_router(agent.router)


def _detect_gpu() -> dict[str, object]:
    """Detect GPU availability."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "count": torch.cuda.device_count(),
                "devices": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
            }
    except ImportError:
        pass

    # Fallback: check for nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    return {
        "available": nvidia_smi is not None,
        "count": 0,
        "devices": [],
        "note": "torch not installed; nvidia-smi " + ("found" if nvidia_smi else "not found"),
    }


@app.get("/api/v1/health")
async def health() -> dict[str, object]:
    """Health check endpoint with GPU info."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "gpu": _detect_gpu(),
    }
