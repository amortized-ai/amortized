"""GPU allocation endpoint."""

from __future__ import annotations

import logging
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from amortized.db import get_db
from amortized.db.repository import Repository

logger = logging.getLogger("amortized.api.gpu")

router = APIRouter(prefix="/api/v1/gpu", tags=["gpu"])


class GpuJobAllocation(BaseModel):
    job_id: str
    job_name: str
    status: str
    gpus_requested: int
    gpu_type: str | None = None
    memory_requested_gib: float | None = None


class GpuAllocationResponse(BaseModel):
    available: bool
    reason: str | None = Field(None, description="Set when available=False")
    total_gpus: int = 0
    allocated_gpus: int = 0
    total_memory_requested_gib: float = 0.0
    gpu_devices: list[str] = Field(default_factory=list)
    jobs: list[GpuJobAllocation] = Field(default_factory=list)


def _detect_gpu_info() -> dict[str, Any]:
    """Detect GPU availability on the server."""
    import shutil

    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "count": torch.cuda.device_count(),
                "devices": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ],
            }
    except ImportError:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    return {
        "available": nvidia_smi is not None,
        "count": 0,
        "devices": [],
    }


async def _get_running_gpu_jobs(repo: Repository) -> list[dict[str, Any]]:
    """Get running jobs that have GPU allocations from their config."""
    active_statuses = ("running", "provisioning", "queued")
    jobs: list[dict[str, Any]] = []
    for status_val in active_statuses:
        from amortized.models import JobStatus

        status_jobs = await repo.list_jobs(status=JobStatus(status_val))
        jobs.extend(status_jobs)
    return jobs


def _extract_gpu_info_from_job(job: dict[str, Any]) -> tuple[int, str | None]:
    """Extract GPU count and type from job config."""
    config = job.get("config", {})
    if not isinstance(config, dict):
        return 0, None

    compute = config.get("compute", {})
    if isinstance(compute, dict):
        gpus = compute.get("gpus", 0)
        gpu_type = compute.get("gpu_type")
        if isinstance(gpus, int) and gpus > 0:
            return gpus, gpu_type

    gpus = config.get("gpus", 0)
    gpu_type = config.get("gpu_type")
    if isinstance(gpus, int) and gpus > 0:
        return gpus, gpu_type

    return 0, None


@router.get(
    "/allocation",
    response_model=GpuAllocationResponse,
    operation_id="get_gpu_allocation",
    summary="Get current GPU allocation across active training jobs.",
)
async def get_gpu_allocation(
    conn: asyncpg.Connection = Depends(get_db),
) -> GpuAllocationResponse:
    gpu_info = _detect_gpu_info()

    if not gpu_info["available"]:
        return GpuAllocationResponse(
            available=False,
            reason="No GPU detected on this server",
        )

    total_gpus: int = gpu_info.get("count", 0) or 0
    gpu_devices: list[str] = gpu_info.get("devices", [])

    repo = Repository(conn)
    try:
        active_jobs = await _get_running_gpu_jobs(repo)
    except Exception:
        logger.exception("Failed to query active jobs")
        return GpuAllocationResponse(
            available=False,
            reason="Cannot reach compute cluster",
        )

    gpu_jobs: list[GpuJobAllocation] = []
    allocated_gpus = 0
    total_memory = 0.0

    for job in active_jobs:
        gpus_requested, gpu_type = _extract_gpu_info_from_job(job)
        if gpus_requested <= 0:
            continue

        job_name = job.get("config", {}).get("topic", "") or job.get("id", "")[:12]
        gpu_jobs.append(
            GpuJobAllocation(
                job_id=job["id"],
                job_name=job_name,
                status=job["status"],
                gpus_requested=gpus_requested,
                gpu_type=gpu_type,
            )
        )
        allocated_gpus += gpus_requested

    gpu_jobs.sort(key=lambda j: j.gpus_requested, reverse=True)

    return GpuAllocationResponse(
        available=True,
        total_gpus=total_gpus,
        allocated_gpus=allocated_gpus,
        total_memory_requested_gib=total_memory,
        gpu_devices=gpu_devices,
        jobs=gpu_jobs,
    )
