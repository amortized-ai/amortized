"""Job management endpoints."""

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from amortized_runtime.db import (
    create_job,
    get_job,
    list_artifacts,
    list_jobs,
    update_job_status,
)
from amortized_runtime.db import get_db as _get_db
from amortized_runtime.models import (
    Artifact,
    Job,
    JobStatus,
    JobType,
    SDGJobConfig,
    TrainingJobConfig,
    TrainingMetric,
)

logger = logging.getLogger("amortized_runtime.routers.jobs")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.post("/training", status_code=201, response_model=Job)
async def create_training_job(
    config: TrainingJobConfig,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    """Create a new LoRA SFT training job."""
    job = Job(type=JobType.training, config=config.model_dump(exclude_none=True))
    job.output_dir = config.ckpt_output_dir

    row = await create_job(
        db,
        job_id=job.id,
        job_type=JobType.training,
        config=config.model_dump(exclude_none=True),
        created_at=job.created_at,
        output_dir=job.output_dir,
    )
    logger.info("Created training job %s", job.id)
    return Job(**row)


@router.post("/sdg", status_code=201, response_model=Job)
async def create_sdg_job(
    config: SDGJobConfig,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    """Create a new synthetic data generation job."""
    job = Job(type=JobType.sdg, config=config.model_dump(exclude_none=True))

    row = await create_job(
        db,
        job_id=job.id,
        job_type=JobType.sdg,
        config=config.model_dump(exclude_none=True),
        created_at=job.created_at,
    )
    logger.info("Created SDG job %s", job.id)
    return Job(**row)


@router.get("", response_model=list[Job])
async def get_jobs(
    status: JobStatus | None = None,
    type: JobType | None = None,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Job]:
    """List all jobs with optional status/type filters."""
    rows = await list_jobs(db, status=status, job_type=type)
    return [Job(**row) for row in rows]


@router.get("/{job_id}", response_model=Job)
async def get_job_detail(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    """Get detailed information about a specific job."""
    row = await get_job(db, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return Job(**row)


@router.get("/{job_id}/metrics", response_model=list[TrainingMetric])
async def get_job_metrics(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[TrainingMetric]:
    """Return parsed training metrics for a training job."""
    row = await get_job(db, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if row["type"] != JobType.training.value:
        raise HTTPException(status_code=400, detail="Metrics only available for training jobs")

    output_dir = row.get("output_dir")
    if not output_dir:
        return []

    metrics_path = Path(output_dir) / "training_metrics.jsonl"
    if not metrics_path.exists():
        return []

    metrics: list[TrainingMetric] = []
    for line in metrics_path.read_text().strip().splitlines():
        if line.strip():
            data = json.loads(line)
            metrics.append(TrainingMetric(**data))
    return metrics


@router.get("/{job_id}/artifacts", response_model=list[Artifact])
async def get_job_artifacts(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Artifact]:
    """List output artifacts for a job."""
    row = await get_job(db, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    rows = await list_artifacts(db, job_id)
    return [Artifact(**r) for r in rows]


@router.delete("/{job_id}", response_model=Job)
async def cancel_job(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Any:
    """Cancel a running or pending job."""
    row = await get_job(db, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    current_status = row["status"]
    if current_status in (JobStatus.completed.value, JobStatus.failed.value):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{current_status}'",
        )
    if current_status == JobStatus.cancelled.value:
        return Job(**row)

    from datetime import UTC, datetime

    updated = await update_job_status(
        db,
        job_id,
        status=JobStatus.cancelled,
        updated_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
    )
    logger.info("Cancelled job %s", job_id)
    return Job(**updated)  # type: ignore[arg-type]
