"""Job management endpoints."""

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from amortized_runtime.db import (
    create_job,
    get_artifact,
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

logger = logging.getLogger("amortized_runtime.api.jobs")

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
            try:
                data = json.loads(line)
                metrics.append(TrainingMetric(**data))
            except (json.JSONDecodeError, Exception):
                continue  # skip malformed or summary lines
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


@router.get("/{job_id}/artifacts/{artifact_id}/preview")
async def preview_artifact(
    job_id: str,
    artifact_id: str,
    lines: int = 5,
    db: aiosqlite.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Preview the contents of an artifact file.

    For text-based files (.jsonl, .json, .csv, .txt, .log, .md), returns
    the first N lines.  For binary files (.safetensors, .bin, .model, .pt),
    returns file size and type information instead.
    """
    lines = max(1, min(lines, 50))

    row = await get_job(db, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    artifact = await get_artifact(db, artifact_id)
    if artifact is None or artifact["job_id"] != job_id:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    file_path = Path(artifact["path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk")

    binary_exts = {".safetensors", ".bin", ".model", ".pt", ".gguf"}
    if file_path.suffix.lower() in binary_exts:
        return {
            "type": "binary",
            "format": file_path.suffix.lstrip("."),
            "size": file_path.stat().st_size,
            "filename": file_path.name,
        }

    # Text-based file — read first N lines
    preview_lines: list[str] = []
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    break
                preview_lines.append(line.rstrip("\n"))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read artifact: {exc}") from exc

    return {
        "type": "text",
        "format": file_path.suffix.lstrip("."),
        "filename": file_path.name,
        "lines": preview_lines,
        "total_size": file_path.stat().st_size,
    }


@router.get("/{job_id}/artifacts/{artifact_id}/download")
async def download_artifact(
    job_id: str,
    artifact_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> FileResponse:
    """Download a specific artifact file."""
    row = await get_job(db, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    artifact = await get_artifact(db, artifact_id)
    if artifact is None or artifact["job_id"] != job_id:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    file_path = Path(artifact["path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream",
    )


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

    # Kill the subprocess if the job is running and has a PID
    pid = row.get("pid")
    if current_status == JobStatus.running.value and pid is not None:
        from amortized_runtime.worker import kill_job_process

        await kill_job_process(pid)
        logger.info("Killed process %d for job %s", pid, job_id)

    updated = await update_job_status(
        db,
        job_id,
        status=JobStatus.cancelled,
        updated_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
    )
    logger.info("Cancelled job %s", job_id)
    return Job(**updated)  # type: ignore[arg-type]
