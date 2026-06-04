"""Job management endpoints — thin HTTP wrapper over core domain logic."""

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from amortized.core.artifacts import get_artifact as core_get_artifact
from amortized.core.artifacts import list_artifacts as core_list_artifacts
from amortized.core.jobs import (
    InvalidJobStateError,
    JobNotFoundError,
)
from amortized.core.jobs import (
    cancel_job as core_cancel_job,
)
from amortized.core.jobs import (
    create_job as core_create_job,
)
from amortized.core.jobs import (
    get_job as core_get_job,
)
from amortized.core.jobs import (
    list_jobs as core_list_jobs,
)
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import (
    Artifact,
    Job,
    JobStatus,
    JobType,
    SynthJobConfig,
    TrainingJobConfig,
    TrainingMetric,
)

logger = logging.getLogger("amortized.api.jobs")

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.post("/training", status_code=201, response_model=Job)
async def create_training_job(
    config: TrainingJobConfig,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=JobType.training,
            config=config.model_dump(exclude_none=True),
            output_dir=config.ckpt_output_dir,
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Job(**row)


@router.post("/sdg", status_code=201, response_model=Job)
async def create_sdg_job(
    config: SynthJobConfig,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    repo = Repository(db)
    try:
        row = await core_create_job(
            repo,
            job_type=JobType.sdg,
            config=config.model_dump(exclude_none=True),
        )
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Job(**row)


@router.get("", response_model=list[Job])
async def get_jobs(
    status: JobStatus | None = None,
    type: JobType | None = None,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Job]:
    repo = Repository(db)
    rows = await core_list_jobs(repo, status=status, job_type=type)
    return [Job(**row) for row in rows]


@router.get("/{job_id}", response_model=Job)
async def get_job_detail(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return Job(**row)


@router.get("/{job_id}/metrics", response_model=list[TrainingMetric])
async def get_job_metrics(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[TrainingMetric]:
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
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
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return metrics


@router.get("/{job_id}/artifacts", response_model=list[Artifact])
async def get_job_artifacts(
    job_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Artifact]:
    repo = Repository(db)
    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    rows = await core_list_artifacts(repo, job_id)
    return [Artifact(**r) for r in rows]


@router.get("/{job_id}/artifacts/{artifact_id}/preview")
async def preview_artifact(
    job_id: str,
    artifact_id: str,
    lines: int = 5,
    db: aiosqlite.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Preview the contents of an artifact file."""
    lines = max(1, min(lines, 50))
    repo = Repository(db)

    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    artifact = await core_get_artifact(repo, artifact_id)
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
    repo = Repository(db)

    row = await core_get_job(repo, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    artifact = await core_get_artifact(repo, artifact_id)
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
    repo = Repository(db)
    try:
        row = await core_cancel_job(repo, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from exc
    except InvalidJobStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Job(**row)
