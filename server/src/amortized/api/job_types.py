"""Job type registry and universal job submission endpoints."""

import logging
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from amortized.core.job_types import (
    UnknownJobTypeError,
    get_schema,
    list_job_types,
    validate_config,
)
from amortized.core.jobs import create_job as core_create_job
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import Job, JobRequest, JobType

logger = logging.getLogger("amortized.api.job_types")

router = APIRouter(tags=["jobs"])


@router.post("/api/v1/jobs", status_code=201, response_model=Job)
async def create_job_universal(
    request: JobRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Job:
    try:
        errors = validate_config(request.type, request.config)
    except UnknownJobTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    job_type = JobType(request.type)

    output_dir = request.config.get("ckpt_output_dir")

    repo = Repository(db)
    row = await core_create_job(
        repo,
        job_type=job_type,
        config=request.config,
        output_dir=output_dir,
        metadata=request.metadata,
    )
    return Job(**row)


job_types_router = APIRouter(prefix="/api/v1/job-types", tags=["job-types"])


@job_types_router.get("")
async def get_job_types() -> list[dict[str, str]]:
    return list_job_types()


@job_types_router.get("/{job_type}/schema")
async def get_job_type_schema(job_type: str) -> dict[str, Any]:
    try:
        return get_schema(job_type)
    except UnknownJobTypeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
