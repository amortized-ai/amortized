"""Top-level artifact CRUD endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from amortized.core.artifacts import (
    delete_artifact as core_delete_artifact,
)
from amortized.core.artifacts import (
    get_artifact as core_get_artifact,
)
from amortized.core.artifacts import (
    list_all_artifacts as core_list_all_artifacts,
)
from amortized.core.artifacts import (
    register_artifact as core_register_artifact,
)
from amortized.core.storage import LocalStorage, get_storage
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import Artifact, ArtifactRequest, UploadUrlRequest, UploadUrlResponse

logger = logging.getLogger("amortized.api.artifacts")

router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])


@router.post("", status_code=201, response_model=Artifact)
async def create_artifact(
    req: ArtifactRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Artifact:
    repo = Repository(db)
    row = await core_register_artifact(
        repo,
        name=req.name,
        artifact_type=req.artifact_type,
        location=req.location,
        metadata=req.metadata,
        producer_job=req.producer_job,
    )
    return Artifact(**row)


@router.get("", response_model=list[Artifact])
async def list_artifacts(
    type: str | None = None,
    producer_job: str | None = None,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Artifact]:
    repo = Repository(db)
    rows = await core_list_all_artifacts(repo, artifact_type=type, producer_job=producer_job)
    return [Artifact(**r) for r in rows]


@router.get("/{artifact_id}", response_model=Artifact)
async def get_artifact(
    artifact_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Artifact:
    repo = Repository(db)
    row = await core_get_artifact(repo, artifact_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
    return Artifact(**row)


@router.delete("/{artifact_id}", status_code=204)
async def delete_artifact(
    artifact_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> None:
    repo = Repository(db)
    existing = await core_get_artifact(repo, artifact_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
    await core_delete_artifact(repo, artifact_id)


@router.post("/upload-url", response_model=UploadUrlResponse)
async def get_upload_url(req: UploadUrlRequest) -> UploadUrlResponse:
    storage = get_storage()
    if isinstance(storage, LocalStorage):
        raise HTTPException(
            status_code=400,
            detail="Pre-signed uploads not available with local storage backend",
        )
    try:
        result = storage.generate_upload_url(req.name, req.content_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return UploadUrlResponse(**result)


@router.get("/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Any:
    repo = Repository(db)
    artifact = await core_get_artifact(repo, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    location = artifact.get("location") or artifact.get("path", "")

    if location.startswith(("s3://", "gs://")):
        storage = get_storage()
        if not isinstance(storage, LocalStorage):
            after_scheme = location.split("//", 1)[1]
            key = after_scheme.split("/", 1)[1] if "/" in after_scheme else location
            try:
                url = storage.generate_download_url(key)
                return JSONResponse({"location": url})
            except Exception:
                pass
        return JSONResponse({"location": location})

    if location.startswith(("http://", "https://")):
        return JSONResponse({"location": location})

    file_path = Path(location)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream",
    )
