"""Top-level artifact CRUD endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from amortized.config import settings as _settings
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
    a = Artifact(**row)
    a.download_url = f"/api/v1/artifacts/{a.id}/download"
    return a


@router.get("", response_model=list[Artifact])
async def list_artifacts(
    type: str | None = None,
    producer_job: str | None = None,
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Artifact]:
    repo = Repository(db)
    rows = await core_list_all_artifacts(repo, artifact_type=type, producer_job=producer_job)
    artifacts = []
    for r in rows:
        a = Artifact(**r)
        a.download_url = f"/api/v1/artifacts/{a.id}/download"
        artifacts.append(a)
    return artifacts


@router.get("/{artifact_id}", response_model=Artifact)
async def get_artifact(
    artifact_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> Artifact:
    repo = Repository(db)
    row = await core_get_artifact(repo, artifact_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
    a = Artifact(**row)
    a.download_url = f"/api/v1/artifacts/{a.id}/download"
    return a


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


@router.post("/upload", status_code=201, response_model=Artifact)
async def upload_artifact(
    file: UploadFile,
    artifact_type: str = Form("dataset"),
    name: str | None = Form(None),
    db: aiosqlite.Connection = Depends(_get_db),
) -> Artifact:
    artifact_name = name or (file.filename or "upload")
    upload_dir = Path(_settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / artifact_name
    content = await file.read()
    dest.write_bytes(content)

    repo = Repository(db)
    row = await core_register_artifact(
        repo,
        name=artifact_name,
        artifact_type=artifact_type,
        location=str(dest.resolve()),
        metadata={
            "original_filename": file.filename or "",
            "size_bytes": len(content),
        },
    )
    a = Artifact(**row)
    a.download_url = f"/api/v1/artifacts/{a.id}/download"
    return a


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
        logger.exception("Failed to generate upload URL")
        raise HTTPException(status_code=500, detail="Failed to generate upload URL") from exc
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
            if key.startswith(_settings.storage_prefix):
                key = key[len(_settings.storage_prefix) :]
            try:
                url = storage.generate_download_url(key)
                return JSONResponse({"location": url})
            except Exception as exc:
                logger.warning("Failed to generate pre-signed download URL for %s: %s", key, exc)
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
