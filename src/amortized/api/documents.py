from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from urllib.parse import urlparse

import asyncpg
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from amortized.config import settings
from amortized.core.jobs import create_job
from amortized.core.mlflow_client import MLflowClient
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import (
    ChunkerType,
    ConvertUrlRequest,
    DocumentChunk,
    DocumentChunks,
    DocumentResult,
    DocumentSummary,
    DocumentUploadAccepted,
    JobType,
    OutputFormat,
)

logger = logging.getLogger("amortized.api.documents")

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
_ALLOWED_EXTENSIONS = (".pdf", ".docx", ".pptx", ".html", ".txt", ".md", ".xlsx")

_BLOCKED_HOSTNAMES = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.azure.com",
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
    }
)


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[/\\:\x00]", "_", name)
    if len(name) > 255:
        name = name[:255]
    return name or f"upload-{uuid.uuid4().hex[:8]}"


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http:// and https:// URLs are allowed")
    hostname = parsed.hostname or ""
    if hostname in _BLOCKED_HOSTNAMES:
        raise HTTPException(status_code=400, detail="Access to this host is not allowed")
    if hostname.endswith((".svc.cluster.local", ".local", ".internal")):
        raise HTTPException(status_code=400, detail="Access to internal services is not allowed")
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            raise HTTPException(
                status_code=400, detail="Access to private addresses is not allowed"
            )
    except ValueError:
        pass


def _tracking_uri() -> str:
    if not settings.mlflow_tracking_uri:
        raise HTTPException(status_code=503, detail="MLflow tracking is not configured")
    return settings.mlflow_tracking_uri.rstrip("/")


def _format_timestamp(ts: int | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts / 1000, tz=UTC).isoformat()


@router.post(
    "/convert",
    response_model=DocumentUploadAccepted,
    status_code=202,
    operation_id="convert_document",
    summary="Upload a document and create an async processing job.",
)
async def convert_document(
    file: UploadFile,
    chunker_type: ChunkerType = ChunkerType.sentence,
    chunk_size: int = Query(2048, ge=64, le=8192),
    chunk_overlap: int = Query(200, ge=0),
    db: asyncpg.Connection = Depends(_get_db),
) -> DocumentUploadAccepted:
    filename = _sanitize_filename(file.filename or f"upload-{uuid.uuid4().hex[:8]}")
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes, max {_MAX_UPLOAD_BYTES})",
        )

    config_dict = {
        "filename": filename,
        "chunker_type": chunker_type.value,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    client = MLflowClient(_tracking_uri())
    experiment_id = await client.ensure_experiment("amortized/uploads")
    run_id = await client.create_run(experiment_id, name=filename, tags={"job_type": "document"})
    await client.upload_artifact(run_id, f"source/{filename}", file_bytes)

    config_dict["mlflow_upload_run_id"] = run_id
    config_dict["artifact_path"] = "source"

    repo = Repository(db)
    job = await create_job(repo, job_type=JobType.upload, config=config_dict)

    return DocumentUploadAccepted(
        job_id=job["id"],
        filename=filename,
        status="processing",
    )


@router.post(
    "/convert/url",
    response_model=DocumentUploadAccepted,
    status_code=202,
    operation_id="convert_document_url",
    summary="Convert a document from URL via async processing job.",
)
async def convert_document_url(
    request: ConvertUrlRequest,
    db: asyncpg.Connection = Depends(_get_db),
) -> DocumentUploadAccepted:
    _validate_url(request.url)
    opts = request.options

    path = urlparse(request.url).path
    filename = _sanitize_filename(path.rsplit("/", 1)[-1] or "document")
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    source_bytes: bytes | None = None
    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, max_redirects=5
        ) as dl_client:
            dl_resp = await dl_client.get(request.url)
            if dl_resp.is_success and len(dl_resp.content) <= _MAX_UPLOAD_BYTES:
                source_bytes = dl_resp.content
    except (httpx.HTTPError, httpx.TimeoutException):
        logger.warning("Could not download source: %s", request.url)

    if not source_bytes:
        raise HTTPException(
            status_code=400,
            detail="Could not download document from URL",
        )

    config_dict = {
        "filename": filename,
        "chunker_type": opts.chunker_type.value,
        "chunk_size": opts.chunk_size,
        "chunk_overlap": opts.chunk_overlap,
    }

    client = MLflowClient(_tracking_uri())
    experiment_id = await client.ensure_experiment("amortized/uploads")
    run_id = await client.create_run(experiment_id, name=filename, tags={"job_type": "document"})
    await client.upload_artifact(run_id, f"source/{filename}", source_bytes)

    config_dict["mlflow_upload_run_id"] = run_id
    config_dict["artifact_path"] = "source"

    repo = Repository(db)
    job = await create_job(repo, job_type=JobType.upload, config=config_dict)

    return DocumentUploadAccepted(
        job_id=job["id"],
        filename=filename,
        status="processing",
    )


@router.get(
    "",
    response_model=list[DocumentSummary],
    operation_id="list_documents",
    summary="List processed documents from MLflow.",
)
async def list_documents() -> list[DocumentSummary]:
    client = MLflowClient(_tracking_uri())
    try:
        exp_ids = await client.list_experiment_ids()
        if not exp_ids:
            return []
        runs = await client.search_runs(
            experiment_ids=exp_ids,
            filter_string="tags.job_type = 'document' AND attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot connect to MLflow") from None
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="MLflow request timed out") from None

    results: list[DocumentSummary] = []
    for run in runs:
        info = run.get("info", {})
        tags = {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}
        run_id = info.get("run_id", "")
        fmt = tags.get("format", "md")
        ext = fmt if fmt != "text" else "txt"
        has_content = await client.artifact_exists(run_id, f"parsed_content.{ext}")
        results.append(
            DocumentSummary(
                document_id=run_id,
                mlflow_run_id=run_id,
                filename=tags.get("filename", info.get("run_name", "")),
                format=OutputFormat(fmt),
                created_at=_format_timestamp(info.get("start_time")),
                content_available=has_content,
            )
        )
    return results


@router.get(
    "/{document_id}/content",
    response_model=DocumentResult,
    operation_id="get_document_content",
    summary="Get parsed content of a document.",
)
async def get_document_content(document_id: str) -> DocumentResult:
    client = MLflowClient(_tracking_uri())
    try:
        try:
            run = await client.get_run(document_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {document_id}",
                ) from None
            raise
        info = run.get("info", {})
        tags = {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}

        fmt = tags.get("format", "md")
        ext = fmt if fmt != "text" else "txt"
        content = await client.get_artifact_text(document_id, f"parsed_content.{ext}")
        if content is None:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found for document {document_id}",
            )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot connect to MLflow") from None
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="MLflow request timed out") from None
    except HTTPException:
        raise
    except Exception:
        logger.warning("Failed to retrieve document %s", document_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document") from None

    return DocumentResult(
        document_id=document_id,
        mlflow_run_id=document_id,
        filename=tags.get("filename", info.get("run_name", "")),
        content=content,
        format=OutputFormat(fmt),
        processing_time=float(tags.get("processing_time", "0")),
        status="success",
    )


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunks,
    operation_id="get_document_chunks",
    summary="Get chunks for a document.",
)
async def get_document_chunks(document_id: str) -> DocumentChunks:
    client = MLflowClient(_tracking_uri())
    try:
        run = await client.get_run(document_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Document not found") from None
        raise

    tags = {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}
    filename = tags.get("filename", run.get("info", {}).get("run_name", ""))

    metadata_text = await client.get_artifact_text(document_id, "chunks/metadata.json")
    if not metadata_text:
        return DocumentChunks(document_id=document_id, filename=filename, chunks=[])

    metadata = json.loads(metadata_text)
    chunks: list[DocumentChunk] = []
    for i, meta in enumerate(metadata):
        text = await client.get_artifact_text(document_id, f"chunks/chunk_{i:03d}.md")
        chunks.append(
            DocumentChunk(
                chunk_index=meta.get("chunk_index", i),
                text=text or "",
                num_tokens=meta.get("num_tokens"),
                headings=meta.get("headings") or [],
                page_numbers=meta.get("page_numbers") or [],
            )
        )

    return DocumentChunks(document_id=document_id, filename=filename, chunks=chunks)


@router.delete(
    "/{document_id}",
    status_code=204,
    operation_id="delete_document",
)
async def delete_document(
    document_id: str,
) -> None:
    tracking_uri: str | None = None
    with contextlib.suppress(HTTPException):
        tracking_uri = _tracking_uri()

    if tracking_uri:
        try:
            client = MLflowClient(tracking_uri)
            await client.delete_run(document_id)
        except httpx.HTTPError:
            logger.warning("Failed to delete MLflow run %s", document_id, exc_info=True)
