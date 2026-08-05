"""Document processing endpoints — proxy to docling-serve with MLflow artifact storage."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query, UploadFile

from amortized.config import settings
from amortized.core.mlflow_client import MLflowClient
from amortized.models import (
    ConvertUrlRequest,
    DocumentChunk,
    DocumentChunks,
    DocumentResult,
    DocumentSummary,
    OutputFormat,
)

logger = logging.getLogger("amortized.api.documents")

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

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


def _docling_url() -> str:
    if not settings.docling_url:
        raise HTTPException(status_code=503, detail="Document processing service is not configured")
    return settings.docling_url.rstrip("/")


def _tracking_uri() -> str:
    if not settings.mlflow_tracking_uri:
        raise HTTPException(status_code=503, detail="MLflow tracking is not configured")
    return settings.mlflow_tracking_uri.rstrip("/")


async def _store_in_mlflow(
    filename: str,
    content: str,
    output_format: str,
    source_bytes: bytes | None = None,
    processing_time: float = 0.0,
    chunks: list[dict[str, Any]] | None = None,
) -> str:
    client = MLflowClient(_tracking_uri())
    experiment_id = await client.ensure_experiment("amortized/documents")
    chunk_count = len(chunks) if chunks else 0
    run_id = await client.create_run(
        experiment_id,
        name=filename,
        tags={
            "job_type": "document",
            "filename": filename,
            "format": output_format,
            "processing_time": str(processing_time),
            "content_length": str(len(content)),
            "chunk_count": str(chunk_count),
        },
    )

    try:
        if source_bytes:
            safe_name = _sanitize_filename(filename)
            await client.upload_artifact(run_id, f"source/{safe_name}", source_bytes)

        ext = output_format if output_format != "text" else "txt"
        await client.upload_artifact(
            run_id,
            f"parsed_content.{ext}",
            content.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
        )

        if chunks:
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")
                await client.upload_artifact(
                    run_id,
                    f"chunks/chunk_{i:03d}.md",
                    chunk_text.encode("utf-8"),
                    content_type="text/plain; charset=utf-8",
                )
            metadata = [
                {
                    "chunk_index": c.get("chunk_index", i),
                    "num_tokens": c.get("num_tokens"),
                    "headings": c.get("headings"),
                    "page_numbers": c.get("page_numbers"),
                }
                for i, c in enumerate(chunks)
            ]
            await client.upload_artifact(
                run_id,
                "chunks/metadata.json",
                json.dumps(metadata, indent=2).encode("utf-8"),
                content_type="application/json",
            )

        await client.finish_run(run_id)
    except Exception:
        await client.fail_run_quiet(run_id)
        raise

    return run_id


async def _call_docling(client: httpx.AsyncClient, url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        resp = await client.post(url, **kwargs)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail="Document processing service is temporarily unavailable",
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, detail="Document processing request timed out"
        ) from None
    except httpx.TransportError:
        raise HTTPException(
            status_code=502, detail="Document processing service communication error"
        ) from None
    if resp.is_error:
        logger.error("Docling-serve returned %d: %s", resp.status_code, resp.text[:500])
        raise HTTPException(
            status_code=502, detail=f"Document processing error: {resp.status_code}"
        )
    try:
        return resp.json()  # type: ignore[no-any-return]
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Document processing service returned non-JSON response",
        ) from None


_DEFAULT_CHUNK_MAX_TOKENS = 2048
_DOCLING_TIMEOUT = 600.0


_DOCLING_LABEL_PREFIX: dict[str, str] = {
    "title": "# ",
    "subtitle-level-1": "## ",
    "section-header": "## ",
}


def _docling_json_to_markdown(doc_json: dict[str, Any]) -> str:
    """Reconstruct markdown from a DoclingDocument JSON export."""
    parts: list[str] = []
    for node in doc_json.get("texts", []):
        label = node.get("label", "text")
        text = node.get("text", "")
        if not text:
            continue
        prefix = _DOCLING_LABEL_PREFIX.get(label, "")
        parts.append(f"{prefix}{text}")
    return "\n\n".join(parts)


async def _convert_and_chunk(
    base_url: str,
    filename: str,
    file_bytes: bytes,
    max_tokens: int = _DEFAULT_CHUNK_MAX_TOKENS,
) -> tuple[str, list[dict[str, Any]], float]:
    """Single call to docling-serve: parse, chunk, and return (content, chunks, time)."""
    async with httpx.AsyncClient(timeout=_DOCLING_TIMEOUT) as client:
        data = await _call_docling(
            client,
            f"{base_url}/v1/chunk/hybrid/file",
            files={"files": (filename, file_bytes, "application/octet-stream")},
            data={
                "chunking_options": json.dumps({"max_tokens": max_tokens}),
                "include_converted_doc": "true",
            },
        )

    chunks = data.get("chunks")
    if chunks is None:
        logger.warning("Docling response missing 'chunks' key for %s", filename)
        chunks = []

    content = ""
    documents = data.get("documents", [])
    if documents:
        doc_content = documents[0].get("content", {})
        md = doc_content.get("md_content")
        if md:
            content = md
        else:
            doc_json = doc_content.get("json_content")
            if doc_json:
                content = _docling_json_to_markdown(doc_json)

    processing_time: float = data.get("processing_time", 0.0)
    return content, chunks, processing_time


async def _store_and_build(
    filename: str,
    content: str,
    output_format: OutputFormat,
    source_bytes: bytes | None,
    processing_time: float,
    chunks: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Store in MLflow, appending any warnings. Returns (chunks, mlflow_run_id)."""
    mlflow_run_id: str | None = None
    if settings.mlflow_tracking_uri:
        try:
            mlflow_run_id = await _store_in_mlflow(
                filename,
                content,
                output_format.value,
                source_bytes=source_bytes,
                processing_time=processing_time,
                chunks=chunks,
            )
        except HTTPException:
            raise
        except httpx.HTTPError:
            logger.warning("Failed to store document in MLflow", exc_info=True)
            warnings.append("Document converted but not stored in MLflow")

    return chunks, mlflow_run_id


def _build_result(
    filename: str,
    content: str,
    output_format: OutputFormat,
    processing_time: float,
    status: str,
    warnings: list[str],
    chunks: list[dict[str, Any]],
    mlflow_run_id: str | None,
) -> DocumentResult:
    return DocumentResult(
        document_id=mlflow_run_id or str(uuid.uuid4()),
        mlflow_run_id=mlflow_run_id,
        filename=filename,
        content=content,
        chunk_count=len(chunks),
        format=output_format,
        processing_time=processing_time,
        status=status,
        warnings=warnings,
    )


@router.post(
    "/convert",
    response_model=DocumentResult,
    operation_id="convert_document",
    summary="Upload and convert a document via docling-serve.",
)
async def convert_document(
    file: UploadFile,
    output_format: OutputFormat = OutputFormat.md,
    do_ocr: bool = True,
    ocr_engine: str = "easyocr",
    table_mode: str = "fast",
    chunk_max_tokens: int = Query(_DEFAULT_CHUNK_MAX_TOKENS, ge=64, le=8192),
) -> DocumentResult:
    base_url = _docling_url()
    filename = _sanitize_filename(file.filename or f"upload-{uuid.uuid4().hex[:8]}")
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes, max {_MAX_UPLOAD_BYTES})",
        )

    warnings: list[str] = []
    content, chunks, processing_time = await _convert_and_chunk(
        base_url, filename, file_bytes, chunk_max_tokens
    )

    if not content.strip():
        warnings.append("Document parsed but no content was extracted")

    chunks, mlflow_run_id = await _store_and_build(
        filename,
        content,
        output_format,
        source_bytes=file_bytes,
        processing_time=processing_time,
        chunks=chunks,
        warnings=warnings,
    )

    return _build_result(
        filename,
        content,
        output_format,
        processing_time,
        "success",
        warnings,
        chunks,
        mlflow_run_id,
    )


@router.post(
    "/convert/url",
    response_model=DocumentResult,
    operation_id="convert_document_url",
    summary="Convert a document from URL via docling-serve.",
)
async def convert_document_url(request: ConvertUrlRequest) -> DocumentResult:
    _validate_url(request.url)
    base_url = _docling_url()
    opts = request.options
    output_format = opts.output_format

    filename = _sanitize_filename(request.url.rsplit("/", 1)[-1] or "document")
    warnings: list[str] = []

    source_bytes: bytes | None = None
    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, max_redirects=5
        ) as dl_client:
            dl_resp = await dl_client.get(request.url)
            if dl_resp.is_success and len(dl_resp.content) <= _MAX_UPLOAD_BYTES:
                source_bytes = dl_resp.content
    except (httpx.HTTPError, httpx.TimeoutException):
        logger.warning("Could not download source for archival: %s", request.url)
        warnings.append("Source document could not be archived")

    if not source_bytes:
        raise HTTPException(
            status_code=400,
            detail="Could not download document from URL",
        )

    content, chunks, processing_time = await _convert_and_chunk(
        base_url, filename, source_bytes, opts.chunk_max_tokens
    )

    if not content.strip():
        warnings.append("Document parsed but no content was extracted")

    chunks, mlflow_run_id = await _store_and_build(
        filename,
        content,
        output_format,
        source_bytes=source_bytes,
        processing_time=processing_time,
        chunks=chunks,
        warnings=warnings,
    )

    return _build_result(
        filename,
        content,
        output_format,
        processing_time,
        "success",
        warnings,
        chunks,
        mlflow_run_id,
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
        experiment_id = await client.get_experiment("amortized/documents")
        if experiment_id is None:
            return []
        runs = await client.search_runs(
            experiment_ids=[experiment_id],
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


def _format_timestamp(ts: int | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts / 1000, tz=UTC).isoformat()


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
