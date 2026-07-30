"""Document processing endpoints — proxy to docling-serve with MLflow artifact storage."""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import aiosqlite
import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from amortized.config import settings
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import (
    ConvertUrlRequest,
    DocumentResult,
    DocumentSection,
    DocumentSections,
    DocumentSummary,
    OutputFormat,
)

logger = logging.getLogger("amortized.api.documents")

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

_FORMAT_MAP = {
    "md": "md_content",
    "text": "text_content",
    "json": "json_content",
    "html": "html_content",
}

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

_BLOCKED_HOSTNAMES = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.azure.com",
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
})


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


def _extract_content(document: dict[str, Any], output_format: str) -> str:
    key = _FORMAT_MAP.get(output_format, "md_content")
    content = document.get(key, "")
    if isinstance(content, dict):
        import json

        return json.dumps(content, indent=2)
    return str(content)


async def _store_in_mlflow(
    filename: str,
    content: str,
    output_format: str,
    source_bytes: bytes | None = None,
    processing_time: float = 0.0,
) -> str:
    tracking_uri = _tracking_uri()
    run_id: str | None = None
    async with httpx.AsyncClient(timeout=30.0) as client:
        experiment_name = "amortized/documents"
        resp = await client.get(
            f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": experiment_name},
        )
        if resp.status_code == 404 or "RESOURCE_DOES_NOT_EXIST" in resp.text:
            create_resp = await client.post(
                f"{tracking_uri}/api/2.0/mlflow/experiments/create",
                json={"name": experiment_name},
            )
            if create_resp.status_code == 409:
                refetch = await client.get(
                    f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name",
                    params={"experiment_name": experiment_name},
                )
                refetch.raise_for_status()
                experiment_id = refetch.json()["experiment"]["experiment_id"]
            else:
                create_resp.raise_for_status()
                experiment_id = create_resp.json()["experiment_id"]
        else:
            resp.raise_for_status()
            exp = resp.json()["experiment"]
            experiment_id = exp["experiment_id"]
            if exp.get("lifecycle_stage") == "deleted":
                await client.post(
                    f"{tracking_uri}/api/2.0/mlflow/experiments/restore",
                    json={"experiment_id": experiment_id},
                )
                logger.info("Restored deleted experiment %s", experiment_name)

        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        run_resp = await client.post(
            f"{tracking_uri}/api/2.0/mlflow/runs/create",
            json={
                "experiment_id": experiment_id,
                "run_name": filename,
                "start_time": now_ms,
                "tags": [
                    {"key": "job_type", "value": "document"},
                    {"key": "filename", "value": filename},
                    {"key": "format", "value": output_format},
                    {"key": "processing_time", "value": str(processing_time)},
                    {"key": "content_length", "value": str(len(content))},
                ],
            },
        )
        run_resp.raise_for_status()
        run_id = run_resp.json()["run"]["info"]["run_id"]

        try:
            if source_bytes:
                safe_name = _sanitize_filename(filename)
                src_resp = await client.put(
                    f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts"
                    f"/source/{safe_name}",
                    params={"run_id": run_id},
                    content=source_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                )
                src_resp.raise_for_status()

            ext = output_format if output_format != "text" else "txt"
            artifact_path = f"parsed_content.{ext}"
            artifact_resp = await client.put(
                f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{artifact_path}",
                params={"run_id": run_id},
                content=content.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )
            artifact_resp.raise_for_status()

            update_resp = await client.post(
                f"{tracking_uri}/api/2.0/mlflow/runs/update",
                json={
                    "run_id": run_id,
                    "status": "FINISHED",
                    "end_time": int(datetime.now(UTC).timestamp() * 1000),
                },
            )
            update_resp.raise_for_status()
        except Exception:
            try:
                await client.post(
                    f"{tracking_uri}/api/2.0/mlflow/runs/update",
                    json={"run_id": run_id, "status": "FAILED"},
                )
            except Exception:
                logger.debug("Could not mark MLflow run %s as FAILED", run_id)
            raise

        return run_id


async def _call_docling(
    client: httpx.AsyncClient, url: str, **kwargs: Any
) -> dict[str, Any]:
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
        logger.error(
            "Docling-serve returned %d: %s", resp.status_code, resp.text[:500]
        )
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        data = await _call_docling(
            client,
            f"{base_url}/v1/convert/file",
            files={
                "files": (
                    filename,
                    file_bytes,
                    "application/octet-stream",
                ),
            },
            data={
                "to_formats": output_format.value,
                "do_ocr": str(do_ocr).lower(),
                "ocr_engine": ocr_engine,
                "table_mode": table_mode,
            },
        )

    document = data.get("document", {})
    content = _extract_content(document, output_format.value)
    processing_time = data.get("processing_time", 0.0)
    status = data.get("status", "success")

    warnings: list[str] = []
    if not content.strip():
        warnings.append("Document parsed but no content was extracted")

    mlflow_run_id: str | None = None
    if settings.mlflow_tracking_uri:
        try:
            mlflow_run_id = await _store_in_mlflow(
                filename,
                content,
                output_format.value,
                source_bytes=file_bytes,
                processing_time=processing_time,
            )
        except HTTPException:
            raise
        except (httpx.HTTPError, httpx.TimeoutException):
            logger.warning("Failed to store document in MLflow", exc_info=True)
            warnings.append("Document converted but not stored in MLflow")

    document_id = mlflow_run_id or str(uuid.uuid4())
    return DocumentResult(
        document_id=document_id,
        mlflow_run_id=mlflow_run_id,
        filename=filename,
        content=content,
        format=output_format,
        processing_time=processing_time,
        status=status,
        warnings=warnings,
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        data = await _call_docling(
            client,
            f"{base_url}/v1/convert/source",
            json={
                "sources": [{"kind": "http", "url": request.url}],
                "options": {
                    "to_formats": [output_format.value],
                    "do_ocr": opts.do_ocr,
                    "ocr_engine": opts.ocr_engine,
                    "table_mode": opts.table_mode,
                },
            },
        )

    document = data.get("document", {})
    content = _extract_content(document, output_format.value)
    processing_time = data.get("processing_time", 0.0)
    status = data.get("status", "success")

    filename = _sanitize_filename(request.url.rsplit("/", 1)[-1] or "document")

    warnings: list[str] = []
    if not content.strip():
        warnings.append("Document parsed but no content was extracted")

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

    mlflow_run_id: str | None = None
    if settings.mlflow_tracking_uri:
        try:
            mlflow_run_id = await _store_in_mlflow(
                filename,
                content,
                output_format.value,
                source_bytes=source_bytes,
                processing_time=processing_time,
            )
        except HTTPException:
            raise
        except (httpx.HTTPError, httpx.TimeoutException):
            logger.warning("Failed to store document in MLflow", exc_info=True)
            warnings.append("Document converted but not stored in MLflow")

    document_id = mlflow_run_id or str(uuid.uuid4())
    return DocumentResult(
        document_id=document_id,
        mlflow_run_id=mlflow_run_id,
        filename=filename,
        content=content,
        format=output_format,
        processing_time=processing_time,
        status=status,
        warnings=warnings,
    )


@router.get(
    "",
    response_model=list[DocumentSummary],
    operation_id="list_documents",
    summary="List processed documents from MLflow.",
)
async def list_documents() -> list[DocumentSummary]:
    tracking_uri = _tracking_uri()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            exp_resp = await client.get(
                f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": "amortized/documents"},
            )
            if (
                exp_resp.status_code == 404
                or "RESOURCE_DOES_NOT_EXIST" in exp_resp.text
            ):
                return []
            exp_resp.raise_for_status()
            experiment_id = exp_resp.json()["experiment"]["experiment_id"]

            runs_resp = await client.post(
                f"{tracking_uri}/api/2.0/mlflow/runs/search",
                json={
                    "experiment_ids": [experiment_id],
                    "filter": "tags.job_type = 'document' AND attributes.status = 'FINISHED'",
                    "order_by": ["start_time DESC"],
                    "max_results": 100,
                },
            )
            runs_resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502, detail="Cannot connect to MLflow"
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, detail="MLflow request timed out"
        ) from None

    results: list[DocumentSummary] = []
    for run in runs_resp.json().get("runs", []):
        info = run.get("info", {})
        tags = {
            t["key"]: t["value"]
            for t in run.get("data", {}).get("tags", [])
        }
        results.append(
            DocumentSummary(
                document_id=info.get("run_id", ""),
                mlflow_run_id=info.get("run_id", ""),
                filename=tags.get("filename", info.get("run_name", "")),
                format=OutputFormat(tags.get("format", "md")),
                created_at=_format_timestamp(info.get("start_time")),
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
    tracking_uri = _tracking_uri()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            run_resp = await client.get(
                f"{tracking_uri}/api/2.0/mlflow/runs/get",
                params={"run_id": document_id},
            )
            if run_resp.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {document_id}",
                )
            run_resp.raise_for_status()
            run = run_resp.json()["run"]
            info = run.get("info", {})
            tags = {
                t["key"]: t["value"]
                for t in run.get("data", {}).get("tags", [])
            }

            fmt = tags.get("format", "md")
            ext = fmt if fmt != "text" else "txt"
            artifact_path = f"parsed_content.{ext}"
            content_resp = await client.get(
                f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts"
                f"/{artifact_path}",
                params={"run_id": document_id},
            )
            if content_resp.is_error:
                raise HTTPException(
                    status_code=404,
                    detail=f"Artifact not found for document {document_id}",
                )
            content = content_resp.text
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502, detail="Cannot connect to MLflow"
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, detail="MLflow request timed out"
        ) from None
    except HTTPException:
        raise
    except Exception:
        logger.warning(
            "Failed to retrieve document %s", document_id, exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve document"
        ) from None

    return DocumentResult(
        document_id=document_id,
        mlflow_run_id=document_id,
        filename=tags.get("filename", info.get("run_name", "")),
        content=content,
        format=OutputFormat(fmt),
        processing_time=float(tags.get("processing_time", "0")),
        status="success",
    )


def _extract_sections(content: str) -> list[DocumentSection]:
    """Parse markdown headings and return structured sections."""
    import re

    lines = content.split("\n")
    sections: list[DocumentSection] = []
    current_heading = ""
    current_level = 0
    current_lines: list[str] = []

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            if current_heading:
                body = "\n".join(current_lines).strip()
                preview = body[:200].strip()
                if len(body) > 200:
                    preview += "..."
                sections.append(DocumentSection(
                    heading=current_heading,
                    level=current_level,
                    char_count=len(body),
                    preview=preview,
                ))
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading:
        body = "\n".join(current_lines).strip()
        preview = body[:200].strip()
        if len(body) > 200:
            preview += "..."
        sections.append(DocumentSection(
            heading=current_heading,
            level=current_level,
            char_count=len(body),
            preview=preview,
        ))

    return sections


@router.get(
    "/{document_id}/sections",
    response_model=DocumentSections,
    operation_id="get_document_sections",
    summary="Get section headings and structure of a document. Use this to understand document layout before building SDG configs.",
)
async def get_document_sections(document_id: str) -> DocumentSections:
    result = await get_document_content(document_id)
    sections = _extract_sections(result.content)
    return DocumentSections(
        document_id=document_id,
        filename=result.filename,
        total_chars=len(result.content),
        sections=sections,
    )


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
    db: aiosqlite.Connection = Depends(_get_db),
) -> None:
    tracking_uri: str | None = None
    with contextlib.suppress(HTTPException):
        tracking_uri = _tracking_uri()

    if tracking_uri:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{tracking_uri}/api/2.0/mlflow/runs/delete",
                    json={"run_id": document_id},
                )
                if resp.status_code != 404:
                    resp.raise_for_status()
        except httpx.HTTPError:
            logger.warning("Failed to delete MLflow run %s", document_id, exc_info=True)

    repo = Repository(db)
    await repo.delete_document(document_id)
