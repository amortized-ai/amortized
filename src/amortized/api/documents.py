"""Document processing endpoints — proxy to docling-serve with MLflow artifact storage."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, UploadFile

from amortized.config import settings
from amortized.models import (
    ConvertUrlRequest,
    DocumentResult,
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


def _docling_url() -> str:
    if not settings.docling_url:
        raise HTTPException(status_code=503, detail="Docling-serve is not configured")
    return settings.docling_url.rstrip("/")


def _tracking_uri() -> str:
    if not settings.mlflow_tracking_uri:
        raise HTTPException(
            status_code=503, detail="MLflow tracking is not configured"
        )
    return settings.mlflow_tracking_uri.rstrip("/")


def _extract_content(document: dict[str, Any], output_format: str) -> str:
    key = _FORMAT_MAP.get(output_format, "md_content")
    content = document.get(key, "")
    if isinstance(content, dict):
        import json

        return json.dumps(content, indent=2)
    return str(content)


async def _store_in_mlflow(
    filename: str, content: str, output_format: str
) -> str:
    tracking_uri = _tracking_uri()
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
                refetch = await client.post(
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
            experiment_id = resp.json()["experiment"]["experiment_id"]

        run_resp = await client.post(
            f"{tracking_uri}/api/2.0/mlflow/runs/create",
            json={
                "experiment_id": experiment_id,
                "run_name": filename,
                "tags": [
                    {"key": "job_type", "value": "document"},
                    {"key": "filename", "value": filename},
                    {"key": "format", "value": output_format},
                ],
            },
        )
        run_resp.raise_for_status()
        run_id: str = run_resp.json()["run"]["info"]["run_id"]

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

        return run_id


async def _call_docling(
    client: httpx.AsyncClient, url: str, **kwargs: Any
) -> dict[str, Any]:
    try:
        resp = await client.post(url, **kwargs)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to docling-serve at {settings.docling_url}",
        ) from None
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, detail="Docling-serve request timed out"
        ) from None
    if resp.is_error:
        logger.error(
            "Docling-serve returned %d: %s", resp.status_code, resp.text[:500]
        )
        raise HTTPException(
            status_code=502, detail=f"Docling-serve error: {resp.status_code}"
        )
    try:
        return resp.json()  # type: ignore[no-any-return]
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Docling-serve returned non-JSON response",
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
    filename = file.filename or f"upload-{uuid.uuid4().hex[:8]}"
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
                    file.content_type or "application/octet-stream",
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
    mlflow_run_id: str | None = None
    if settings.mlflow_tracking_uri:
        try:
            mlflow_run_id = await _store_in_mlflow(
                filename, content, output_format.value
            )
        except Exception:
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

    filename = request.url.rsplit("/", 1)[-1] or "document"

    warnings: list[str] = []
    mlflow_run_id: str | None = None
    if settings.mlflow_tracking_uri:
        try:
            mlflow_run_id = await _store_in_mlflow(
                filename, content, output_format.value
            )
        except Exception:
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
                    "filter": "tags.job_type = 'document'",
                    "order_by": ["start_time DESC"],
                    "max_results": 100,
                },
            )
            runs_resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502, detail="Cannot connect to MLflow"
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
        processing_time=0.0,
        status="success",
    )


def _format_timestamp(ts: int | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts / 1000, tz=UTC).isoformat()
