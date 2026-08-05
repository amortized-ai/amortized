"""Artifact proxy — stream files from MLflow's artifact store."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from amortized.config import settings
from amortized.core.mlflow_client import MLflowClient

logger = logging.getLogger("amortized.api.artifacts")

router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])

_CONTENT_TYPES: dict[str, str] = {
    ".jsonl": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".yaml": "text/yaml; charset=utf-8",
    ".yml": "text/yaml; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".parquet": "application/octet-stream",
}


def _content_type_for(path: str) -> str:
    for ext, ct in _CONTENT_TYPES.items():
        if path.endswith(ext):
            return ct
    return "application/octet-stream"


def _get_mlflow_client() -> MLflowClient:
    if not settings.mlflow_tracking_uri:
        raise HTTPException(
            status_code=503,
            detail="MLflow tracking URI is not configured (set AMORTIZED_MLFLOW_TRACKING_URI)",
        )
    return MLflowClient(settings.mlflow_tracking_uri)


@router.get(
    "/{experiment_id}/{run_id}",
    operation_id="list_artifacts",
    summary="List artifact files for a run.",
)
async def list_artifacts(
    experiment_id: str,
    run_id: str,
    path: str = "",
) -> dict[str, Any]:
    client = _get_mlflow_client()
    try:
        files = await client.list_artifacts(run_id, path)
        return {"files": files}
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Artifacts not found for run {run_id}",
            ) from None
        logger.warning("MLflow list_artifacts failed for run %s: %s", run_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to list artifacts from MLflow: {exc}",
        ) from None
    except httpx.ConnectError as exc:
        logger.warning("MLflow connection failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to MLflow: {exc}",
        ) from None


@router.get(
    "/{experiment_id}/{run_id}/{path:path}",
    operation_id="get_artifact_content",
    summary="Fetch an artifact file from MLflow. Parquet files are returned as JSON.",
)
async def get_artifact_content(
    experiment_id: str,
    run_id: str,
    path: str,
) -> Response:
    client = _get_mlflow_client()
    try:
        body = await client.get_artifact(run_id, path)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {path}",
            ) from None
        logger.warning("MLflow get_artifact failed for run %s path %s: %s", run_id, path, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to read artifact from MLflow: {exc}",
        ) from None
    except httpx.ConnectError as exc:
        logger.warning("MLflow connection failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to MLflow: {exc}",
        ) from None

    if path.endswith(".parquet"):
        import io

        import pyarrow.parquet as pq
        from fastapi.responses import JSONResponse

        table = pq.read_table(io.BytesIO(body))  # type: ignore[no-untyped-call]
        records = table.to_pylist()
        return JSONResponse(content=records)

    content_type = _content_type_for(path)
    return Response(content=body, media_type=content_type)
