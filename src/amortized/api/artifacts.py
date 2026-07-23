"""Artifact proxy — stream files from S3 (MinIO) without mlflow-artifacts."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from amortized.config import settings

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
}


def _content_type_for(path: str) -> str:
    for ext, ct in _CONTENT_TYPES.items():
        if path.endswith(ext):
            return ct
    return "application/octet-stream"


def _get_s3_client():  # type: ignore[no-untyped-def]
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint_url = (
        os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("MLFLOW_S3_ENDPOINT_URL") or ""
    )
    if not endpoint_url:
        raise HTTPException(
            status_code=503,
            detail="S3 endpoint is not configured"
            " (set AWS_S3_ENDPOINT_URL or MLFLOW_S3_ENDPOINT_URL)",
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        config=BotoConfig(signature_version="s3v4"),
    )


def _get_bucket() -> str:
    bucket = settings.storage_bucket or os.environ.get("AMORTIZED_STORAGE_BUCKET", "amortized")
    if not bucket:
        raise HTTPException(status_code=503, detail="Storage bucket is not configured")
    return bucket


@router.get(
    "/{experiment_id}/{run_id}/{path:path}",
    operation_id="get_artifact_content",
    summary="Fetch an artifact file directly from S3.",
)
async def get_artifact_content(
    experiment_id: str,
    run_id: str,
    path: str,
) -> Response:
    bucket = _get_bucket()
    s3_key = f"mlflow/{experiment_id}/{run_id}/artifacts/{path}"

    try:
        from botocore.exceptions import ClientError

        s3 = _get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        body = obj["Body"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {path}",
            ) from None
        logger.warning("S3 read failed for %s/%s: %s", bucket, s3_key, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to read artifact from S3: {exc}",
        ) from None
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("S3 read failed for %s/%s: %s", bucket, s3_key, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to read artifact from S3: {exc}",
        ) from None

    content_type = _content_type_for(path)
    return Response(content=body, media_type=content_type)
