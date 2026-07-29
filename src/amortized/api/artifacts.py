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
    ".parquet": "application/octet-stream",
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


def _read_s3_object(bucket: str, s3_key: str) -> bytes:
    from botocore.exceptions import ClientError

    try:
        s3 = _get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        return obj["Body"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {s3_key}",
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


@router.get(
    "/{experiment_id}/{run_id}",
    operation_id="list_artifacts",
    summary="List artifact files for a run.",
)
async def list_artifacts(
    experiment_id: str,
    run_id: str,
    path: str = "",
) -> dict:
    bucket = _get_bucket()
    prefix = f"mlflow/{experiment_id}/{run_id}/artifacts/"
    if path:
        prefix += path.rstrip("/") + "/"

    try:
        s3 = _get_s3_client()
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):]
            if relative:
                files.append({
                    "path": (path + "/" + relative).lstrip("/") if path else relative,
                    "file_size": obj.get("Size", 0),
                })
        return {"files": files}
    except Exception as exc:
        logger.warning("S3 list failed for %s/%s: %s", bucket, prefix, exc)
        raise HTTPException(
            status_code=502, detail=f"Failed to list artifacts: {exc}",
        ) from None


@router.get(
    "/{experiment_id}/{run_id}/{path:path}",
    operation_id="get_artifact_content",
    summary="Fetch an artifact file directly from S3. Parquet files are returned as JSON.",
)
async def get_artifact_content(
    experiment_id: str,
    run_id: str,
    path: str,
) -> Response:
    bucket = _get_bucket()
    s3_key = f"mlflow/{experiment_id}/{run_id}/artifacts/{path}"
    body = _read_s3_object(bucket, s3_key)

    if path.endswith(".parquet"):
        import io

        import pyarrow.parquet as pq
        from fastapi.responses import JSONResponse

        table = pq.read_table(io.BytesIO(body))
        records = table.to_pylist()
        return JSONResponse(content=records)

    content_type = _content_type_for(path)
    return Response(content=body, media_type=content_type)
