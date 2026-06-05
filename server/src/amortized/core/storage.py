"""Storage backend for artifact upload/download with pre-signed URLs."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from amortized.config import settings

logger = logging.getLogger("amortized.core.storage")

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None
    BotoConfig = None

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    gcs_storage = None


class StorageBackend(Protocol):
    def generate_upload_url(
        self,
        key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> dict[str, Any]: ...

    def generate_download_url(self, key: str, expires_in: int = 3600) -> str: ...


class LocalStorage:
    def generate_upload_url(
        self,
        key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> dict[str, Any]:
        raise NotImplementedError("Local storage does not support pre-signed URLs")

    def generate_download_url(self, key: str, expires_in: int = 3600) -> str:
        raise NotImplementedError("Local storage does not support pre-signed URLs")


class S3Storage:
    def __init__(self, bucket: str, prefix: str, region: str) -> None:
        if boto3 is None:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        self._bucket = bucket
        self._prefix = prefix
        self._client = boto3.client(
            "s3",
            region_name=region,
            config=BotoConfig(signature_version="s3v4"),
        )

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def generate_upload_url(
        self,
        key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> dict[str, Any]:
        full_key = self._full_key(key)
        url = self._client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self._bucket,
                "Key": full_key,
                "ContentType": content_type,
            },
            ExpiresIn=expires_in,
        )
        return {
            "url": url,
            "method": "PUT",
            "headers": {"Content-Type": content_type},
        }

    def generate_download_url(self, key: str, expires_in: int = 3600) -> str:
        full_key = self._full_key(key)
        return str(
            self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": full_key},
                ExpiresIn=expires_in,
            )
        )


class GCSStorage:
    def __init__(self, bucket: str, prefix: str) -> None:
        if gcs_storage is None:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install google-cloud-storage"
            )
        self._client = gcs_storage.Client()
        self._bucket = self._client.bucket(bucket)
        self._prefix = prefix

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def generate_upload_url(
        self,
        key: str,
        content_type: str,
        expires_in: int = 3600,
    ) -> dict[str, Any]:
        import datetime

        blob = self._bucket.blob(self._full_key(key))
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(seconds=expires_in),
            method="PUT",
            content_type=content_type,
        )
        return {
            "url": url,
            "method": "PUT",
            "headers": {"Content-Type": content_type},
        }

    def generate_download_url(self, key: str, expires_in: int = 3600) -> str:
        import datetime

        blob = self._bucket.blob(self._full_key(key))
        return str(
            blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(seconds=expires_in),
                method="GET",
            )
        )


_storage: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _storage
    if _storage is not None:
        return _storage

    backend = settings.storage_backend
    if backend == "s3":
        _storage = S3Storage(
            settings.storage_bucket,
            settings.storage_prefix,
            settings.storage_region,
        )
    elif backend == "gcs":
        _storage = GCSStorage(settings.storage_bucket, settings.storage_prefix)
    else:
        _storage = LocalStorage()

    return _storage


def reset_storage() -> None:
    global _storage
    _storage = None
