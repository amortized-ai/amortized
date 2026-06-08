"""Settings API — API key management and backend configuration."""

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from amortized.core.compute import (
    list_backends,
    register_backend,
    unregister_backend,
)
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import (
    ApiKeyCreate,
    ApiKeyInfo,
    BackendCreate,
    ComputeBackendInfo,
)

logger = logging.getLogger("amortized.api.settings")

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------


@router.post("/api-keys", status_code=201, response_model=ApiKeyInfo)
async def add_api_key(
    body: ApiKeyCreate,
    db: aiosqlite.Connection = Depends(_get_db),
) -> dict[str, Any]:
    repo = Repository(db)
    row = await repo.create_api_key(
        key_id=str(uuid.uuid4()),
        name=body.name,
        provider=body.provider,
        key_value=body.key,
        created_at=datetime.now(UTC).isoformat(),
    )
    return row


@router.get("/api-keys", response_model=list[ApiKeyInfo])
async def list_api_keys(
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[dict[str, Any]]:
    repo = Repository(db)
    return await repo.list_api_keys()


@router.delete("/api-keys/{key_id}", status_code=204)
async def delete_api_key(
    key_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> None:
    repo = Repository(db)
    deleted = await repo.delete_api_key(key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _load_config() -> dict[str, Any]:
    config_path = Path.home() / ".amortized" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(config_path.read_text()) or {}
    except ImportError:
        return {}


def _save_config(config: dict[str, Any]) -> None:
    import yaml

    config_path = Path.home() / ".amortized" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False))


@router.post("/backends", status_code=201, response_model=ComputeBackendInfo)
async def add_backend(body: BackendCreate) -> dict[str, Any]:
    if body.type == "ssh":
        from amortized.backends.ssh import SSHBackend

        backend = SSHBackend(
            host=body.host,
            user=body.user,
            key_path=body.key_path,
            remote_base_dir=body.remote_base_dir,
        )
        backend.name = body.name
        register_backend(backend)

        config = _load_config()
        config.setdefault("compute", {}).setdefault("backends", {})[body.name] = {
            "type": body.type,
            "host": body.host,
            "user": body.user,
            "key_path": body.key_path,
            "remote_base_dir": body.remote_base_dir,
        }
        _save_config(config)

        return {
            "name": body.name,
            "capabilities": sorted(c.value for c in backend.capabilities()),
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported backend type: {body.type!r}",
    )


@router.get("/backends", response_model=list[ComputeBackendInfo])
async def list_settings_backends() -> list[dict[str, object]]:
    return list_backends()


@router.delete("/backends/{name}", status_code=204)
async def delete_backend(name: str) -> None:
    if name == "local":
        raise HTTPException(
            status_code=400,
            detail="Cannot remove the built-in local backend",
        )
    removed = unregister_backend(name)
    if not removed:
        raise HTTPException(status_code=404, detail="Backend not found")

    config = _load_config()
    backends = config.get("compute", {}).get("backends", {})
    backends.pop(name, None)
    _save_config(config)


@router.post("/backends/{name}/test", response_model=ComputeBackendInfo)
async def test_backend(name: str) -> dict[str, Any]:
    from amortized.core.compute import get_backend

    try:
        backend = get_backend(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Backend not found") from None

    return {
        "name": backend.name,
        "capabilities": sorted(c.value for c in backend.capabilities()),
    }
