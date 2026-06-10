"""Settings API — API key management and backend configuration."""

import asyncio
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

_config_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------


@router.post("/api-keys", status_code=201, response_model=ApiKeyInfo)
async def add_api_key(
    body: ApiKeyCreate,
    db: aiosqlite.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Store an LLM provider API key.

    The server uses it for SDG and eval jobs when no key is in the job
    config. Supports any LiteLLM provider (openai, anthropic, google).
    """
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
    """List stored API keys (redacted).

    Returns provider names and last 4 chars only — never the full key.
    Check which providers are configured before submitting jobs.
    """
    repo = Repository(db)
    return await repo.list_api_keys()


@router.delete("/api-keys/{key_id}", status_code=204)
async def delete_api_key(
    key_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> None:
    """Remove a stored API key by ID."""
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
    """Register a compute backend for job dispatch.

    SSH backends connect to remote GPU nodes. Available immediately
    and persisted across server restarts.
    """
    if body.name == "local":
        raise HTTPException(400, detail="Cannot use reserved name 'local'")

    if body.type == "ssh":
        from amortized.backends.ssh import SSHBackend

        backend = SSHBackend(
            host=body.host,
            user=body.user,
            key_path=body.key_path,
            remote_base_dir=body.remote_base_dir,
            name=body.name,
            container_runtime=body.container_runtime,
        )
        register_backend(backend)

        async with _config_lock:
            config = _load_config()
            config.setdefault("compute", {}).setdefault("backends", {})[body.name] = {
                "type": body.type,
                "host": body.host,
                "user": body.user,
                "key_path": body.key_path,
                "remote_base_dir": body.remote_base_dir,
                "container_runtime": body.container_runtime,
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
    """List registered compute backends with their capabilities."""
    return list_backends()


@router.delete("/backends/{name}", status_code=204)
async def delete_backend(name: str) -> None:
    """Remove a compute backend. The built-in 'local' backend cannot be removed."""
    if name == "local":
        raise HTTPException(
            status_code=400,
            detail="Cannot remove the built-in local backend",
        )
    removed = unregister_backend(name)
    if not removed:
        raise HTTPException(status_code=404, detail="Backend not found")

    async with _config_lock:
        config = _load_config()
        backends = config.get("compute", {}).get("backends", {})
        backends.pop(name, None)
        _save_config(config)


@router.post("/backends/{name}/test")
async def test_backend(name: str) -> dict[str, Any]:
    """Test connectivity to a compute backend.

    For SSH backends, attempts a connection and queries GPU info via
    nvidia-smi. Returns healthy status and GPU details if available.
    """
    from amortized.core.compute import get_backend

    try:
        backend = get_backend(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Backend not found") from None

    result: dict[str, Any] = {
        "name": backend.name,
        "capabilities": sorted(c.value for c in backend.capabilities()),
        "healthy": True,
    }

    if hasattr(backend, "_connect"):
        try:
            conn = await backend._connect()
            try:
                nvsmi = (
                    "nvidia-smi --query-gpu=name,memory.total"
                    " --format=csv,noheader,nounits"
                    " 2>/dev/null || echo 'no-gpu'"
                )
                gpu_result = await conn.run(nvsmi)
                gpu_info = gpu_result.stdout.strip() if gpu_result.stdout else ""
                if gpu_info and gpu_info != "no-gpu":
                    result["gpu_info"] = gpu_info
            finally:
                conn.close()
        except Exception as exc:
            result["healthy"] = False
            result["error"] = str(exc)

    return result
