"""Compute backend API endpoints."""

from fastapi import APIRouter, HTTPException

from amortized.core.compute import get_backend, list_backends

router = APIRouter(prefix="/api/v1/compute", tags=["compute"])


@router.get("")
async def list_compute_backends() -> list[dict[str, object]]:
    return list_backends()


@router.get("/{name}/status")
async def compute_backend_status(name: str) -> dict[str, object]:
    try:
        backend = get_backend(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Backend {name!r} not found") from None
    return {
        "name": backend.name,
        "capabilities": sorted(c.value for c in backend.capabilities()),
        "healthy": True,
    }
