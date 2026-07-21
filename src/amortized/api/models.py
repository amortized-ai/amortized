"""Gateway model discovery endpoint."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter

from amortized.config import settings
from amortized.models import GatewayModel, ModelsResponse

logger = logging.getLogger("amortized.api.models")

router = APIRouter(prefix="/api/v1", tags=["models"])

_models_cache: list[GatewayModel] | None = None
_models_cache_time: float = 0
_CACHE_TTL = 60
_cache_lock = asyncio.Lock()


async def _fetch_gateway_models() -> list[GatewayModel]:
    global _models_cache, _models_cache_time

    now = datetime.now(UTC).timestamp()
    if _models_cache is not None and now - _models_cache_time < _CACHE_TTL:
        return _models_cache

    async with _cache_lock:
        now = datetime.now(UTC).timestamp()
        if _models_cache is not None and now - _models_cache_time < _CACHE_TTL:
            return _models_cache

        tracking_uri = settings.mlflow_tracking_uri
        if not tracking_uri:
            _models_cache = []
            _models_cache_time = now
            return _models_cache

        url = f"{tracking_uri.rstrip('/')}/api/2.0/endpoints"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.warning("Failed to fetch gateway endpoints from %s", url)
            _models_cache = []
            _models_cache_time = now
            return _models_cache

        models: list[GatewayModel] = []
        for endpoint in data.get("endpoints", []):
            model_info = endpoint.get("model", {})
            if isinstance(model_info, str):
                model_info = {"name": model_info}
            models.append(
                GatewayModel(
                    name=endpoint.get("name", ""),
                    provider=model_info.get("provider", ""),
                    endpoint_type=endpoint.get("endpoint_type", ""),
                )
            )

        _models_cache = models
        _models_cache_time = now
        return _models_cache


@router.get(
    "/models",
    response_model=ModelsResponse,
    operation_id="list_models",
    summary="List available models from the MLflow AI Gateway.",
)
async def list_models() -> ModelsResponse:
    models = await _fetch_gateway_models()
    return ModelsResponse(
        models=models,
        gateway_url=settings.gateway_url,
    )
