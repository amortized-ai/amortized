"""Model discovery endpoint.

Prefers the MLflow AI Gateway when it is available. MLflow distributions without
gateway support (e.g. the RHOAI enterprise MLflow, 3.14.x) return no endpoints, so
we fall back to direct providers enabled by dropped-in API keys — a stopgap until a
proper gateway alternative is in place. See :mod:`amortized.core.model_catalog`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient
from amortized.core.model_catalog import enabled_models
from amortized.models import GatewayModel, ModelsResponse

logger = logging.getLogger("amortized.api.models")

router = APIRouter(prefix="/api/v1", tags=["models"])


@router.get(
    "/models",
    response_model=ModelsResponse,
    operation_id="list_models",
    summary="List available models (MLflow AI Gateway, or direct providers via dropped-in keys).",
)
async def list_models() -> ModelsResponse:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    gateway_models: list[GatewayModel] = []
    if tracking_uri:
        try:
            client = MLflowClient(tracking_uri)
            raw_models = await client.list_gateway_models()
            gateway_models = [GatewayModel(**m) for m in raw_models]
        except Exception:
            logger.warning("Failed to fetch gateway models from MLflow", exc_info=True)

    if gateway_models:
        return ModelsResponse(models=gateway_models, gateway_url=config_mod.settings.gateway_url)

    # Gateway unavailable or empty — fall back to direct providers (dropped-in keys).
    direct = [
        GatewayModel(name=model, provider=provider, model_name=model)
        for provider, model in enabled_models()
    ]
    return ModelsResponse(models=direct, gateway_url="")
