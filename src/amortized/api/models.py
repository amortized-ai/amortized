"""Gateway model discovery endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient
from amortized.models import GatewayModel, ModelsResponse

logger = logging.getLogger("amortized.api.models")

router = APIRouter(prefix="/api/v1", tags=["models"])


@router.get(
    "/models",
    response_model=ModelsResponse,
    operation_id="list_models",
    summary="List available models from the MLflow AI Gateway.",
)
async def list_models() -> ModelsResponse:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri:
        return ModelsResponse(models=[], gateway_url=config_mod.settings.gateway_url)

    try:
        client = MLflowClient(tracking_uri)
        raw_models = await client.list_gateway_models()
    except Exception:
        logger.warning("Failed to fetch gateway models from MLflow", exc_info=True)
        return ModelsResponse(models=[], gateway_url=config_mod.settings.gateway_url)

    models = [GatewayModel(**m) for m in raw_models]
    return ModelsResponse(
        models=models,
        gateway_url=config_mod.settings.gateway_url,
    )
