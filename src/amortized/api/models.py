"""Gateway model discovery endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient
from amortized.models import GatewayModel, ModelsResponse

logger = logging.getLogger("amortized.api.models")

router = APIRouter(prefix="/api/v1", tags=["models"])


async def _fetch_gateway_models() -> list[GatewayModel]:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri:
        return []

    try:
        client = MLflowClient(tracking_uri)
        endpoints = await client.list_gateway_endpoints()
    except Exception:
        logger.warning("Failed to fetch gateway endpoints from MLflow", exc_info=True)
        return []

    models: list[GatewayModel] = []
    for endpoint in endpoints:
        provider = ""
        model_name = ""
        for mapping in endpoint.get("model_mappings", []):
            model_def = mapping.get("model_definition", {})
            if model_def:
                provider = model_def.get("provider", "")
                model_name = model_def.get("model_name", "")
                break
        models.append(
            GatewayModel(
                name=endpoint.get("name", ""),
                provider=provider,
                model_name=model_name,
            )
        )
    return models


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
        gateway_url=config_mod.settings.gateway_url,
    )
