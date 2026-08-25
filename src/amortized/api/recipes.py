"""Starter template browsing from agents/*/skills reference payloads."""

import logging
from typing import Any

from fastapi import APIRouter

from amortized.core.recipes import list_starter_templates

logger = logging.getLogger("amortized.api.recipes")

router = APIRouter(prefix="/api/v1/recipes", tags=["recipes"])


@router.get(
    "/starter-templates",
    operation_id="list_starter_templates",
    summary="List curated starter templates",
    description=(
        "Returns researcher-tested reference configs for SDG and training jobs. "
        "Each template includes a complete, ready-to-submit config that can be "
        "adapted to the user's domain. Use these as starting points instead of "
        "building configs from scratch."
    ),
)
async def get_starter_templates() -> list[dict[str, Any]]:
    return list_starter_templates()
