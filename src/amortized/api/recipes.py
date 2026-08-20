"""Starter template browsing from agent/skills reference payloads."""

import logging
from typing import Any

from fastapi import APIRouter

from amortized.core.recipes import list_starter_templates

logger = logging.getLogger("amortized.api.recipes")

router = APIRouter(prefix="/api/v1/recipes", tags=["recipes"])


@router.get("/starter-templates", operation_id="list_starter_templates")
async def get_starter_templates() -> list[dict[str, Any]]:
    return list_starter_templates()
