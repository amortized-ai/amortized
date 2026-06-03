"""Synthesis pipeline discovery endpoints."""

import logging
from typing import Any

from fastapi import APIRouter

from amortized.models import PipelineInfo

logger = logging.getLogger("amortized.api.flows")

router = APIRouter(prefix="/api/v1/flows", tags=["flows"])


def _discover_pipelines() -> list[PipelineInfo]:
    """Discover available synthesis pipelines."""
    try:
        from amortized_synth import list_pipelines

        pipelines: list[dict[str, Any]] = list_pipelines()
        return [PipelineInfo(**p) for p in pipelines]
    except ImportError:
        logger.debug("amortized_synth not installed, no pipelines available")
        return []
    except Exception:
        logger.exception("Failed to discover synthesis pipelines")
        return []


@router.get("", response_model=list[PipelineInfo])
async def get_pipelines() -> list[PipelineInfo]:
    """List available synthesis pipelines."""
    return _discover_pipelines()
