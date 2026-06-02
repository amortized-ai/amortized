"""SDG flow discovery endpoints."""

import io
import logging
import sys

from fastapi import APIRouter

from amortized_runtime.models import FlowInfo

logger = logging.getLogger("amortized_runtime.routers.flows")

router = APIRouter(prefix="/api/v1/flows", tags=["flows"])

# Tag-to-category mapping for SDG flows
_TAG_CATEGORY_MAP: dict[str, str] = {
    "knowledge": "knowledge_infusion",
    "qa": "knowledge_infusion",
    "summary": "knowledge_infusion",
    "evaluation": "evaluation",
    "eval": "evaluation",
    "rag": "evaluation",
    "agent": "agentic",
    "agentic": "agentic",
    "mcp": "agentic",
    "red_team": "red_team",
    "adversarial": "red_team",
    "text": "text_analysis",
    "classification": "text_analysis",
    "sentiment": "text_analysis",
    "code": "code_evaluation",
}


def _tags_to_category(tags: list[str]) -> str:
    """Map a list of tags to a single category string."""
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in _TAG_CATEGORY_MAP:
            return _TAG_CATEGORY_MAP[tag_lower]
    return "unknown"


def _discover_flows() -> list[FlowInfo]:
    """Discover available SDG flows.

    Uses sdg_hub FlowRegistry when available, falls back to empty list.
    """
    try:
        from sdg_hub import FlowRegistry

        # discover_flows() prints a Rich table to stdout — suppress it
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            FlowRegistry.discover_flows()
        finally:
            sys.stdout = old_stdout

        flows: list[FlowInfo] = []
        for flow_id, entry in FlowRegistry._entries.items():
            tags = getattr(entry, "tags", []) or []
            flows.append(
                FlowInfo(
                    id=flow_id,
                    name=getattr(entry, "name", flow_id),
                    description=getattr(entry, "description", ""),
                    category=_tags_to_category(tags),
                )
            )
        return flows
    except ImportError:
        logger.debug("sdg_hub not installed, no flows available")
        return []
    except Exception:
        logger.exception("Failed to discover SDG flows")
        return []


@router.get("", response_model=list[FlowInfo])
async def get_flows() -> list[FlowInfo]:
    """List available SDG flows."""
    return _discover_flows()
