"""SDG flow discovery endpoints."""

import logging

from fastapi import APIRouter

from amortized_runtime.models import FlowInfo

logger = logging.getLogger("amortized_runtime.routers.flows")

router = APIRouter(prefix="/api/v1/flows", tags=["flows"])

# Mock flow data — replaced by FlowRegistry.discover_flows() when sdg_hub is installed
_MOCK_FLOWS: list[FlowInfo] = [
    FlowInfo(
        id="knowledge-qa",
        name="Knowledge Q&A Generation",
        description="Generate question-answer pairs from knowledge documents",
        category="knowledge_infusion",
    ),
    FlowInfo(
        id="rag-eval",
        name="RAG Evaluation Dataset",
        description="Generate evaluation datasets for RAG pipelines",
        category="evaluation",
    ),
    FlowInfo(
        id="mcp-distillation",
        name="MCP Agent Distillation",
        description="Distill MCP agent tool-use behavior into training data",
        category="agentic",
    ),
    FlowInfo(
        id="text-summarization",
        name="Text Summarization",
        description="Generate extractive and abstractive summaries from documents",
        category="text_analysis",
    ),
]


def _discover_flows() -> list[FlowInfo]:
    """Discover available SDG flows.

    Uses sdg_hub FlowRegistry when available, falls back to mock data.
    """
    try:
        from sdg_hub import FlowRegistry

        FlowRegistry.discover_flows()
        flows: list[FlowInfo] = []
        for flow_id in FlowRegistry.list_flows():
            info = FlowRegistry.get_flow_info(flow_id)
            flows.append(
                FlowInfo(
                    id=flow_id,
                    name=info.get("name", flow_id),
                    description=info.get("description", ""),
                    category=info.get("category", "unknown"),
                )
            )
        return flows
    except ImportError:
        logger.debug("sdg_hub not installed, using mock flow data")
        return _MOCK_FLOWS


@router.get("", response_model=list[FlowInfo])
async def get_flows() -> list[FlowInfo]:
    """List available SDG flows."""
    return _discover_flows()
