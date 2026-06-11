"""Synthesis pipeline discovery endpoints."""

import logging

from fastapi import APIRouter

from amortized.models import PipelineInfo, SynthCapabilities

logger = logging.getLogger("amortized.api.flows")

router = APIRouter(prefix="/api/v1/flows", tags=["flows"])


def _discover_pipelines() -> list[PipelineInfo]:
    """Discover available synthesis strategies via asynth."""
    try:
        import asynth  # noqa: F401

        return [
            PipelineInfo(
                name="general",
                description="General synthesis strategy with attribute system",
                supports_multi_turn=True,
                config_schema={"strategy_params": "GeneralSynthesisParams"},
            )
        ]
    except ImportError:
        logger.debug("asynth not installed, no strategies available")
        return []


@router.get("", response_model=list[PipelineInfo])
async def get_pipelines() -> list[PipelineInfo]:
    """List available synthesis pipelines."""
    return _discover_pipelines()


@router.get("/capabilities", response_model=SynthCapabilities)
async def get_capabilities() -> SynthCapabilities:
    """Report asynth capabilities and available features."""
    try:
        import asynth

        return SynthCapabilities(
            available=True,
            version=getattr(asynth, "__version__", "unknown"),
            strategies=["general"],
            attribute_types=["sampled", "generated", "multiturn", "transformed"],
            data_sources=["dataset", "document", "example"],
            environment_types=["synthetic", "deterministic"],
            judge_templates=_list_judge_templates(),
        )
    except ImportError:
        return SynthCapabilities(available=False)


def _list_judge_templates() -> list[str]:
    from amortized.core.judge_templates import list_judge_templates

    return list_judge_templates()
