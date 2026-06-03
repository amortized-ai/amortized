"""Pipeline registry for Amortized Synth."""

from __future__ import annotations

from typing import Any

from amortized_synth.pipelines.attribute import AttributePipeline
from amortized_synth.pipelines.base import BasePipeline
from amortized_synth.pipelines.conversation import ConversationPipeline
from amortized_synth.pipelines.transform import TransformPipeline

_REGISTRY: dict[str, type[BasePipeline]] = {
    "conversation": ConversationPipeline,
    "attribute": AttributePipeline,
    "transform": TransformPipeline,
}


def get_pipeline(name: str, **kwargs: Any) -> BasePipeline:
    """Get a pipeline instance by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown pipeline '{name}'. Available: {available}")
    return cls(**kwargs)


def list_pipelines_info() -> list[dict[str, Any]]:
    """List available pipelines with metadata."""
    result: list[dict[str, Any]] = []
    for name, cls in _REGISTRY.items():
        result.append(
            {
                "name": name,
                "description": cls.description,
                "supports_multi_turn": cls.supports_multi_turn,
                "config_schema": cls.config_schema(),
            }
        )
    return result


__all__ = [
    "AttributePipeline",
    "BasePipeline",
    "ConversationPipeline",
    "TransformPipeline",
    "get_pipeline",
    "list_pipelines_info",
]
