"""Configuration for synthesis jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amortized_synth.inference import ModelConfig


@dataclass
class PipelineConfig:
    """Pipeline-specific configuration."""

    system_prompt: str | None = None
    user_simulator_prompt: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    output_format: str = "messages"


@dataclass
class SynthConfig:
    """Top-level synthesis configuration."""

    pipeline: str
    model: ModelConfig
    num_samples: int = 100
    max_turns: int = 5
    checkpoint_interval: int = 50
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynthConfig:
        """Create from a flat config dictionary (e.g., from job config JSON)."""
        model_config = ModelConfig(
            model=data["model"],
            api_base=data.get("api_base"),
            api_key=data.get("api_key"),
            max_concurrent=data.get("max_concurrent", 16),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2048),
        )
        pipeline_config = PipelineConfig(
            system_prompt=data.get("system_prompt"),
            user_simulator_prompt=data.get("user_simulator_prompt"),
            attributes=data.get("attributes") or {},
            output_format=data.get("output_format", "messages"),
        )
        return cls(
            pipeline=data["pipeline"],
            model=model_config,
            num_samples=data.get("num_samples", 100),
            max_turns=data.get("max_turns", 5),
            checkpoint_interval=data.get("checkpoint_interval", 50),
            pipeline_config=pipeline_config,
        )
