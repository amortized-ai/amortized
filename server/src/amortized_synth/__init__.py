"""Amortized Synth — purpose-built conversation synthesis engine."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

from amortized_synth.config import SynthConfig
from amortized_synth.engine import SynthEngine
from amortized_synth.inference import InferenceClient
from amortized_synth.pipelines import get_pipeline, list_pipelines_info
from amortized_synth.types import Conversation, SynthResult, SynthStats, Turn


async def synthesize(
    config: SynthConfig,
    seeds: list[dict[str, Any]],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> SynthResult:
    """Main entry point. Creates engine + pipeline from config, runs synthesis."""
    client = InferenceClient(config.model)

    pipeline_kwargs: dict[str, Any] = {}
    if config.pipeline_config.system_prompt:
        pipeline_kwargs["system_prompt"] = config.pipeline_config.system_prompt
    if config.pipeline_config.user_simulator_prompt:
        pipeline_kwargs["user_simulator_prompt"] = config.pipeline_config.user_simulator_prompt
    if config.pipeline_config.attributes:
        pipeline_kwargs["attributes"] = config.pipeline_config.attributes

    pipeline = get_pipeline(config.pipeline, **pipeline_kwargs)

    engine = SynthEngine(client, pipeline)
    return await engine.run(
        seeds,
        max_turns=config.max_turns,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
        checkpoint_interval=config.checkpoint_interval,
        on_progress=on_progress,
    )


def list_pipelines() -> list[dict[str, Any]]:
    """List available pipelines with descriptions and config schemas."""
    return list_pipelines_info()


__all__ = [
    "Conversation",
    "SynthConfig",
    "SynthEngine",
    "SynthResult",
    "SynthStats",
    "Turn",
    "list_pipelines",
    "synthesize",
]
