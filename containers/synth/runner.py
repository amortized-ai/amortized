"""Synthesis container runner — wraps asynth execution."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "/app")

from shared.context import RunContext

logger = logging.getLogger("synth.runner")

try:
    from asynth import LiteLLMInferenceConfig, SynthesisConfig, synthesize
    from asynth.configs.params.synthesis_params import GeneralSynthesisParams

    _HAS_ASYNTH = True
except ImportError:
    _HAS_ASYNTH = False


def _build_strategy_params(cls: type, params: dict[str, Any]) -> Any:
    """Build GeneralSynthesisParams, preferring from_dict() when available."""
    if hasattr(cls, "from_dict"):
        return cls.from_dict(params)
    return cls(**params)


def _check_output_quality(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Warn if output looks like template echo from LLM auth failures."""
    if not results or not config.get("strategy_params", {}).get("generated_attributes"):
        return
    for attr in config["strategy_params"]["generated_attributes"]:
        attr_id = attr.get("id", "")
        if not attr_id:
            continue
        sample_values = [r.get(attr_id, "") for r in results[:5] if attr_id in r]
        if len(set(sample_values)) == 1 and len(sample_values) > 1:
            logger.warning(
                "All %d samples for '%s' are identical — LLM may not have run",
                len(sample_values),
                attr_id,
            )


def _simulate_synth(ctx: RunContext) -> None:
    config = ctx.config
    num_samples = int(config.get("num_samples", 100) or 100)
    output_dir = ctx.work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generated.jsonl"

    ctx.emit("progress", {"message": "Simulating synthesis (asynth not installed)", "phase": "simulate"})

    with open(output_path, "w") as f:
        for i in range(num_samples):
            row = {
                "id": i,
                "input": f"Simulated input {i}",
                "output": f"Simulated output {i}",
                "model": config.get("model", "simulated"),
            }
            f.write(json.dumps(row) + "\n")
            time.sleep(0.005)

    ctx.save_artifact("dataset", output_path)


def main() -> None:
    ctx = RunContext.from_environment()
    ctx.start_heartbeat()

    try:
        ctx.emit("progress", {"message": "Starting synthesis", "phase": "init"})

        if _HAS_ASYNTH:
            config = ctx.config
            output_dir = ctx.work_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "generated_data.jsonl"

            inference_config = LiteLLMInferenceConfig(
                model=config["model"],
                api_base=config.get("api_base"),
                api_key=config.get("api_key"),
                temperature=config.get("temperature", 0.7),
                max_concurrency=config.get("max_concurrency", 16),
                max_tokens=config.get("max_tokens"),
                top_p=config.get("top_p"),
                seed=config.get("seed"),
                num_retries=config.get("num_retries", 3),
            )

            raw_strategy = config.get("strategy_params")
            if raw_strategy and isinstance(raw_strategy, dict):
                merged = dict(raw_strategy)
                if config.get("input_data") and "input_data" not in merged:
                    merged["input_data"] = config["input_data"]
                if config.get("input_documents") and "input_documents" not in merged:
                    merged["input_documents"] = config["input_documents"]
                strategy_params = _build_strategy_params(GeneralSynthesisParams, merged)
            else:
                kwargs: dict[str, Any] = {}
                if config.get("input_data"):
                    kwargs["input_data"] = config["input_data"]
                if config.get("input_documents"):
                    kwargs["input_documents"] = config["input_documents"]
                strategy_params = _build_strategy_params(GeneralSynthesisParams, kwargs)

            synth_config = SynthesisConfig(
                num_samples=config.get("num_samples", 100),
                output_path=str(output_path),
                inference_config=inference_config,
                strategy_params=strategy_params,
            )

            ctx.emit("progress", {"message": "Running synthesis", "phase": "generate"})

            results = synthesize(synth_config)

            _check_output_quality(results, config)

            ctx.save_artifact("dataset", output_path)
            ctx.emit("progress", {"message": "Synthesis complete", "phase": "done"})
        else:
            _simulate_synth(ctx)
            ctx.emit("progress", {"message": "Simulated synthesis complete", "phase": "done"})

        ctx.emit("state_change", {"state": "succeeded"})

    except Exception as exc:
        ctx.emit("error", {"message": str(exc)})
        raise
    finally:
        ctx.stop_heartbeat()


if __name__ == "__main__":
    main()
