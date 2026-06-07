"""Synthesis container runner — wraps asynth execution."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from shared.context import RunContext

try:
    from asynth import LiteLLMInferenceConfig, SynthesisConfig, synthesize
    from asynth.configs.params.synthesis_params import GeneralSynthesisParams

    _HAS_ASYNTH = True
except ImportError:
    _HAS_ASYNTH = False


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
                max_concurrency=config.get("max_concurrent", 16),
            )

            raw_strategy = config.get("strategy_params")
            if raw_strategy and isinstance(raw_strategy, dict):
                strategy_params = GeneralSynthesisParams(**raw_strategy)
            else:
                strategy_params = GeneralSynthesisParams()

            synth_config = SynthesisConfig(
                num_samples=config.get("num_samples", 100),
                output_path=str(output_path),
                inference_config=inference_config,
                strategy_params=strategy_params,
            )

            ctx.emit("progress", {"message": "Running synthesis", "phase": "generate"})

            results = synthesize(synth_config)

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
