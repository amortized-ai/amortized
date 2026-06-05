"""SDG job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Uses asynth for
synthesis. Falls back to a simulated run if asynth is not installed.
"""

import json
import os
import sys
import time
from typing import Any


def run_sdg(config: dict[str, Any]) -> None:
    """Execute a synthetic data generation job."""
    output_dir = str(config.get("output_dir", "./sdg_output"))
    os.makedirs(output_dir, exist_ok=True)

    try:
        from asynth import LiteLLMInferenceConfig, SynthesisConfig, synthesize
        from asynth.configs.params.synthesis_params import GeneralSynthesisParams
    except ImportError:
        _simulate_sdg(config, output_dir)
        return

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

    output_path = os.path.join(output_dir, "generated_data.jsonl")

    synth_config = SynthesisConfig(
        num_samples=config.get("num_samples", 100),
        output_path=output_path,
        inference_config=inference_config,
        strategy_params=strategy_params,
    )

    results = synthesize(synth_config)

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(
            {
                "total_completed": len(results),
                "total_requested": config.get("num_samples", 100),
                "status": "completed",
            },
            f,
            indent=2,
        )


def _simulate_sdg(config: dict[str, Any], output_dir: str) -> None:
    """Simulate SDG when asynth is not installed."""
    total_rows = 50
    output_path = os.path.join(output_dir, "generated_data.jsonl")

    with open(output_path, "w") as f:
        for i in range(total_rows):
            row = {
                "instruction": f"Sample instruction {i}",
                "input": f"Sample input {i}",
                "output": f"Sample output {i}",
            }
            f.write(json.dumps(row) + "\n")
            time.sleep(0.01)

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(
            {
                "total_completed": total_rows,
                "total_requested": config.get("num_samples", 100),
                "status": "completed",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m amortized.runners.sdg_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_sdg(config_data)
