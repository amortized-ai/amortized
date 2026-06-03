"""SDG job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Uses amortized_synth for
conversation synthesis. Falls back to a simulated run if amortized_synth
or litellm is not installed.
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
        import asyncio

        from amortized_synth import synthesize
        from amortized_synth.config import SynthConfig

        synth_config = SynthConfig.from_dict(config)
        seeds = _load_seeds(config)

        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        def on_progress(completed: int, total: int) -> None:
            progress = {"completed": completed, "total": total, "status": "running"}
            with open(os.path.join(checkpoint_dir, "progress.json"), "w") as f:
                json.dump(progress, f)

        from amortized_synth.pipelines import get_pipeline

        pipeline_kwargs: dict[str, Any] = {}
        if synth_config.pipeline_config.system_prompt:
            pipeline_kwargs["system_prompt"] = synth_config.pipeline_config.system_prompt
        if synth_config.pipeline_config.user_simulator_prompt:
            pipeline_kwargs["user_simulator_prompt"] = (
                synth_config.pipeline_config.user_simulator_prompt
            )
        if synth_config.pipeline_config.attributes:
            pipeline_kwargs["attributes"] = synth_config.pipeline_config.attributes

        pipeline = get_pipeline(synth_config.pipeline, **pipeline_kwargs)

        result = asyncio.run(
            synthesize(
                synth_config,
                seeds,
                on_progress=on_progress,
                checkpoint_dir=checkpoint_dir,
            )
        )

        output_path = os.path.join(output_dir, "generated_data.jsonl")
        with open(output_path, "w") as f:
            for conv in result.conversations:
                f.write(json.dumps(pipeline.format_output(conv)) + "\n")

        with open(os.path.join(output_dir, "stats.json"), "w") as f:
            json.dump(
                {
                    "total_requested": result.stats.total_requested,
                    "total_completed": result.stats.total_completed,
                    "total_failed": result.stats.total_failed,
                    "total_tokens_used": result.stats.total_tokens_used,
                    "total_turns_generated": result.stats.total_turns_generated,
                    "elapsed_seconds": result.stats.elapsed_seconds,
                },
                f,
                indent=2,
            )

    except ImportError:
        _simulate_sdg(config, output_dir)


def _load_seeds(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load seed data from file or generate default seeds."""
    seed_path = config.get("seed_data_path")
    if seed_path and os.path.exists(seed_path):
        seeds: list[dict[str, Any]] = []
        with open(seed_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(json.loads(line))
        return seeds

    num_samples = config.get("num_samples", 100)
    return [{"topic": f"topic_{i}", "persona": "a curious user"} for i in range(num_samples)]


def _simulate_sdg(config: dict[str, Any], output_dir: str) -> None:
    """Simulate SDG when amortized_synth is not installed."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_rows = 50
    metadata = {
        "pipeline": str(config.get("pipeline", "conversation")),
        "total_rows": total_rows,
        "completed_rows": 0,
        "status": "running",
    }
    metadata_path = os.path.join(checkpoint_dir, "flow_metadata.json")

    for i in range(1, total_rows + 1):
        metadata["completed_rows"] = i
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        time.sleep(0.01)

    metadata["status"] = "completed"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    output_path = os.path.join(output_dir, "generated_data.jsonl")
    with open(output_path, "w") as f:
        for i in range(total_rows):
            row = {
                "instruction": f"Sample instruction {i}",
                "input": f"Sample input {i}",
                "output": f"Sample output {i}",
            }
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m amortized.runners.sdg_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_sdg(config_data)
