"""Synthesis container runner — wraps SDG Hub flow execution."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from shared.context import RunContext

try:
    from datasets import Dataset
    from sdg_hub import Flow, FlowRegistry

    _HAS_SDG = True
except ImportError:
    _HAS_SDG = False


def _simulate_synth(ctx: RunContext) -> None:
    config = ctx.config
    num_samples = int(config.get("num_samples", 100) or 100)
    output_dir = ctx.work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generated.jsonl"

    ctx.emit("progress", {"message": "Simulating synthesis (sdg_hub not installed)", "phase": "simulate"})

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

        if _HAS_SDG:
            config = ctx.config

            FlowRegistry.discover_flows()
            flow_path = FlowRegistry.get_flow_path(config["flow_id"])
            flow = Flow.from_yaml(flow_path)

            flow.set_model_config(
                model=config["model"],
                api_base=config.get("api_base", "http://localhost:8101/v1"),
                api_key=config.get("api_key", ""),
            )

            dataset = Dataset.from_json(config["dataset_path"])

            checkpoint_dir = str(ctx.work_dir / "checkpoints")
            log_dir = str(ctx.work_dir / "logs")
            output_dir = ctx.work_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            ctx.emit("progress", {"message": "Running flow", "phase": "generate"})

            result = flow.generate(
                dataset,
                runtime_params=config.get("runtime_params", {}),
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
            )

            output_path = output_dir / "generated.jsonl"
            result.to_json(str(output_path))

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
