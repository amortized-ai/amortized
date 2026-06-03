"""Synthesis container runner — wraps SDG Hub flow execution."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/app")

from shared.context import RunContext


def main() -> None:
    ctx = RunContext.from_environment()
    ctx.start_heartbeat()

    try:
        ctx.emit("progress", {"message": "Starting synthesis", "phase": "init"})

        from datasets import Dataset
        from sdg_hub import Flow, FlowRegistry

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
        ctx.emit("state_change", {"state": "succeeded"})

    except Exception as exc:
        ctx.emit("error", {"message": str(exc)})
        raise
    finally:
        ctx.stop_heartbeat()


if __name__ == "__main__":
    main()
