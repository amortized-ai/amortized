"""SDG job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Imports sdg_hub and runs
Flow.generate(). Falls back to a simulated run if sdg_hub is not installed.
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
        import io

        from datasets import Dataset
        from sdg_hub import Flow, FlowRegistry

        # discover_flows() prints a Rich table to stdout — suppress it
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            FlowRegistry.discover_flows()
        finally:
            sys.stdout = old_stdout

        flow_id = str(config["flow_id"])
        flow_path = FlowRegistry.get_flow_path(flow_id)
        flow = Flow.from_yaml(flow_path)
        flow.set_model_config(
            model=str(config["model"]),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        )

        dataset_path = config.get("dataset_path")
        if not dataset_path:
            raise ValueError(
                "An input dataset is required for SDG flows. "
                "Each flow expects specific columns in the dataset. "
                "Use the /api/v1/flows endpoint to see required columns for each flow."
            )
        dataset = Dataset.from_json(str(dataset_path))

        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        result = flow.generate(
            dataset,
            runtime_params=config.get("runtime_params") or {},
            checkpoint_dir=checkpoint_dir,
        )

        # Save result
        output_path = os.path.join(output_dir, "generated_data.jsonl")
        result.to_json(output_path)

    except ImportError:
        _simulate_sdg(config, output_dir)


def _simulate_sdg(config: dict[str, Any], output_dir: str) -> None:
    """Simulate SDG when sdg_hub is not installed."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Write progress metadata
    total_rows = 50
    metadata = {
        "flow_id": str(config.get("flow_id", "")),
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

    # Write simulated generated data
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
        print("Usage: python -m amortized_runtime.runners.sdg_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_sdg(config_data)
