"""Inference job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Uses simulation mode
until container-based vLLM inference is deployed.
"""

import json
import os
import sys
import time
from typing import Any


def run_inference(config: dict[str, Any]) -> None:
    """Execute a vLLM batch inference job (simulation mode)."""
    output_path = str(config["output_path"])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    batch_size = int(config.get("batch_size", 32) or 32)
    total_samples = batch_size * 3

    results = []
    for i in range(total_samples):
        results.append({
            "index": i,
            "input": f"sample_{i}",
            "output": f"Generated response for sample {i}",
            "model": str(config["model_path"]),
            "tokens_generated": int(config.get("max_tokens", 2048) or 2048),
        })
        time.sleep(0.01)

    with open(output_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m amortized.runners.inference_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_inference(config_data)
