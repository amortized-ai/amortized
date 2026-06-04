"""Inference job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Imports vllm and runs
batch inference. Falls back to a simulated run if vllm is not installed.
"""

import json
import os
import sys
import time
from typing import Any


def run_inference(config: dict[str, Any]) -> None:
    """Execute a vLLM batch inference job."""
    output_path = str(config["output_path"])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
    except ImportError:
        LLM = None  # noqa: N806
        SamplingParams = None  # noqa: N806

    if LLM is not None and SamplingParams is not None:
        llm = LLM(model=config["model_path"])
        params = SamplingParams(
            temperature=float(config.get("temperature", 0.0) or 0.0),
            max_tokens=int(config.get("max_tokens", 2048) or 2048),
        )

        input_path = config.get("input_path")
        if input_path:
            with open(input_path) as f:
                prompts = [json.loads(line)["prompt"] for line in f]
        else:
            prompts = config.get("prompts", [])

        outputs = llm.generate(prompts, params)

        with open(output_path, "w") as f:
            for output in outputs:
                f.write(json.dumps({
                    "prompt": output.prompt,
                    "output": output.outputs[0].text,
                }) + "\n")
    else:
        _simulate_inference(config, output_path)


def _simulate_inference(config: dict[str, Any], output_path: str) -> None:
    """Simulate inference when vllm is not installed."""
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
