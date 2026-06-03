"""Eval job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Uses simulation mode
until container-based LLM-as-judge evaluation is deployed.
"""

import json
import os
import sys
import time
from typing import Any


def run_eval(config: dict[str, Any]) -> None:
    """Execute an LLM-as-judge evaluation job (simulation mode)."""
    output_dir = str(config.get("output_dir", "./eval_output"))
    os.makedirs(output_dir, exist_ok=True)

    max_samples = int(config.get("max_samples", 20) or 20)

    scores: list[dict[str, Any]] = []
    score_values: list[float] = []
    for i in range(max_samples):
        score = round(0.7 + (i % 4) * 0.075, 3)
        scores.append({
            "index": i,
            "model": str(config["model"]),
            "judge_model": str(config["judge_model"]),
            "score": score,
            "reasoning": f"Simulated judge reasoning for sample {i}",
        })
        score_values.append(score)
        time.sleep(0.01)

    results_path = os.path.join(output_dir, "eval_results.jsonl")
    with open(results_path, "w") as f:
        for row in scores:
            f.write(json.dumps(row) + "\n")

    avg_score = sum(score_values) / len(score_values) if score_values else 0.0
    summary = {
        "model": str(config["model"]),
        "judge_model": str(config["judge_model"]),
        "num_samples": len(scores),
        "average_score": round(avg_score, 3),
    }
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m amortized.runners.eval_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_eval(config_data)
