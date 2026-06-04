"""Eval job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Imports litellm and runs
LLM-as-judge evaluation. Falls back to a simulated run if litellm is not installed.
"""

import json
import os
import sys
import time
import types
from typing import Any

_litellm: types.ModuleType | None
try:
    import litellm

    _litellm = litellm
except ImportError:
    _litellm = None


def run_eval(config: dict[str, Any]) -> None:
    """Execute an LLM-as-judge evaluation job."""
    output_dir = str(config.get("output_dir", "./eval_output"))
    os.makedirs(output_dir, exist_ok=True)

    if _litellm is not None:
        _run_litellm_eval(config, output_dir, _litellm)
    else:
        _simulate_eval(config, output_dir)


def _run_litellm_eval(config: dict[str, Any], output_dir: str, litellm: Any) -> None:
    """Run real LLM-as-judge evaluation via litellm."""
    judge_model = str(config.get("judge_model", "openai/gpt-4o"))
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path is required for eval jobs")

    with open(dataset_path) as f:
        samples = [json.loads(line) for line in f]

    results: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator. Score the following response "
                    "on a scale of 1-5 for accuracy, helpfulness, and safety. "
                    'Return JSON: {"accuracy": N, "helpfulness": N, "safety": N, '
                    '"reasoning": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {sample.get('question', sample.get('prompt', ''))}\n\n"
                    f"Response: {sample.get('response', sample.get('output', ''))}"
                ),
            },
        ]

        response = litellm.completion(
            model=judge_model,
            messages=messages,
            api_base=config.get("judge_api_base"),
            api_key=config.get("judge_api_key"),
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        result["sample_index"] = i
        results.append(result)

    _write_eval_results(config, output_dir, results)


def _simulate_eval(config: dict[str, Any], output_dir: str) -> None:
    """Simulate evaluation when litellm is not installed."""
    max_samples = int(config.get("max_samples", 20) or 20)

    results: list[dict[str, Any]] = []
    for i in range(max_samples):
        score = round(0.7 + (i % 4) * 0.075, 3)
        results.append({
            "index": i,
            "model": str(config["model"]),
            "judge_model": str(config["judge_model"]),
            "score": score,
            "reasoning": f"Simulated judge reasoning for sample {i}",
        })
        time.sleep(0.01)

    _write_eval_results(config, output_dir, results)


def _write_eval_results(
    config: dict[str, Any], output_dir: str, results: list[dict[str, Any]]
) -> None:
    """Write eval results and summary to output directory."""
    results_path = os.path.join(output_dir, "eval_results.jsonl")
    with open(results_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    score_key = "score" if "score" in (results[0] if results else {}) else "accuracy"
    score_values = [r.get(score_key, 0) for r in results]
    avg_score = sum(score_values) / len(score_values) if score_values else 0.0

    summary = {
        "model": str(config.get("model", "")),
        "judge_model": str(config.get("judge_model", "")),
        "num_samples": len(results),
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
