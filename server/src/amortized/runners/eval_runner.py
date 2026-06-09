"""Eval job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Uses asynth judges for
LLM-based and rule-based evaluation. Falls back to a simulated run if
asynth is not installed.
"""

import json
import os
import sys
import time
from typing import Any

try:
    from asynth import JudgeConfig, LiteLLMInferenceConfig, create_judge

    _has_asynth = True
except ImportError:
    _has_asynth = False


def run_eval(config: dict[str, Any]) -> None:
    """Execute an evaluation job using asynth judges."""
    output_dir = str(config.get("output_dir", "./eval_output"))
    os.makedirs(output_dir, exist_ok=True)

    if _has_asynth:
        _run_asynth_eval(config, output_dir)
    else:
        _simulate_eval(config, output_dir)


def _load_dataset(path: str) -> list[dict[str, Any]]:
    """Read JSONL lines from a file path."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_asynth_eval(config: dict[str, Any], output_dir: str) -> None:
    """Run evaluation via asynth's judge system."""
    dataset_path = config.get("dataset_path") or config.get("dataset")
    if not dataset_path:
        raise ValueError("dataset or dataset_path is required for eval jobs")

    evaluator_type = config.get("evaluator_type", "llm")
    judgment_type = config.get("judgment_type", "bool")
    response_format = config.get("response_format", "json")

    if evaluator_type == "rule_based":
        judge_config = JudgeConfig.from_dict(
            {
                "rule_judge_params": {
                    "rule_type": config.get("rule_config", {}).get("rule_type", "regex"),
                    "input_fields": config.get("variables", []),
                    "rule_config": config.get("rule_config", {}),
                    "response_format": response_format,
                    "judgment_type": judgment_type,
                },
            }
        )
    else:
        judge_params: dict[str, Any] = {
            "prompt_template": config.get("judge_prompt", "Evaluate the following: {response}"),
            "response_format": response_format,
            "judgment_type": judgment_type,
        }
        if config.get("variables"):
            judge_params["prompt_template_placeholders"] = config["variables"]
        judge_config = JudgeConfig.from_dict({"judge_params": judge_params})

    inference_config = None
    if evaluator_type == "llm":
        model = config.get("judge_model") or config.get("model", "openai/gpt-4o-mini")
        inf_params = config.get("inference_params", {})
        inference_config = LiteLLMInferenceConfig(
            model=model,
            temperature=inf_params.get("temperature", 1.0),
            max_tokens=inf_params.get("max_tokens"),
            top_p=inf_params.get("top_p"),
            seed=inf_params.get("seed"),
            api_base=inf_params.get("api_base"),
            api_key=inf_params.get("api_key"),
        )

    judge = create_judge(judge_config, inference_config=inference_config)

    data = _load_dataset(dataset_path)
    outputs = judge.judge(data)

    results: list[dict[str, Any]] = []
    for i, output in enumerate(outputs):
        entry: dict[str, Any] = {
            "index": i,
            "passed": output.passed,
            "score": output.score,
            "explanation": output.explanation,
            "raw_output": output.raw_output,
        }
        results.append(entry)

    _write_eval_results(output_dir, results)


def _simulate_eval(config: dict[str, Any], output_dir: str) -> None:
    """Simulate evaluation when asynth is not installed."""
    max_samples = int(config.get("max_samples", 10) or 10)

    results: list[dict[str, Any]] = []
    for i in range(max_samples):
        passed = i % 3 != 0
        results.append(
            {
                "index": i,
                "passed": passed,
                "score": 0.8 if passed else 0.3,
                "explanation": f"Simulated judgment for sample {i}",
                "raw_output": "",
            }
        )
        time.sleep(0.01)

    _write_eval_results(output_dir, results)


def _write_eval_results(output_dir: str, results: list[dict[str, Any]]) -> None:
    """Write per-sample results and aggregate summary."""
    results_path = os.path.join(output_dir, "eval_results.json")
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    failed = total - passed
    scores: list[float] = [float(r["score"]) for r in results if r.get("score") is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    output = {
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "average_score": round(avg_score, 4),
        },
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m amortized.runners.eval_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_eval(config_data)
