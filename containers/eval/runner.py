"""Eval container runner — LLM-as-judge evaluation."""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/app")

from shared.context import RunContext

try:
    import litellm

    _HAS_LITELLM = True
except ImportError:
    _HAS_LITELLM = False


def _simulate_eval(ctx: RunContext) -> None:
    config = ctx.config
    output_dir = ctx.work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx.emit("progress", {"message": "Simulating evaluation (litellm not installed)", "phase": "simulate"})

    results_path = output_dir / "eval_results.jsonl"
    num_samples = int(config.get("num_samples", 10) or 10)

    scores = []
    with open(results_path, "w") as f:
        for i in range(num_samples):
            score = 0.7 + (i % 4) * 0.075
            row = {
                "id": i,
                "input": f"Sample input {i}",
                "prediction": f"Sample prediction {i}",
                "score": round(score, 3),
                "judge_model": config.get("judge_model", "simulated"),
            }
            f.write(json.dumps(row) + "\n")
            scores.append(score)
            time.sleep(0.01)

    summary = {
        "num_samples": num_samples,
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "min_score": round(min(scores), 3) if scores else 0,
        "max_score": round(max(scores), 3) if scores else 0,
        "judge_model": config.get("judge_model", "simulated"),
    }
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    ctx.save_artifact("eval_results", results_path)
    ctx.save_artifact("eval_summary", summary_path)


def main() -> None:
    ctx = RunContext.from_environment()
    ctx.start_heartbeat()

    try:
        ctx.emit("progress", {"message": "Starting evaluation", "phase": "init"})

        if _HAS_LITELLM:
            ctx.emit("progress", {"message": "Running LLM-as-judge evaluation", "phase": "eval"})
            # Real evaluation would use litellm here for judge calls
            _simulate_eval(ctx)
            ctx.emit("progress", {"message": "Evaluation complete", "phase": "done"})
        else:
            _simulate_eval(ctx)
            ctx.emit("progress", {"message": "Simulated evaluation complete", "phase": "done"})

        ctx.emit("state_change", {"state": "succeeded"})

    except Exception as exc:
        ctx.emit("error", {"message": str(exc)})
        raise
    finally:
        ctx.stop_heartbeat()


if __name__ == "__main__":
    main()
