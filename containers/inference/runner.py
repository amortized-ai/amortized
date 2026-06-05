"""Inference container runner — vLLM batch inference."""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/app")

from shared.context import RunContext

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _simulate_inference(ctx: RunContext) -> None:
    config = ctx.config
    output_dir = ctx.work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    ctx.emit("progress", {"message": "Simulating inference (vllm not installed)", "phase": "simulate"})

    prompts = config.get("prompts", ["Hello, world!"])
    with open(results_path, "w") as f:
        for i, prompt in enumerate(prompts):
            row = {
                "id": i,
                "prompt": prompt,
                "output": f"Simulated response to: {prompt}",
                "model": config.get("model_path", "simulated"),
            }
            f.write(json.dumps(row) + "\n")
            time.sleep(0.01)

    ctx.save_artifact("results", results_path)


def main() -> None:
    ctx = RunContext.from_environment()
    ctx.start_heartbeat()

    try:
        ctx.emit("progress", {"message": "Starting inference", "phase": "init"})

        if _HAS_VLLM:
            config = ctx.config
            output_dir = ctx.work_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            llm = LLM(model=config["model_path"])
            sampling_params = SamplingParams(
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 256),
            )

            prompts = config.get("prompts", [])
            outputs = llm.generate(prompts, sampling_params)

            results_path = output_dir / "results.jsonl"
            with open(results_path, "w") as f:
                for i, output in enumerate(outputs):
                    row = {
                        "id": i,
                        "prompt": output.prompt,
                        "output": output.outputs[0].text,
                        "model": config["model_path"],
                    }
                    f.write(json.dumps(row) + "\n")

            ctx.save_artifact("results", results_path)
            ctx.emit("progress", {"message": "Inference complete", "phase": "done"})
        else:
            _simulate_inference(ctx)
            ctx.emit("progress", {"message": "Simulated inference complete", "phase": "done"})

        ctx.emit("state_change", {"state": "succeeded"})

    except Exception as exc:
        ctx.emit("error", {"message": str(exc)})
        raise
    finally:
        ctx.stop_heartbeat()


if __name__ == "__main__":
    main()
