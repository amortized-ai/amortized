"""Training container runner — wraps Training Hub's lora_sft."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from shared.context import RunContext

try:
    from training_hub import lora_sft
except ImportError:
    lora_sft = None  # type: ignore[assignment]


def _simulate_training(ctx: RunContext) -> None:
    config = ctx.config
    output_dir = config.get("ckpt_output_dir", str(ctx.work_dir / "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = int(config.get("num_epochs", 3) or 3)
    max_steps = num_epochs * 100
    metrics_path = os.path.join(output_dir, "training_metrics.jsonl")

    ctx.emit("progress", {"message": "Simulating training (training_hub not installed)", "phase": "simulate"})

    with open(metrics_path, "w") as f:
        for step in range(1, max_steps + 1):
            loss = 3.5 * (0.95**step)
            lr = float(config.get("learning_rate", 2e-4) or 2e-4)
            epoch = step / 100.0
            metric = {
                "step": step,
                "loss": round(loss, 4),
                "epoch": round(epoch, 2),
                "learning_rate": lr,
                "max_steps": max_steps,
            }
            f.write(json.dumps(metric) + "\n")
            f.flush()
            time.sleep(0.01)

    adapter_config = {
        "peft_type": "LORA",
        "r": int(config.get("lora_r", 16) or 16),
        "lora_alpha": int(config.get("lora_alpha", 32) or 32),
        "base_model_name_or_path": str(config.get("model_path", "")),
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    with open(os.path.join(output_dir, "adapter_model.safetensors"), "wb") as f:
        f.write(b"\x00" * 256)

    ctx.save_artifact("model", Path(output_dir))


def main() -> None:
    ctx = RunContext.from_environment()
    ctx.start_heartbeat()

    try:
        ctx.emit("progress", {"message": "Starting training", "phase": "init"})

        config = ctx.config
        output_dir = config.get("ckpt_output_dir", str(ctx.work_dir / "outputs"))

        if lora_sft is not None:
            result = lora_sft(
                model_path=config["model_path"],
                data_path=config["data_path"],
                ckpt_output_dir=output_dir,
                learning_rate=config.get("learning_rate", 2e-4),
                num_epochs=config.get("num_epochs", 3),
                micro_batch_size=config.get("micro_batch_size", 2),
                max_seq_len=config.get("max_seq_len", 2048),
                lora_r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                load_in_4bit=config.get("load_in_4bit", False),
            )
            ctx.emit("progress", {"message": "Training complete", "phase": "done"})
            ctx.save_artifact("model", Path(output_dir))
        else:
            _simulate_training(ctx)
            ctx.emit("progress", {"message": "Simulated training complete", "phase": "done"})

        ctx.emit("state_change", {"state": "succeeded"})

    except Exception as exc:
        ctx.emit("error", {"message": str(exc)})
        raise
    finally:
        ctx.stop_heartbeat()


if __name__ == "__main__":
    main()
