"""Training container runner — wraps Training Hub's lora_sft."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/app")

from shared.context import RunContext


def main() -> None:
    ctx = RunContext.from_environment()
    ctx.start_heartbeat()

    try:
        ctx.emit("progress", {"message": "Starting training", "phase": "init"})

        from training_hub import lora_sft

        config = ctx.config
        output_dir = config.get("ckpt_output_dir", str(ctx.work_dir / "outputs"))

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

        ctx.emit("state_change", {"state": "succeeded"})

    except Exception as exc:
        ctx.emit("error", {"message": str(exc)})
        raise
    finally:
        ctx.stop_heartbeat()


if __name__ == "__main__":
    main()
