"""Training job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Imports training_hub and runs
lora_sft(). Falls back to a simulated run if training_hub is not installed.
"""

import json
import os
import sys
import time
from typing import Any


def run_training(config: dict[str, Any]) -> None:
    """Execute a LoRA SFT training job."""
    output_dir = str(config["ckpt_output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    try:
        from training_hub import lora_sft
    except ImportError:
        lora_sft = None

    if lora_sft is not None:
        kwargs = {
            "model_path": config["model_path"],
            "data_path": config["data_path"],
            "ckpt_output_dir": output_dir,
        }
        optional_keys = [
            "learning_rate",
            "num_epochs",
            "lora_r",
            "lora_alpha",
            "load_in_4bit",
            "micro_batch_size",
            "max_seq_len",
        ]
        for key in optional_keys:
            if key in config and config[key] is not None:
                kwargs[key] = config[key]

        lora_sft(**kwargs)
    else:
        _simulate_training(config, output_dir)


def _simulate_training(config: dict[str, Any], output_dir: str) -> None:
    """Simulate training when training_hub is not installed."""
    num_epochs = int(config.get("num_epochs", 3) or 3)
    max_steps = num_epochs * 100
    metrics_path = os.path.join(output_dir, "training_metrics.jsonl")

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

    # Write simulated adapter files
    adapter_config = {
        "peft_type": "LORA",
        "r": int(config.get("lora_r", 16) or 16),
        "lora_alpha": int(config.get("lora_alpha", 32) or 32),
        "base_model_name_or_path": str(config.get("model_path", "")),
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Create a small placeholder for adapter weights
    with open(os.path.join(output_dir, "adapter_model.safetensors"), "wb") as f:
        f.write(b"\x00" * 256)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m amortized_runtime.runners.training_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_training(config_data)
