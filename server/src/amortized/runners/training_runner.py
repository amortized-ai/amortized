"""Training job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Dispatches to the appropriate
training_hub function based on config["algorithm"]. Falls back to a simulated
run if training_hub is not installed.
"""

import json
import os
import sys
import time
from typing import Any

_ALGORITHM_PARAMS: dict[str, list[str]] = {
    "lora_sft": [
        "learning_rate",
        "num_epochs",
        "lora_r",
        "lora_alpha",
        "load_in_4bit",
        "micro_batch_size",
        "max_seq_len",
        "bf16",
    ],
    "full_sft": [
        "learning_rate",
        "num_epochs",
        "effective_batch_size",
        "max_seq_len",
        "warmup_steps",
        "max_tokens_per_gpu",
        "bf16",
    ],
    "grpo": [
        "learning_rate",
        "num_iterations",
        "group_size",
        "prompt_batch_size",
        "temperature",
        "max_tokens",
        "lora_r",
        "lora_alpha",
        "gpu_memory_utilization",
    ],
    "osft": [
        "learning_rate",
        "num_epochs",
        "effective_batch_size",
        "max_seq_len",
        "unfreeze_rank_ratio",
        "use_liger",
        "lr_scheduler",
        "bf16",
    ],
    "gepa": [
        "seed_candidate",
        "task_lm",
        "num_iterations",
    ],
}

_ALGORITHM_ALIASES: dict[str, str] = {
    "sft": "full_sft",
    "lora_grpo": "grpo",
}


def _import_training_func(algorithm: str) -> Any:
    """Import the training_hub function for the given algorithm."""
    import_map: dict[str, tuple[str, str]] = {
        "lora_sft": ("training_hub", "lora_sft"),
        "full_sft": ("training_hub", "sft"),
        "grpo": ("training_hub", "lora_grpo"),
        "osft": ("training_hub", "osft"),
        "gepa": ("training_hub", "gepa"),
    }
    module_name, func_name = import_map[algorithm]
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _build_kwargs(config: dict[str, Any], algorithm: str, output_dir: str) -> dict[str, Any]:
    """Build kwargs for the training function from config."""
    if algorithm == "gepa":
        kwargs: dict[str, Any] = {}
    else:
        kwargs = {
            "model_path": config["model_path"],
            "ckpt_output_dir": output_dir,
        }
        if algorithm != "grpo":
            kwargs["data_path"] = config["data_path"]

    for key in _ALGORITHM_PARAMS.get(algorithm, []):
        if key in config and config[key] is not None:
            kwargs[key] = config[key]

    return kwargs


def run_training(config: dict[str, Any]) -> None:
    """Execute a training job using the algorithm specified in config."""
    output_dir = str(config["ckpt_output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    algorithm = config.get("algorithm", "lora_sft")
    algorithm = _ALGORITHM_ALIASES.get(algorithm, algorithm)

    if algorithm not in _ALGORITHM_PARAMS:
        raise ValueError(
            f"Unknown training algorithm: {algorithm!r}. "
            f"Supported: {', '.join(sorted(_ALGORITHM_PARAMS))}"
        )

    try:
        train_func = _import_training_func(algorithm)
    except ImportError:
        train_func = None

    # Clear stale metrics from any previous run to avoid appending to old data
    metrics_path = os.path.join(output_dir, "training_metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    if train_func is not None:
        kwargs = _build_kwargs(config, algorithm, output_dir)
        train_func(**kwargs)
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
        print("Usage: python -m amortized.runners.training_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_training(config_data)
