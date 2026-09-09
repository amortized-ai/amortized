"""Make a Training Hub text-backbone export servable by adding architectures.

Training Hub (mini_trainer) extracts the CausalLM text backbone from VLM
models for training; its hf_format exports carry the bare text config
(model_type like "qwen3_5_text", no "architectures" key). vLLM (and most
serving stacks) require an explicit architecture. The correct class is
resolved generically via transformers' own AutoModelForCausalLM registry —
the same mapping HF uses when loading these checkpoints for inference.

Usage: python3 patch_model_config.py <model_dir>
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    model_dir = sys.argv[1]
    config_path = f"{model_dir}/config.json"

    with open(config_path) as f:
        config = json.load(f)

    if config.get("architectures"):
        print(f"architectures already set: {config['architectures']}")
        return 0

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config)
    arch = type(model).__name__

    config["architectures"] = [arch]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"patched architectures: [{arch}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
