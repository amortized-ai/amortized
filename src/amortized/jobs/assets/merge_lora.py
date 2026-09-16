"""Merge a PEFT LoRA adapter into its base model for serving.

lora_sft training runs export ONLY the adapter (adapter_config.json +
adapter_model.safetensors, plus checkpoints) — there is no merged
model/hf_format export like sft/osft produce. vLLM cannot serve a bare
adapter, so the eval flow merges it first: load the base model named by
adapter_config.json's base_model_name_or_path (downloaded from the hub
if needed), apply the adapter, merge_and_unload(), and save a standard
checkpoint. This mirrors what patch_model_config.py does for bare text
towers — a pre-serve transform that makes the export servable.

Usage: python3 merge_lora.py <adapter_dir> <out_dir>
"""

from __future__ import annotations

import sys


def main() -> int:
    adapter_dir, out_dir = sys.argv[1], sys.argv[2]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The base model id lives in the adapter config — the model the
    # training job actually fine-tuned on top of.
    base_id = _base_model_id(adapter_dir)

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)

    # Tokenizer: the adapter dir carries the training-time tokenizer
    # (chat template included); fall back to the base model's.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.save_pretrained(out_dir)

    # processor_config.json / training_args.bin / checkpoint-* are
    # training artifacts, not servable weights — save_pretrained wrote
    # everything the server needs.
    print(f"merged adapter {adapter_dir} into {base_id} -> {out_dir}")
    return 0


def _base_model_id(adapter_dir: str) -> str:
    import json

    with open(f"{adapter_dir}/adapter_config.json") as f:
        config = json.load(f)
    base = str(config.get("base_model_name_or_path", "")).strip()
    if not base:
        raise SystemExit(
            f"adapter_config.json in {adapter_dir} has no base_model_name_or_path"
        )
    return base


if __name__ == "__main__":
    sys.exit(main())
