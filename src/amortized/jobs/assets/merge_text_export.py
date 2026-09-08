"""Merge a Training Hub text-only HF export back into its base multimodal model.

Training Hub's hf_format export saves Qwen3.5 checkpoints as the bare text
tower (model_type qwen3_5_text, weights model.layers.*) which vLLM cannot
serve. This script rebuilds a servable ConditionalGeneration checkpoint:
base model weights + config (vision tower untouched), with the fine-tuned
text-tower weights grafted in (model.X -> model.language_model.X).

Usage: merge_text_export.py <tuned_export_dir> <base_model_id> <out_dir>
"""

import json
import os
import shutil
import sys

from safetensors.torch import load_file, save_file


def main() -> None:
    tuned_dir, base_model, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    # 1. Download the base model (weights land in HF cache)
    from huggingface_hub import snapshot_download

    base_dir = snapshot_download(
        base_model, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"]
    )

    # 2. Verify the tuned export really is a bare text tower
    with open(os.path.join(tuned_dir, "config.json")) as f:
        tuned_cfg = json.load(f)
    if tuned_cfg.get("model_type") != "qwen3_5_text":
        # Nothing to merge — copy as-is (already servable)
        print(f"model_type={tuned_cfg.get('model_type')!r} — copying without merge")
        shutil.copytree(tuned_dir, out_dir, dirs_exist_ok=True)
        return

    # 3. Load base weights and graft the tuned text tower
    base_st = os.path.join(base_dir, "model.safetensors-00001-of-00001.safetensors")
    if not os.path.exists(base_st):
        shards = [
            f
            for f in os.listdir(base_dir)
            if f.endswith(".safetensors") and not f.endswith("index.json")
        ]
        if len(shards) != 1:
            raise SystemExit(f"expected single base shard, found {shards}")
        base_st = os.path.join(base_dir, shards[0])

    merged = load_file(base_st)
    tuned = load_file(os.path.join(tuned_dir, "model.safetensors"))
    grafted = 0
    skipped = []
    for key, tensor in tuned.items():
        if key == "lm_head.weight":
            # Base uses tie_word_embeddings — lm_head is served as embed_tokens
            # (identical tensor in the export), which is grafted below.
            continue
        if key.startswith("model."):
            base_key = f"model.language_model.{key[len('model.') :]}"
        else:
            base_key = key
        if base_key not in merged:
            # Non-persistent buffers (e.g. rotary_emb.inv_freq) are recomputed
            if "rotary_emb" in key or "inv_freq" in key:
                skipped.append(key)
                continue
            raise SystemExit(f"tuned weight {key!r} has no base counterpart {base_key!r}")
        merged[base_key] = tensor
        grafted += 1
    print(f"grafted {grafted} tensors onto base model (skipped: {skipped})")

    save_file(merged, os.path.join(out_dir, "model.safetensors"))

    # 4. Base config + tokenizers (servable wrapper), tuned chat template
    for fn in os.listdir(base_dir):
        if fn.endswith((".json", ".txt")) and fn != "model.safetensors.index.json":
            shutil.copy(os.path.join(base_dir, fn), os.path.join(out_dir, fn))
    tuned_template = os.path.join(tuned_dir, "chat_template.jinja")
    if os.path.exists(tuned_template):
        shutil.copy(tuned_template, os.path.join(out_dir, "chat_template.jinja"))
    print(f"merged model written to {out_dir}")


if __name__ == "__main__":
    main()
