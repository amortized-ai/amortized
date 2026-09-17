"""Merge a PEFT LoRA adapter into its base model for serving.

lora_sft training runs export ONLY the adapter (adapter_config.json +
adapter_model.safetensors, plus checkpoints) — there is no merged
model/hf_format export like sft/osft produce. vLLM cannot serve a bare
adapter, so the eval flow merges it first: load the base model named by
adapter_config.json's base_model_name_or_path, apply the adapter,
merge_and_unload(), and save a standard checkpoint. This mirrors what
patch_model_config.py does for bare text towers — a pre-serve transform
that makes the export servable.

The merge is architecture-agnostic: target modules are derived from the
checkpoint keys themselves and validated against the loaded base, so
any adapter whose modules exist in the base model merges cleanly. Two
naming quirks are auto-detected rather than assumed:

1. Composite-trained checkpoints (the Qwen3.5 text tower is the known
   case) carry an extra ``language_model`` segment in their module
   paths (``model.language_model.layers.*``) that the flat-loaded base
   doesn't have. Module paths are matched as-is first; only when that
   fails is the segment stripped. The adapter's auto-generated
   target_modules regex has the same blind spot (it covers
   linear-attention modules only under the composite branch), so it is
   replaced by the exact module list from the checkpoint.

2. save_pretrained may write weights in the hub's checkpoint naming
   (``model.language_model.*``) instead of the model's own parameter
   names. The saved shards are re-keyed only when the flattened form
   matches the merged model's parameters — otherwise they are left
   untouched, and an unexpected naming fails loudly instead of
   producing a corrupt checkpoint.

Usage: python3 merge_lora.py <adapter_dir> <out_dir>
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile


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

    patched = _patched_adapter(adapter_dir, model, dtype)
    try:
        pm = PeftModel.from_pretrained(model, patched, autocast_adapter_dtype=False)
        merged = pm.merge_and_unload()
        param_names = {n for n, _ in merged.named_parameters(remove_duplicate=False)}
        merged.save_pretrained(out_dir)
        _rekey_checkpoint(out_dir, param_names)
    finally:
        shutil.rmtree(patched, ignore_errors=True)

    # Tokenizer: the adapter dir carries the training-time tokenizer
    # (chat template included); fall back to the base model's.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.save_pretrained(out_dir)

    print(f"merged adapter {adapter_dir} into {base_id} -> {out_dir}")
    return 0


def _patched_adapter(adapter_dir: str, model, dtype) -> str:
    """Rewrite the adapter so peft's standard loader applies it fully.

    Tensor keys are matched against the loaded base's module names
    (as-is first, then with a composite ``language_model`` segment
    stripped), and target_modules is replaced by the exact module list
    the checkpoint was trained on — peft then injects (and loads)
    every one of them.
    """
    from safetensors.torch import load_file, save_file

    path = f"{adapter_dir}/adapter_model.safetensors"
    if not glob.glob(path):
        raise SystemExit(f"no adapter_model.safetensors in {adapter_dir}")
    tensors = load_file(path)

    module_names = {n for n, _ in model.named_modules()}
    remapped: dict = {}
    targets: set[str] = set()
    flattened = 0
    for key, tensor in tensors.items():
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if not key.endswith(suffix):
                continue
            target = key[: -len(suffix)].removeprefix("base_model.model.")
            if target not in module_names:
                # Composite-trained checkpoints carry an extra
                # language_model segment; standard checkpoints match
                # the loaded model as-is.
                stripped = target.replace("model.language_model.", "model.")
                if stripped in module_names:
                    target = stripped
                    flattened += 1
            if target not in module_names:
                raise SystemExit(f"adapter targets unknown module: {target}")
            targets.add(target)
            remapped[f"base_model.model.{target}{suffix}"] = tensor.to(dtype)
            break
        else:
            raise SystemExit(f"unexpected adapter tensor (not lora_A/lora_B): {key}")

    # LoRA initializes B at zero; if every B is still zero the merge is
    # a no-op and something went wrong upstream (e.g. an untrained export).
    b_keys = [k for k in remapped if k.endswith(".lora_B.weight")]
    if b_keys and all(remapped[k].abs().sum().item() == 0 for k in b_keys):
        raise SystemExit("all adapter lora_B tensors are zero — merge would be a no-op")

    with open(f"{adapter_dir}/adapter_config.json") as f:
        config = json.load(f)
    config["target_modules"] = sorted(targets)

    patched = tempfile.mkdtemp(
        prefix="patched-adapter-", dir=os.path.dirname(os.path.abspath(adapter_dir))
    )
    save_file(remapped, f"{patched}/adapter_model.safetensors")
    with open(f"{patched}/adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(
        f"patched adapter: {len(targets)} target modules, {len(remapped)} tensors"
        f" ({flattened} keys flattened from composite naming)"
    )
    return patched


def _rekey_checkpoint(out_dir: str, param_names: set[str]) -> None:
    """Make the saved weights match the model's own parameter names.

    Some families (the Qwen3.5 text tower is the known case) are saved
    in the hub's composite checkpoint naming (model.language_model.*)
    while vLLM expects the flat parameter names. The shards are
    re-keyed only when the flattened form matches the merged model's
    parameters; anything else fails loudly rather than corrupting the
    checkpoint.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    def flatten(key: str) -> str:
        return key.replace("model.language_model.", "model.")

    shards = []
    all_keys: set[str] = set()
    for shard in sorted(glob.glob(f"{out_dir}/*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            shards.append((shard, {k: f.get_tensor(k) for k in keys}))
        all_keys |= set(keys)

    if all_keys <= param_names:
        # save_pretrained already wrote the model's own parameter names
        # (the normal case for most architectures) — nothing to do.
        print(f"checkpoint keys already match parameter names ({len(all_keys)} tensors)")
        return
    if not {flatten(k) for k in all_keys} <= param_names:
        raise SystemExit(
            "saved checkpoint naming matches neither the merged model's parameter"
            " names nor the flattened text-tower form — refusing to rekey"
        )
    for shard, tensors in shards:
        save_file({flatten(k): v for k, v in tensors.items()}, shard)
    index_path = f"{out_dir}/model.safetensors.index.json"
    if glob.glob(index_path):
        with open(index_path) as f:
            index = json.load(f)
        index["weight_map"] = {flatten(k): v for k, v in index["weight_map"].items()}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    print(f"rekeyed checkpoint shards in {out_dir} to flat parameter names")


def _base_model_id(adapter_dir: str) -> str:
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
