"""Merge a PEFT LoRA adapter into its base model for serving.

lora_sft training runs export ONLY the adapter (adapter_config.json +
adapter_model.safetensors, plus checkpoints) — there is no merged
model/hf_format export like sft/osft produce. vLLM cannot serve a bare
adapter, so the eval flow merges it first: load the base model named by
adapter_config.json's base_model_name_or_path, apply the adapter,
merge_and_unload(), and save a standard checkpoint. This mirrors what
patch_model_config.py does for bare text towers — a pre-serve transform
that makes the export servable.

Two mismatches between the training-side save and this image's loading
stack are handled explicitly:

1. The training stack fine-tuned the composite model whose module paths
   carry a ``language_model`` segment (``model.language_model.layers.*``),
   while transformers here loads the same weights flat
   (``model.layers.*``). The adapter's auto-generated target_modules
   regex covers the composite naming for linear-attention modules but
   NOT the flat one, so peft would skip them (and every checkpoint key
   would miss its module). The adapter is therefore rewritten into a
   patched dir: tensor keys remapped flat, and target_modules replaced
   by the explicit module list derived from the checkpoint itself.

2. save_pretrained writes weights in the hub's checkpoint naming for
   this family (``model.language_model.*``), while vLLM (via the same
   text-tower serving the osft exports use) expects the flat
   ``model.*`` names. The saved safetensors are re-keyed to flat.

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
        merged.save_pretrained(out_dir)
        _rekey_checkpoint(out_dir)
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

    - Tensor keys: strip the training-side ``language_model`` segment
      (model.language_model.layers... -> model.layers...).
    - target_modules: the config's regex only covers linear-attention
      modules under the composite naming, so replace it with the exact
      module list the checkpoint was trained on — peft then injects
      (and loads) every one of them.
    """
    from safetensors.torch import load_file, save_file

    path = f"{adapter_dir}/adapter_model.safetensors"
    if not glob.glob(path):
        raise SystemExit(f"no adapter_model.safetensors in {adapter_dir}")
    tensors = load_file(path)
    remapped = {k.replace("model.language_model.", "model."): v for k, v in tensors.items()}

    module_names = {n for n, _ in model.named_modules()}
    targets: set[str] = set()
    for key in remapped:
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if key.endswith(suffix):
                target = key[: -len(suffix)].removeprefix("base_model.model.")
                if target not in module_names:
                    raise SystemExit(f"adapter targets unknown module: {target}")
                targets.add(target)
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
    save_file(
        {k: v.to(dtype) for k, v in remapped.items()},
        f"{patched}/adapter_model.safetensors",
    )
    with open(f"{patched}/adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"patched adapter: {len(targets)} target modules, {len(remapped)} tensors")
    return patched


def _rekey_checkpoint(out_dir: str) -> None:
    """Flatten the saved weights to the text-tower naming vLLM expects.

    save_pretrained writes the hub's composite checkpoint naming
    (model.language_model.*); strip the segment so the export matches
    the osft hf_format layout (model.* + lm_head), which the patched
    vLLM registry serves.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    def rekey(key: str) -> str:
        return key.replace("model.language_model.", "model.")

    for shard in sorted(glob.glob(f"{out_dir}/*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            tensors = {k: f.get_tensor(k) for k in keys}
        rekeyed = {rekey(k): v for k, v in tensors.items()}
        if rekeyed != tensors:
            save_file(rekeyed, shard)
    index_path = f"{out_dir}/model.safetensors.index.json"
    if glob.glob(index_path):
        with open(index_path) as f:
            index = json.load(f)
        index["weight_map"] = {rekey(k): v for k, v in index["weight_map"].items()}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    with open(f"{out_dir}/config.json") as f:
        config = json.load(f)
    print(f"rekeyed checkpoint shards in {out_dir} (config model_type={config.get('model_type')})")


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
