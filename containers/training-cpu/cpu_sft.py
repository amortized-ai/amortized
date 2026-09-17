#!/usr/bin/env python3
"""CPU-only SFT runner for the amortized training-cpu image.

The CPU image deliberately excludes unsloth (lora_sft's only training-hub
backend, it caps transformers<5) and instructlab-training (thub sft's only
backend) is hard-CUDA (unconditional ``torch.cuda.set_device()``), so the
``thub`` CLI the GPU path dispatches cannot run here. This script is what the
CPU branch of the training job builder dispatches instead: it consumes the
SAME builder-generated thub-format ``config.yaml`` and runs TRL SFTTrainer +
PEFT LoRA on plain CPU — the stack this image ships (torch+cpu, trl, peft,
accelerate).

Config keys (all emitted by ``amortized.jobs.training``):

- ``model_path``            HuggingFace model id or local path (required)
- ``data_path``             JSONL chat-format dataset, ``{"messages": [...]}``
- ``learning_rate``         default 2e-4 (LoRA) / 2e-5 (full-param)
- ``num_epochs``            default 1
- ``effective_batch_size``  default 8; split into micro-batch x grad-accum
- ``max_seq_len``           default 2048
- ``lora_r``                present -> PEFT LoRA; absent -> full-param SFT
- ``lora_alpha``            default 2 * lora_r
- ``lora_dropout``          default 0.05
- ``lora_target_modules``   optional explicit override; otherwise every
                            ``torch.nn.Linear`` of the loaded model is targeted
                            (PEFT cannot auto-infer for model types missing
                            from its mapping, e.g. Qwen3.5)
- ``ckpt_output_dir``       final model/adapter saved to ``<dir>/final``
- ``gradient_checkpointing``optional
- ``bf16``                  accepted but ignored — CPU is always fp32 here

CPU hard-rules (mirroring the builder's enforced defaults): fp32 only
(bf16=False), no CUDA calls, and a single dataloader worker —
instructlab-training's default ``num_cpu_procs=16`` map pool is exactly what
crashes small CPU containers, so this runner keeps its own workers at 1.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_CONFIG_PATH = "/amortized/config.yaml"
DEFAULT_OUTPUT_DIR = "/amortized/work/output"
# Keep the micro-batch modest: CPU jobs are sized for tiny models in small
# memory limits, so we never ask for more than 4 sequences at a time and make
# up the effective batch with gradient accumulation.
MAX_MICRO_BATCH = 4


def _load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} is not a mapping")
    missing = [k for k in ("model_path", "data_path") if not cfg.get(k)]
    if missing:
        raise ValueError(f"config {path} is missing required keys: {missing}")
    return cfg


def _load_model(model_path: str) -> Any:
    """Load the base model in fp32 on CPU.

    transformers 5.x renamed ``torch_dtype`` -> ``dtype``; try the new name
    first and fall back for older versions.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    return model.to("cpu")


def _resolve_lora_target_modules(cfg: dict[str, Any], model: Any) -> list[str]:
    """Resolve LoRA ``target_modules`` for the loaded model.

    PEFT only auto-infers target modules for model types listed in its
    ``target_module_mapping``; anything not yet in that mapping (e.g.
    Qwen3.5's ``model_type='qwen3_5'`` in peft 0.21) raises
    ``ValueError: Please specify target_modules`` at adapter injection —
    before a single training step. Instead of relying on PEFT's mapping,
    default to every linear projection of the loaded model (each named
    module matches itself exactly in PEFT's target check), EXCLUDING the
    output embedding layer (``lm_head``): TRL's default ``chunked_nll``
    loss refuses a PEFT-wrapped lm_head, and PEFT's own ``all-linear``
    mode excludes it too. An explicit ``lora_target_modules`` list in the
    config wins if present.
    """
    explicit = cfg.get("lora_target_modules")
    if explicit:
        modules = [str(m) for m in explicit]
    else:
        output_embedding = model.get_output_embeddings()
        modules = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and module is not output_embedding
        ]
    if not modules:
        raise ValueError(
            "no LoRA target modules found: the model has no torch.nn.Linear layers "
            "and 'lora_target_modules' is not set in the config"
        )
    return modules


def _build_sft_args(cfg: dict[str, Any], ckpt_output_dir: str) -> SFTConfig:
    """Map thub-format config keys onto TRL SFTConfig (CPU-forced values)."""
    lora_r = cfg.get("lora_r")
    lr_default = 2e-4 if lora_r else 2e-5
    effective_batch = int(cfg.get("effective_batch_size", 8) or 8)
    micro_batch = max(1, min(effective_batch, MAX_MICRO_BATCH))
    grad_accum = max(1, effective_batch // micro_batch)
    max_seq_len = int(cfg.get("max_seq_len", 2048) or 2048)

    # trl renamed SFTConfig.max_seq_length -> max_length in newer releases;
    # the image may carry either, so adapt to whichever SFTConfig declares.
    params = inspect.signature(SFTConfig.__init__).parameters
    length_kwarg = {"max_length" if "max_length" in params else "max_seq_length": max_seq_len}

    return SFTConfig(
        output_dir=ckpt_output_dir,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(cfg.get("learning_rate", lr_default) or lr_default),
        num_train_epochs=float(cfg.get("num_epochs", 1) or 1),
        # CPU hard rules: fp32, single-process, no CUDA anywhere.
        use_cpu=True,
        bf16=False,
        fp16=False,
        # instructlab-training's num_cpu_procs=16 default crashes small CPU
        # containers; this runner owns its dataloader and keeps it at 1.
        dataloader_num_workers=1,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        # Save only the final model (below); no intermediate checkpoints for
        # smoke-test-sized CPU jobs.
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        **length_kwarg,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only SFT runner (TRL + PEFT)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="thub-format config.yaml")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    model_path = cfg["model_path"]
    data_path = cfg["data_path"]
    ckpt_output_dir = cfg.get("ckpt_output_dir", DEFAULT_OUTPUT_DIR)

    lora_r = cfg.get("lora_r")
    lora_alpha = cfg.get("lora_alpha") or (2 * lora_r if lora_r else None)
    lora_dropout = cfg.get("lora_dropout")
    print("cpu_sft config:", json.dumps(cfg, default=str), flush=True)
    print(f"mode: {'LoRA (peft)' if lora_r else 'full-parameter SFT'}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(model_path)
    if lora_r and cfg.get("gradient_checkpointing"):
        # PEFT + gradient checkpointing needs inputs to require grad.
        model.enable_input_require_grads()

    dataset = load_dataset("json", data_files=data_path, split="train")
    if "messages" not in dataset.column_names:
        print(
            f"error: dataset {data_path} has columns {dataset.column_names}; "
            "expected JSONL chat format with a 'messages' column",
            file=sys.stderr,
        )
        return 2
    print(f"dataset: {len(dataset)} examples from {data_path}", flush=True)

    peft_config = None
    if lora_r:
        peft_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout) if lora_dropout is not None else 0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_resolve_lora_target_modules(cfg, model),
        )

    sft_args = _build_sft_args(cfg, ckpt_output_dir)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    final_dir = os.path.join(ckpt_output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"TRAINING COMPLETE — model saved to {final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
