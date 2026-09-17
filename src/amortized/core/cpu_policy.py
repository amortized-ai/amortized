"""CPU training guardrails — the single warn-vs-reject matrix.

Model size x method x device -> {ok, warn, reject}. Validation
(``/jobs/training/validate``), agent prompts, and Studio copy all derive from
this module so the rules cannot drift apart (#370, #415). GPU jobs (the
default) are untouched.
"""

from __future__ import annotations

from typing import Any

from amortized.core.model_catalog import training_model_cpu_compatibility

# vLLM is CUDA-only: online-RL and vLLM-backed methods cannot run on CPU.
_CPU_REJECTED_ALGORITHMS = frozenset({"grpo", "lora_grpo", "gepa"})

# Allowlist: the ONLY algorithms the CPU runner (containers/training/cpu_sft.py,
# dispatched by amortized.jobs.training) actually implements. Everything else
# must be rejected here — cpu_sft.py would otherwise silently run plain SFT or
# LoRA for configs it does not understand (e.g. osft, dpo, kto, gkd).
_CPU_ALLOWED_ALGORITHMS = frozenset({"sft", "lora_sft"})

# Algorithm values that mean QLoRA regardless of which flags are set.
_CPU_QLORA_ALGORITHMS = frozenset({"qlora", "qlora_sft"})

# The builder aliases these in amortized.jobs.training (algo_aliases); the
# policy sees the raw config value, so normalize the alias here too.
_CPU_ALGORITHM_ALIASES = {"lora": "lora_sft"}

_CPU_NOTICE = (
    "CPU training is slow by design — tiny models and smoke tests only. "
    "A job that looks hung is probably just slow; check the logs before cancelling."
)


def check_cpu_policy(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Apply the CPU guardrail matrix to a training job config.

    Returns ``(errors, warnings)`` — both empty unless ``device == "cpu"``.
    Rejections must block job creation; warnings surface on the confirmation
    card via ``ValidatedJobConfig.warnings``.
    """
    if config.get("device") != "cpu":
        return [], []

    errors: list[str] = []
    warnings: list[str] = [_CPU_NOTICE]

    model = config.get("model_name_or_path", "")
    support = training_model_cpu_compatibility(model)
    if support == "reject":
        errors.append(f"{model} is too large for CPU training — use a smaller model (≤2B) or a GPU")
    elif support == "warn":
        warnings.append(f"{model} on CPU is very slow — expect hours; consider a GPU")
    elif support is None:
        warnings.append(
            f"{model} not in the supported catalog — CPU compatibility unknown; "
            "training may be very slow or fail"
        )

    algorithm = _CPU_ALGORITHM_ALIASES.get(config.get("algorithm", ""), config.get("algorithm", ""))
    if algorithm in _CPU_REJECTED_ALGORITHMS:
        errors.append(f"{algorithm} requires vLLM, which is CUDA-only — it cannot run on CPU")
    elif algorithm == "osft":
        errors.append(
            "OSFT requires the GPU training path (thub/instructlab) — use lora_sft or sft on CPU"
        )
    elif (
        algorithm in _CPU_QLORA_ALGORITHMS
        or config.get("load_in_4bit")
        or config.get("qlora")
        or config.get("bnb_4bit_quant_type")
    ):
        errors.append(
            "QLoRA / 4-bit quantization is not supported on CPU "
            "(bitsandbytes CPU support is experimental)"
        )
    elif algorithm not in _CPU_ALLOWED_ALGORITHMS:
        errors.append(f"{algorithm} is not supported for CPU training — supported: sft, lora_sft")

    if config.get("bf16"):
        warnings.append("bf16 mixed precision has no effect on CPU — training runs in fp32")

    if (
        algorithm == "sft"
        and not config.get("use_peft")
        and not config.get("load_in_4bit")
        and support == "ok"
    ):
        warnings.append(
            "full-parameter SFT on CPU needs ~16 bytes/param of AdamW optimizer state "
            "in RAM — a 0.8B model needs ~13 GB, more than the 8 GB the builder "
            "allocates for it, so expect an OOM kill — consider LoRA or a GPU"
        )

    if (config.get("nproc_per_node") or 1) > 1:
        warnings.append(
            "nproc_per_node > 1 is not supported for CPU training; it will be capped at 1"
        )

    return errors, warnings
