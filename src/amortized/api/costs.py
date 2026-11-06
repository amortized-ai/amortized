"""Cost and resource estimation endpoints."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from amortized.config import settings
from amortized.core.model_arch import ModelArch, resolve_arch

logger = logging.getLogger("amortized.api.costs")

router = APIRouter(prefix="/api/v1/costs", tags=["costs"])

# ---------------------------------------------------------------------------
# Model pricing lookup (local openrouter_costs.json)
# ---------------------------------------------------------------------------

_COSTS_FILE = (
    Path(
        os.environ.get(
            "AMORTIZED_RECIPES_DIR", Path(__file__).resolve().parent.parent.parent.parent
        )
    )
    / "openrouter_costs.json"
)
_pricing_data: list[dict[str, Any]] | None = None


def _load_pricing_data() -> list[dict[str, Any]]:
    global _pricing_data
    if _pricing_data is None:
        with open(_COSTS_FILE, encoding="utf-8") as f:
            _pricing_data = json.load(f)
    return _pricing_data


class ModelPricing(BaseModel):
    model_id: str
    name: str
    prompt_cost_per_1m: float
    completion_cost_per_1m: float
    context_length: int


class GetModelPricingResponse(BaseModel):
    query: str
    models: list[ModelPricing]


@router.get(
    "/models",
    response_model=GetModelPricingResponse,
    operation_id="get_model_pricing",
    summary="Search model pricing by name. Returns matching models with per-1M-token costs.",
)
async def get_model_pricing(
    q: str = Query(
        ..., description="Model name to search for (e.g. 'claude sonnet', 'gpt-4o', 'llama')"
    ),
) -> GetModelPricingResponse:
    data = _load_pricing_data()
    query_lower = q.lower()
    matches = [
        ModelPricing(
            model_id=m["id"],
            name=m["name"],
            prompt_cost_per_1m=m["prompt_cost_per_1m"],
            completion_cost_per_1m=m["completion_cost_per_1m"],
            context_length=m["context_length"],
        )
        for m in data
        if query_lower in m["id"].lower() or query_lower in m["name"].lower()
    ]
    return GetModelPricingResponse(query=q, models=matches[:10])


# ---------------------------------------------------------------------------
# Training resource estimation
#
# Port of training-hub's profiling/memory_estimator.py (BasicEstimator, LoRAEstimator,
# QLoRAEstimator, OSFTEstimator), with the architecture read from the model's HuggingFace
# config instead of loading the model. Each _estimate_vram_* mirrors one estimator class:
# model + gradients + optimizer + activations + outputs, then the 0-30% overhead.
# The target layers for LoRA/OSFT follow mini_trainer's MODEL_CONFIGS default patterns
# (q/k/v/o_proj, gate/up/down_proj); MoE expert weights are not targets.
# ---------------------------------------------------------------------------

FP32 = 4
FP16 = 2
FP8 = 1
FP4 = 0.5
ADAMW_STATES = 2

# training.py defaults when the job config leaves these unset (micro_batch_size 2 x 4).
_DEFAULT_EFFECTIVE_BATCH_SIZE = 8
_DEFAULT_MAX_TOKENS_PER_GPU = 4096
_DEFAULT_MAX_SEQ_LEN = 2048

_GPU_COUNTS = (1, 2, 4, 8)


def _target_weight_shapes(info: ModelArch) -> list[tuple[int, int]]:
    """(out, in) of each per-layer weight matching the default target patterns."""
    h = info.hidden_size
    q_out = info.num_attention_heads * info.head_dim
    kv_out = info.num_kv_heads * info.head_dim
    shapes = [(q_out, h), (kv_out, h), (kv_out, h), (h, q_out)]
    if not info.num_experts:
        inter = info.intermediate_size
        shapes += [(inter, h), (inter, h), (h, inter)]
    return shapes


def _calc_weight_size_total(info: ModelArch) -> int:
    """training-hub _calc_weight_size: sum of (in + out) over LoRA's target weights.
    Catalog models carry training-hub's exact value; others derive it from the config."""
    if info.weight_size_total is not None:
        return info.weight_size_total
    return info.num_hidden_layers * sum(m + n for m, n in _target_weight_shapes(info))


def _calc_osft_params(info: ModelArch) -> float:
    """training-hub _calc_osft_params: bytes to hold the model with each target weight
    (m x n) stored as its SVD factors U (m x k), S (k), V (k x n), k = min(m, n).
    Catalog models carry training-hub's exact value; others derive it from the config."""
    if info.osft_params is not None:
        return info.osft_params
    extra_per_layer = sum(min(m, n) ** 2 + min(m, n) for m, n in _target_weight_shapes(info))
    return (info.num_params + info.num_hidden_layers * extra_per_layer) * FP32


def _resolve_tokens_per_gpu(
    method: str, num_gpus: int, batch_size: int, max_seq_len: int, max_tokens_per_gpu: int
) -> float:
    """training-hub _resolve_tokens_per_gpu; LoRAEstimator ignores max_tokens_per_gpu."""
    tokens_per_gpu = batch_size * max_seq_len / num_gpus
    if method in ("lora", "lora_sft", "qlora", "qlora_sft"):
        return tokens_per_gpu
    return min(max_tokens_per_gpu, tokens_per_gpu)


# Memory components shared by the estimators (training-hub's _calc_* methods).


def _calc_gradients(trainable_params: float, dtype_bytes: float, num_gpus: int) -> float:
    return trainable_params * dtype_bytes / num_gpus


def _calc_optimizer(trainable_params: float, dtype_bytes: float, num_gpus: int) -> float:
    return trainable_params * dtype_bytes * ADAMW_STATES / num_gpus


def _calc_intermediate_activations(
    info: ModelArch, tokens_per_gpu: float, dtype_bytes: float
) -> float:
    return tokens_per_gpu * dtype_bytes * info.num_hidden_layers * info.hidden_size


def _calc_outputs(
    info: ModelArch, tokens_per_gpu: float, dtype_bytes: float, output_constant: float
) -> float:
    return tokens_per_gpu * dtype_bytes * info.vocab_size * output_constant


def _estimate_vram_sft(
    info: ModelArch, tokens_per_gpu: int | float, num_gpus: int
) -> tuple[int, int, int]:
    n = info.num_params

    model_mem = n * FP32 / num_gpus
    grad_mem = _calc_gradients(n, FP32, num_gpus)
    opt_mem = _calc_optimizer(n, FP32, num_gpus)
    act_mem = _calc_intermediate_activations(info, tokens_per_gpu, FP32)
    out_mem = _calc_outputs(info, tokens_per_gpu, FP32, 8 / 3)

    subtotal = model_mem + grad_mem + opt_mem + act_mem + out_mem
    return int(subtotal), int(subtotal * 1.1), int(subtotal * 1.3)


def _estimate_vram_lora(
    info: ModelArch, tokens_per_gpu: int | float, num_gpus: int, lora_r: int
) -> tuple[int, int, int]:
    n = info.num_params
    ab_params = _calc_weight_size_total(info) * lora_r

    model_mem = (n * FP16 + ab_params * FP32) / num_gpus
    grad_mem = _calc_gradients(ab_params, FP16, num_gpus)
    opt_mem = _calc_optimizer(ab_params, FP16, num_gpus)
    act_mem = _calc_intermediate_activations(info, tokens_per_gpu, FP16)
    out_mem = _calc_outputs(info, tokens_per_gpu, FP16, 2)

    subtotal = model_mem + max(grad_mem, out_mem) + opt_mem + act_mem
    return int(subtotal * 0.95), int(subtotal * 1.1), int(subtotal * 1.2)


def _estimate_vram_qlora(
    info: ModelArch, tokens_per_gpu: int | float, num_gpus: int, lora_r: int
) -> tuple[int, int, int]:
    n = info.num_params
    ab_params = _calc_weight_size_total(info) * lora_r

    model_mem = (n * FP4 + ab_params * FP32) / num_gpus
    grad_mem = _calc_gradients(ab_params, FP16, num_gpus)
    opt_mem = _calc_optimizer(ab_params, FP16, num_gpus)
    act_mem = _calc_intermediate_activations(info, tokens_per_gpu, FP16)
    out_mem = _calc_outputs(info, tokens_per_gpu, FP16, 2)

    subtotal = model_mem + max(grad_mem, out_mem) + opt_mem + act_mem
    offload = n * FP8
    effective = offload / num_gpus if subtotal < offload else subtotal
    return int(effective * 0.95), int(effective * 1.1), int(effective * 1.3)


def _estimate_vram_osft(
    info: ModelArch, tokens_per_gpu: int | float, num_gpus: int, unfreeze_ratio: float
) -> tuple[int, int, int]:
    n = info.num_params

    model_mem = _calc_osft_params(info) / num_gpus
    grad_mem = _calc_gradients(n * unfreeze_ratio, FP32, num_gpus)
    opt_mem = _calc_optimizer(n, FP32, num_gpus)
    act_mem = _calc_intermediate_activations(info, tokens_per_gpu, FP32) * unfreeze_ratio
    # training-hub's OSFTEstimator assumes Liger (out_mem = 0), but amortized never sets
    # use_liger and mini_trainer defaults it off, so the fp32 logits are materialized —
    # the 13 GiB allocation that OOMed Qwen3.5-9B (248k vocab) at 15k tokens/GPU.
    out_mem = _calc_outputs(info, tokens_per_gpu, FP32, 7 / 3)

    subtotal = model_mem + grad_mem + opt_mem + act_mem + out_mem
    return int(subtotal), int(subtotal * 1.1), int(subtotal * 1.3)


def _step_batch_size(method: str, body: TrainingEstimateRequest) -> int:
    """Samples per forward pass. lora/qlora run one process that takes
    effective_batch_size // gradient_accumulation_steps per step (training-hub lora.py);
    sft/osft are bounded by max_tokens_per_gpu instead, so they keep the full batch."""
    if method in ("lora", "lora_sft", "qlora", "qlora_sft"):
        return max(1, body.effective_batch_size // body.gradient_accumulation_steps)
    return body.effective_batch_size


def _estimate(
    method: str, info: ModelArch, body: TrainingEstimateRequest, num_gpus: int
) -> tuple[int, int, int]:
    tokens_per_gpu = _resolve_tokens_per_gpu(
        method,
        num_gpus,
        _step_batch_size(method, body),
        body.max_seq_len or _DEFAULT_MAX_SEQ_LEN,
        body.max_tokens_per_gpu or _DEFAULT_MAX_TOKENS_PER_GPU,
    )
    if method == "sft":
        return _estimate_vram_sft(info, tokens_per_gpu, num_gpus)
    elif method in ("lora", "lora_sft"):
        return _estimate_vram_lora(info, tokens_per_gpu, num_gpus, body.lora_r)
    elif method in ("qlora", "qlora_sft"):
        return _estimate_vram_qlora(info, tokens_per_gpu, num_gpus, body.lora_r)
    elif method == "osft":
        return _estimate_vram_osft(info, tokens_per_gpu, num_gpus, body.unfreeze_rank_ratio)
    raise HTTPException(status_code=400, detail=f"Unknown training method: {body.method}")


# Multi-GPU per-GPU memory. training-hub divides every model-state term evenly by num_gpus,
# which is too optimistic for how these jobs actually spread across GPUs:
#
# sft / osft — mini_trainer FSDP2 (each decoder block, then the root module holding the
#   embeddings and lm_head, is its own FSDP unit). Model states shard as training-hub
#   assumes, but each GPU must also gather one whole unit at a time. Following DeepSpeed's
#   estimate_zero3_model_states_mem_needs (deepspeed/runtime/zero/stage3.py), that adds
#   4 bytes (16-bit weight + 16-bit gradient) x the largest unit's parameters to every GPU.
#
# lora / qlora — model splitting (device_map="balanced"): one process places whole layers
#   on each GPU. Weights, adapter, optimizer and the checkpointed activations split about
#   evenly, but the full micro-batch flows through every layer and the last GPU holds the
#   lm_head's full output logits, so the logits term does not split.


def _calc_largest_fsdp_unit(info: ModelArch) -> float:
    """Parameters in the largest FSDP2 unit: the root (embeddings, plus the lm_head when it
    is not tied) or one decoder block."""
    embed = info.vocab_size * info.hidden_size
    root = embed if info.tie_word_embeddings else 2 * embed
    block = (info.num_params - embed) / info.num_hidden_layers
    return max(root, block)


def _estimate_per_gpu(
    method: str, info: ModelArch, body: TrainingEstimateRequest, num_gpus: int
) -> int:
    """Expected bytes per GPU when the job runs on num_gpus GPUs."""
    if num_gpus == 1 or method in ("sft", "osft"):
        _, mid, _ = _estimate(method, info, body, num_gpus)
        if num_gpus == 1:
            return mid
        return mid + int(4 * _calc_largest_fsdp_unit(info) * 1.1)

    # lora / qlora under model splitting: one process, so the tokens are the single-GPU ones.
    tokens_per_gpu = _resolve_tokens_per_gpu(
        method, 1, _step_batch_size(method, body), body.max_seq_len or _DEFAULT_MAX_SEQ_LEN, 0
    )
    n = info.num_params
    ab_params = _calc_weight_size_total(info) * body.lora_r
    model_bytes = FP4 if method in ("qlora", "qlora_sft") else FP16

    model_mem = n * model_bytes + ab_params * FP32
    grad_mem = _calc_gradients(ab_params, FP16, 1)
    opt_mem = _calc_optimizer(ab_params, FP16, 1)
    act_mem = _calc_intermediate_activations(info, tokens_per_gpu, FP16)
    out_mem = _calc_outputs(info, tokens_per_gpu, FP16, 2)

    per_gpu = (model_mem + opt_mem + act_mem) / num_gpus + max(grad_mem / num_gpus, out_mem)
    if model_bytes == FP4:
        # The 8-bit pre-quantization load floors peak memory. training-hub replaces the
        # subtotal with it, which is the same on one GPU but would lower the per-GPU figure.
        per_gpu = max(per_gpu, n * FP8 / num_gpus)
    return int(per_gpu * 1.1)


class TrainingEstimateRequest(BaseModel):
    model_name_or_path: str = Field(
        ...,
        min_length=1,
        description="HuggingFace model id (e.g. 'Qwen/Qwen3.5-4B'); its config is read for "
        "the architecture",
    )
    method: str = Field("lora", description="Training method: sft, osft, lora, qlora")
    effective_batch_size: int = Field(
        _DEFAULT_EFFECTIVE_BATCH_SIZE,
        ge=1,
        description="Same value as the job config (default 8): samples per optimizer step "
        "across all GPUs",
    )
    max_seq_len: int | None = Field(
        None, ge=1, description="Max sequence length (the config's max_length; default 2048)"
    )
    max_tokens_per_gpu: int | None = Field(
        None, ge=1, description="sft/osft token budget per GPU per step (job default 4096)"
    )
    gradient_accumulation_steps: int = Field(
        1,
        ge=1,
        description="lora/qlora only: same value as the job config (default 1); each step "
        "runs effective_batch_size // gradient_accumulation_steps samples",
    )
    lora_r: int = Field(32, ge=1, description="LoRA rank (lora/qlora only)")
    unfreeze_rank_ratio: float = Field(
        0.2, ge=0.0, le=1.0, description="OSFT unfreeze ratio (osft only)"
    )


class TrainingEstimateResponse(BaseModel):
    model: str
    model_size: str = Field(..., description="Parameter count, e.g. '4.7B'")
    arch_source: str = Field(
        ...,
        description="catalog (built-in table of verified models) or huggingface (config "
        "fetched from the hub)",
    )
    method: str
    vram_gb: dict[str, float] = Field(
        ..., description="VRAM to train on a single GPU: low / expected / high"
    )
    per_gpu_vram_gb_by_gpu_count: dict[str, float] = Field(
        ...,
        description="Expected VRAM each GPU needs when the job uses 1, 2, 4 or 8 GPUs "
        "(nproc_per_node). Pick the smallest count whose value fits the GPU's memory.",
    )
    gpu_budget: int = Field(
        ...,
        description="Max GPUs (nproc_per_node) one job may use; a larger count is rejected "
        "at submission",
    )


@router.post(
    "/training/estimate",
    response_model=TrainingEstimateResponse,
    operation_id="estimate_training_resources",
    summary=(
        "Estimate the VRAM a training job (sft, osft, lora, qlora) needs, from the "
        "HuggingFace model id and method. Returns the single-GPU need and the per-GPU need "
        "for 1/2/4/8 GPUs, to suggest nproc_per_node."
    ),
)
async def estimate_training_resources(
    body: TrainingEstimateRequest,
) -> TrainingEstimateResponse:
    try:
        info = await resolve_arch(body.model_name_or_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    method = body.method.lower()
    low, mid, high = _estimate(method, info, body, num_gpus=1)

    to_gb = 1024**3
    return TrainingEstimateResponse(
        model=body.model_name_or_path,
        model_size=info.label,
        arch_source=info.source,
        method=body.method,
        vram_gb={
            "low": round(low / to_gb, 1),
            "expected": round(mid / to_gb, 1),
            "high": round(high / to_gb, 1),
        },
        per_gpu_vram_gb_by_gpu_count={
            str(n): round(_estimate_per_gpu(method, info, body, num_gpus=n) / to_gb, 1)
            for n in _GPU_COUNTS
        },
        gpu_budget=settings.gpu_budget,
    )
