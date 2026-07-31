"""Cost and resource estimation endpoints."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

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
# Training resource estimation (static model size buckets)
# ---------------------------------------------------------------------------

FP32 = 4
FP16 = 2
FP8 = 1
FP4 = 0.5
ADAMW_STATES = 2

MODEL_SIZE_BUCKETS: list[dict[str, Any]] = [
    {
        "label": "0.5B",
        "num_params": 500_000_000,
        "num_hidden_layers": 24,
        "hidden_size": 896,
        "vocab_size": 151936,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
    },
    {
        "label": "1B",
        "num_params": 1_000_000_000,
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "vocab_size": 151936,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
    },
    {
        "label": "3B",
        "num_params": 3_000_000_000,
        "num_hidden_layers": 36,
        "hidden_size": 2048,
        "vocab_size": 151936,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
    },
    {
        "label": "8B",
        "num_params": 8_000_000_000,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "vocab_size": 128256,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    },
    {
        "label": "13B",
        "num_params": 13_000_000_000,
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "vocab_size": 32000,
        "intermediate_size": 13824,
        "num_attention_heads": 40,
        "num_key_value_heads": 40,
    },
    {
        "label": "32B",
        "num_params": 32_000_000_000,
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "vocab_size": 152064,
        "intermediate_size": 27648,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
    },
    {
        "label": "70B",
        "num_params": 70_000_000_000,
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "vocab_size": 128256,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
    },
    {
        "label": "120B",
        "num_params": 120_000_000_000,
        "num_hidden_layers": 96,
        "hidden_size": 8192,
        "vocab_size": 128256,
        "intermediate_size": 32768,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
    },
]


def _resolve_model_size(model_size: str) -> dict[str, Any]:
    """Resolve a model size label or param count to architecture metadata."""
    size_lower = model_size.lower().strip()

    for bucket in MODEL_SIZE_BUCKETS:
        if bucket["label"].lower() == size_lower:
            return bucket

    try:
        raw = size_lower.rstrip("b")
        num = float(raw) * 1_000_000_000
    except ValueError:
        num = 0.0

    if num <= 0:
        return MODEL_SIZE_BUCKETS[0]

    closest = min(MODEL_SIZE_BUCKETS, key=lambda b: abs(b["num_params"] - num))
    return closest


def _calc_weight_size_total(info: dict[str, Any]) -> int:
    """Calculate total LoRA-targetable weight dimensions from model config."""
    hidden = info["hidden_size"]
    intermediate = info.get("intermediate_size", hidden * 4)
    num_heads = info.get("num_attention_heads", 1)
    num_kv_heads = info.get("num_key_value_heads", num_heads)
    head_dim = hidden // num_heads if num_heads else hidden
    kv_dim = num_kv_heads * head_dim

    per_layer = (
        (hidden + hidden)
        + (hidden + kv_dim)
        + (hidden + kv_dim)
        + (hidden + hidden)
        + (hidden + intermediate)
        + (hidden + intermediate)
        + (intermediate + hidden)
    )
    return int(per_layer * info["num_hidden_layers"])


def _estimate_vram_sft(
    info: dict[str, Any], tokens_per_gpu: int, num_gpus: int
) -> tuple[int, int, int]:
    n = info["num_params"]
    layers = info["num_hidden_layers"]
    hidden = info["hidden_size"]
    vocab = info["vocab_size"]

    model_mem = n * FP32 / num_gpus
    grad_mem = n * FP32 / num_gpus
    opt_mem = n * FP32 * ADAMW_STATES / num_gpus
    act_mem = tokens_per_gpu * FP32 * layers * hidden
    out_mem = tokens_per_gpu * FP32 * vocab * (8 / 3)

    subtotal = model_mem + grad_mem + opt_mem + act_mem + out_mem
    return int(subtotal), int(subtotal * 1.1), int(subtotal * 1.3)


def _estimate_vram_lora(
    info: dict[str, Any], tokens_per_gpu: int, num_gpus: int, lora_r: int
) -> tuple[int, int, int]:
    n = info["num_params"]
    layers = info["num_hidden_layers"]
    hidden = info["hidden_size"]
    vocab = info["vocab_size"]
    wst = _calc_weight_size_total(info)
    ab_params = wst * lora_r

    model_mem = (n * FP16 + ab_params * FP32) / num_gpus
    grad_mem = FP16 * ab_params / num_gpus
    opt_mem = FP16 * ab_params * ADAMW_STATES / num_gpus
    act_mem = tokens_per_gpu * FP16 * layers * hidden
    out_mem = tokens_per_gpu * FP16 * vocab * 2

    subtotal = model_mem + max(grad_mem, out_mem) + opt_mem + act_mem
    return int(subtotal * 0.95), int(subtotal * 1.1), int(subtotal * 1.2)


def _estimate_vram_qlora(
    info: dict[str, Any], tokens_per_gpu: int, num_gpus: int, lora_r: int
) -> tuple[int, int, int]:
    n = info["num_params"]
    layers = info["num_hidden_layers"]
    hidden = info["hidden_size"]
    vocab = info["vocab_size"]
    wst = _calc_weight_size_total(info)
    ab_params = wst * lora_r

    model_mem = (n * FP4 + ab_params * FP32) / num_gpus
    grad_mem = FP16 * ab_params / num_gpus
    opt_mem = FP16 * ab_params * ADAMW_STATES / num_gpus
    act_mem = tokens_per_gpu * FP16 * layers * hidden
    out_mem = tokens_per_gpu * FP16 * vocab * 2

    subtotal = model_mem + max(grad_mem, out_mem) + opt_mem + act_mem
    offload = n * FP8
    effective = max(subtotal, offload / num_gpus)
    return int(effective * 0.95), int(effective * 1.1), int(effective * 1.3)


def _estimate_vram_osft(
    info: dict[str, Any], tokens_per_gpu: int, num_gpus: int, unfreeze_ratio: float
) -> tuple[int, int, int]:
    n = info["num_params"]
    layers = info["num_hidden_layers"]
    hidden = info["hidden_size"]
    vocab = info["vocab_size"]
    osft_params = n * FP32

    model_mem = osft_params / num_gpus
    grad_mem = n * FP32 * unfreeze_ratio / num_gpus
    opt_mem = n * FP32 * ADAMW_STATES / num_gpus
    act_mem = tokens_per_gpu * FP32 * layers * hidden * unfreeze_ratio
    out_mem = tokens_per_gpu * FP32 * vocab * (7 / 3)

    subtotal = model_mem + grad_mem + opt_mem + act_mem + out_mem
    return int(subtotal), int(subtotal * 1.1), int(subtotal * 1.3)


class TrainingEstimateRequest(BaseModel):
    model_size: str = Field(
        ...,
        description="Model size: '0.5B', '1B', '3B', '8B', '13B', '32B', '70B', or '120B'",
    )
    method: str = Field("lora", description="Training method: sft, lora, qlora, osft")
    num_gpus: int = Field(1, description="Number of GPUs")
    batch_size: int | None = Field(None, description="Batch size (provide with max_seq_len)")
    max_seq_len: int | None = Field(None, description="Max sequence length")
    max_tokens_per_gpu: int | None = Field(
        None, description="Max tokens per GPU (alternative to batch_size + max_seq_len)"
    )
    lora_r: int = Field(32, description="LoRA rank (only for lora/qlora)")
    unfreeze_rank_ratio: float = Field(0.25, description="Unfreeze ratio (only for osft)")


class TrainingEstimateResponse(BaseModel):
    model_size: str
    method: str
    num_gpus: int
    vram_per_gpu_gb: dict[str, float]
    total_vram_gb: dict[str, float]


@router.post(
    "/training/estimate",
    response_model=TrainingEstimateResponse,
    operation_id="estimate_training_resources",
    summary="Estimate GPU memory requirements for training a model by size.",
)
async def estimate_training_resources(
    body: TrainingEstimateRequest,
) -> TrainingEstimateResponse:
    info = _resolve_model_size(body.model_size)

    if body.max_tokens_per_gpu:
        tokens_per_gpu = body.max_tokens_per_gpu
    elif body.batch_size and body.max_seq_len:
        tokens_per_gpu = body.batch_size * body.max_seq_len // body.num_gpus
    else:
        tokens_per_gpu = 4096

    method = body.method.lower()
    if method == "sft":
        low, mid, high = _estimate_vram_sft(info, tokens_per_gpu, body.num_gpus)
    elif method in ("lora", "lora_sft"):
        low, mid, high = _estimate_vram_lora(info, tokens_per_gpu, body.num_gpus, body.lora_r)
    elif method in ("qlora", "qlora_sft"):
        low, mid, high = _estimate_vram_qlora(info, tokens_per_gpu, body.num_gpus, body.lora_r)
    elif method == "osft":
        low, mid, high = _estimate_vram_osft(
            info, tokens_per_gpu, body.num_gpus, body.unfreeze_rank_ratio
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown training method: {body.method}")

    to_gb = 1024**3
    low_gb = round(low / to_gb, 1)
    mid_gb = round(mid / to_gb, 1)
    high_gb = round(high / to_gb, 1)

    return TrainingEstimateResponse(
        model_size=info["label"],
        method=body.method,
        num_gpus=body.num_gpus,
        vram_per_gpu_gb={"low": low_gb, "expected": mid_gb, "high": high_gb},
        total_vram_gb={
            "low": round(low_gb * body.num_gpus, 1),
            "expected": round(mid_gb * body.num_gpus, 1),
            "high": round(high_gb * body.num_gpus, 1),
        },
    )
