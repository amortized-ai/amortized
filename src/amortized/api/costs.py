"""Cost estimation endpoints for SDG and training workflows."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger("amortized.api.costs")

router = APIRouter(prefix="/api/v1/costs", tags=["costs"])

# ---------------------------------------------------------------------------
# SDG pricing constants
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "vertex_ai/claude-haiku-4-5-20251001": (0.0008, 0.004),
    "vertex_ai/claude-sonnet-4-20250514": (0.003, 0.015),
    "anthropic/claude-haiku-4-5-20251001": (0.0008, 0.004),
    "anthropic/claude-sonnet-4-20250514": (0.003, 0.015),
    "openai/gpt-4o": (0.0025, 0.010),
    "openai/gpt-4o-mini": (0.00015, 0.0006),
}

MODEL_LABELS: dict[str, str] = {
    "vertex_ai/claude-haiku-4-5-20251001": "Claude Haiku",
    "vertex_ai/claude-haiku-4-5@20251001": "Claude Haiku",
    "vertex_ai/claude-sonnet-4-20250514": "Claude Sonnet",
    "vertex_ai/claude-sonnet-4@20250514": "Claude Sonnet",
    "anthropic/claude-haiku-4-5-20251001": "Claude Haiku",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-4o-mini": "GPT-4o Mini",
}


def _resolve_model_label(model: str) -> str:
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    parts = model.rsplit("/", 1)
    if len(parts) == 2:
        name = parts[1].replace("-", " ").replace("@", " ").title()
        return name
    return model


INPUT_TOKENS_PER_SAMPLE = 500
OUTPUT_TOKENS_PER_SAMPLE = 300
FRONTIER_API_COST_PER_SAMPLE = 0.012
FRONTIER_TRAINING_COST_PER_SAMPLE = 0.025
MANUAL_EVAL_COST_PER_SAMPLE = 2.00

EVAL_INPUT_TOKENS_PER_SAMPLE = 800
EVAL_OUTPUT_TOKENS_PER_SAMPLE = 200

# ---------------------------------------------------------------------------
# Training cost constants
# ---------------------------------------------------------------------------

# GPU pricing (EC2 capacity blocks / on-demand)
GPU_PRICING: dict[str, float] = {
    "T4": 0.35,
    "A10G": 1.10,
    "L4": 0.81,
    "A100": 3.50,
    "H100": 4.72,
}

# Model metadata for memory estimation (from training_hub profiling)
TRAINING_MODELS: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {
        "label": "Qwen3 0.6B",
        "description": "Ultra-lightweight, fastest inference, great for prototyping",
        "num_params": 600_000_000,
        "hidden_size": 1024,
        "num_layers": 28,
        "vocab_size": 151936,
        "lora_target_dims": 57344,
        "tokens_per_second": 8500,
    },
    "qwen2.5-1.5b": {
        "label": "Qwen 2.5 1.5B",
        "description": "Small but capable, good balance of speed and quality",
        "num_params": 1_500_000_000,
        "hidden_size": 1536,
        "num_layers": 28,
        "vocab_size": 151936,
        "lora_target_dims": 86016,
        "tokens_per_second": 5000,
    },
    "qwen3-4b": {
        "label": "Qwen3 4B",
        "description": "Larger model, better accuracy, still efficient",
        "num_params": 4_000_000_000,
        "hidden_size": 2560,
        "num_layers": 36,
        "vocab_size": 151936,
        "lora_target_dims": 184320,
        "tokens_per_second": 3000,
    },
    "llama-3.1-8b": {
        "label": "Llama 3.1 8B",
        "description": "Most capable, highest quality, requires more resources",
        "num_params": 8_000_000_000,
        "hidden_size": 4096,
        "num_layers": 32,
        "vocab_size": 128256,
        "lora_target_dims": 262144,
        "tokens_per_second": 1800,
    },
}


def _estimate_memory_gb(
    model_id: str, method: str, batch_size: int = 8, max_seq_len: int = 2048, lora_r: int = 16
) -> float:
    """Estimate peak GPU VRAM in GB using training_hub profiling formulas."""
    info = TRAINING_MODELS.get(model_id, TRAINING_MODELS["qwen3-0.6b"])
    num_params: int = info["num_params"]
    hidden_size: int = info["hidden_size"]
    num_layers: int = info["num_layers"]
    vocab_size: int = info["vocab_size"]
    lora_dims: int = info["lora_target_dims"]
    tokens = batch_size * max_seq_len

    if method == "full_sft":
        model_mem = num_params * 4
        grad_mem = num_params * 4
        opt_mem = num_params * 4 * 2
        act_mem = tokens * 4 * num_layers * hidden_size
        out_mem = tokens * 4 * vocab_size * (8 / 3)
        subtotal = model_mem + grad_mem + opt_mem + act_mem + out_mem
        return (subtotal * 1.1) / (1024**3)

    if method in ("lora_sft", "lora"):
        ab_params = lora_dims * lora_r
        model_mem = num_params * 2 + ab_params * 4
        grad_mem = ab_params * 2
        opt_mem = ab_params * 2 * 2
        act_mem = tokens * 2 * num_layers * hidden_size
        out_mem = tokens * 2 * vocab_size * 2
        peak = model_mem + max(grad_mem, out_mem) + opt_mem + act_mem
        return (peak * 1.1) / (1024**3)

    if method in ("qlora_sft", "qlora"):
        ab_params = lora_dims * lora_r
        model_mem = num_params * 0.5 + ab_params * 4
        grad_mem = ab_params * 2
        opt_mem = ab_params * 2 * 2
        act_mem = tokens * 2 * num_layers * hidden_size
        out_mem = tokens * 2 * vocab_size * 2
        peak = model_mem + max(grad_mem, out_mem) + opt_mem + act_mem
        offload = num_params * 1
        subtotal = max(peak, offload)
        return (subtotal * 1.1) / (1024**3)

    return _estimate_memory_gb(model_id, "lora_sft", batch_size, max_seq_len, lora_r)


def _pick_gpu(vram_gb: float) -> str:
    if vram_gb <= 16:
        return "T4"
    if vram_gb <= 24:
        return "A10G"
    if vram_gb <= 40:
        return "A100"
    return "H100"


def _estimate_training_minutes(
    model_id: str, num_samples: int, num_epochs: int, max_seq_len: int = 2048
) -> float:
    info = TRAINING_MODELS.get(model_id, TRAINING_MODELS["qwen3-0.6b"])
    tokens_per_sec: int = info["tokens_per_second"]
    total_tokens = num_samples * max_seq_len * num_epochs
    seconds = total_tokens / tokens_per_sec
    return seconds / 60


# ---------------------------------------------------------------------------
# OpenRouter live pricing (cached)
# ---------------------------------------------------------------------------


async def _fetch_gateway_models() -> list[tuple[str, str, str]]:
    """Fetch available models from MLflow AI Gateway, returns [(model_id, label, desc)]."""
    import amortized.config as config_mod
    from amortized.core.mlflow_client import MLflowClient

    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri:
        return []
    try:
        client = MLflowClient(tracking_uri)
        models = await client.list_gateway_models()
        result = []
        for m in models:
            provider = m["provider"]
            model_name = m["model_name"]
            if provider and model_name:
                model_id = f"{provider}/{model_name}"
                label = _resolve_model_label(model_id)
                result.append((model_id, label, f"{provider} · {model_name}"))
        return result
    except Exception:
        logger.debug("Could not fetch gateway models, using defaults")
        return []


_openrouter_cache: dict[str, tuple[float, float]] | None = None
_openrouter_cache_time: float = 0
_OPENROUTER_CACHE_TTL = 3600
_openrouter_lock = asyncio.Lock()


async def _fetch_openrouter_pricing() -> dict[str, tuple[float, float]]:
    global _openrouter_cache, _openrouter_cache_time
    now = datetime.now(UTC).timestamp()
    if _openrouter_cache and now - _openrouter_cache_time < _OPENROUTER_CACHE_TTL:
        return _openrouter_cache
    async with _openrouter_lock:
        now = datetime.now(UTC).timestamp()
        if _openrouter_cache and now - _openrouter_cache_time < _OPENROUTER_CACHE_TTL:
            return _openrouter_cache
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://openrouter.ai/api/v1/models")
                resp.raise_for_status()
                data = resp.json()
            pricing: dict[str, tuple[float, float]] = {}
            for model in data.get("data", []):
                mid = model.get("id", "")
                p = model.get("pricing", {})
                prompt_price = float(p.get("prompt", "0"))
                completion_price = float(p.get("completion", "0"))
                if prompt_price > 0 or completion_price > 0:
                    pricing[mid] = (prompt_price * 1000, completion_price * 1000)
            _openrouter_cache = pricing
            _openrouter_cache_time = now
            return pricing
        except Exception:
            logger.warning("Failed to fetch OpenRouter pricing, using hardcoded fallback")
            return dict(MODEL_PRICING)


def _get_pricing(model: str, live: dict[str, tuple[float, float]]) -> tuple[float, float]:
    if model in live:
        return live[model]
    return MODEL_PRICING.get(model, (0.001, 0.005))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class EstimateSdgCostRequest(BaseModel):
    num_samples: int = Field(100, description="Number of training samples to generate")
    model: str = Field("openai/gpt-4o-mini", description="Teacher model ID in LiteLLM format")


class EstimateSdgCostResponse(BaseModel):
    model: str
    model_label: str
    num_samples: int
    tokens: dict[str, int]
    cost: dict[str, float]
    comparison: dict[str, float]


class SdgModelSpec(BaseModel):
    model_id: str
    label: str = ""
    description: str = ""


class CompareSdgModelsRequest(BaseModel):
    num_samples: int = Field(100, description="Number of training samples to generate")
    models: list[SdgModelSpec] | None = Field(
        None, description="Available models to compare (from AI Gateway). Uses defaults if omitted."
    )


class SdgModelComparison(BaseModel):
    model_id: str
    label: str
    description: str
    total_cost: float
    per_sample_cost: float


class CompareSdgModelsResponse(BaseModel):
    num_samples: int
    models: list[SdgModelComparison]


class EstimateTrainingCostRequest(BaseModel):
    num_samples: int = Field(100, description="Number of training samples")
    num_epochs: int = Field(3, description="Number of training epochs")


class TrainingModelEstimate(BaseModel):
    model_id: str
    label: str
    description: str
    gpu_type: str
    vram_gb: int
    estimated_time_minutes: float
    estimated_cost: float
    cost_per_gpu_hour: float


class EstimateTrainingCostResponse(BaseModel):
    num_samples: int
    num_epochs: int
    models: list[TrainingModelEstimate]


class EstimateTrainingMethodCostRequest(BaseModel):
    model_id: str = Field("qwen3-0.6b", description="Student model ID")
    num_samples: int = Field(100, description="Number of training samples")
    num_epochs: int = Field(3, description="Number of training epochs")


class TrainingMethodEstimate(BaseModel):
    method: str
    label: str
    description: str
    gpu_type: str
    vram_gb: float
    estimated_time_minutes: float
    estimated_cost: float
    relative_time: str
    recommended: bool


class TrainingMethodComparison(BaseModel):
    automated_label: str
    automated_cost: float
    manual_training_total: float
    savings_amount: float
    savings_percent: float


class EstimateTrainingMethodCostResponse(BaseModel):
    model_id: str
    model_label: str
    num_samples: int
    num_epochs: int
    methods: list[TrainingMethodEstimate]
    comparison: TrainingMethodComparison


class EvalModelInput(BaseModel):
    model_id: str
    label: str = ""
    description: str = ""


class EstimateEvalCostRequest(BaseModel):
    num_samples: int = Field(100, description="Number of evaluation samples")
    judge_model: str = Field("openai/gpt-4o-mini", description="Judge model ID")
    models: list[EvalModelInput] | None = Field(
        None, description="Available judge models from gateway"
    )


class EvalJudgeOption(BaseModel):
    model_id: str
    label: str
    description: str
    total_cost: float
    per_sample_cost: float
    recommended: bool


class EvalManualComparison(BaseModel):
    manual_evaluation_total: float
    savings_amount: float
    savings_percent: float


class EstimateEvalCostResponse(BaseModel):
    judge_model: str
    judge_model_label: str
    num_samples: int
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    cost_per_sample: float
    comparison: list[EvalJudgeOption]
    manual_comparison: EvalManualComparison


# ---------------------------------------------------------------------------
# Model pricing lookup (local openrouter_costs.json)
# ---------------------------------------------------------------------------

_COSTS_FILE = Path(__file__).resolve().parent.parent.parent.parent / "openrouter_costs.json"
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
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/sdg",
    response_model=EstimateSdgCostResponse,
    operation_id="estimate_sdg_cost",
    summary="Estimate the cost of generating synthetic training data with a teacher model.",
)
async def estimate_sdg_cost(body: EstimateSdgCostRequest) -> EstimateSdgCostResponse:
    live_pricing = await _fetch_openrouter_pricing()
    input_cost_per_1k, output_cost_per_1k = _get_pricing(body.model, live_pricing)

    total_input_tokens = body.num_samples * INPUT_TOKENS_PER_SAMPLE
    total_output_tokens = body.num_samples * OUTPUT_TOKENS_PER_SAMPLE

    input_cost = (total_input_tokens / 1000) * input_cost_per_1k
    output_cost = (total_output_tokens / 1000) * output_cost_per_1k
    sdg_total = input_cost + output_cost

    manual_total = body.num_samples * FRONTIER_API_COST_PER_SAMPLE
    savings_amount = manual_total - sdg_total
    savings_pct = (savings_amount / manual_total * 100) if manual_total > 0 else 0

    return EstimateSdgCostResponse(
        model=body.model,
        model_label=_resolve_model_label(body.model),
        num_samples=body.num_samples,
        tokens={
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_input_tokens + total_output_tokens,
        },
        cost={
            "input": round(input_cost, 4),
            "output": round(output_cost, 4),
            "total": round(sdg_total, 4),
        },
        comparison={
            "manual_labeling_total": round(manual_total, 2),
            "savings_amount": round(savings_amount, 2),
            "savings_percent": round(savings_pct, 1),
        },
    )


@router.post(
    "/sdg/compare",
    response_model=CompareSdgModelsResponse,
    operation_id="compare_sdg_models",
    summary="Compare cost estimates across teacher models for synthetic data generation.",
)
async def compare_sdg_models(body: CompareSdgModelsRequest) -> CompareSdgModelsResponse:
    live_pricing = await _fetch_openrouter_pricing()

    if body.models:
        sdg_models = [
            (m.model_id, m.label or _resolve_model_label(m.model_id), m.description)
            for m in body.models
        ]
    else:
        sdg_models = [
            ("openai/gpt-4o-mini", "GPT-4o Mini", "Fast and affordable"),
        ]

    models = []
    for model_id, label, desc in sdg_models:
        inp_1k, out_1k = _get_pricing(model_id, live_pricing)
        total_input = body.num_samples * INPUT_TOKENS_PER_SAMPLE
        total_output = body.num_samples * OUTPUT_TOKENS_PER_SAMPLE
        cost = (total_input / 1000) * inp_1k + (total_output / 1000) * out_1k
        per_sample = cost / body.num_samples if body.num_samples > 0 else 0
        models.append(
            SdgModelComparison(
                model_id=model_id,
                label=label,
                description=desc,
                total_cost=round(cost, 4),
                per_sample_cost=round(per_sample, 6),
            )
        )

    return CompareSdgModelsResponse(
        num_samples=body.num_samples,
        models=sorted(models, key=lambda m: m.total_cost),
    )


@router.post(
    "/training",
    response_model=EstimateTrainingCostResponse,
    operation_id="estimate_training_cost",
    summary="Estimate training cost and time for each student model size.",
)
async def estimate_training_cost(
    body: EstimateTrainingCostRequest,
) -> EstimateTrainingCostResponse:
    models = []
    for model_id, info in TRAINING_MODELS.items():
        vram_gb = _estimate_memory_gb(model_id, "lora_sft")
        gpu_type = _pick_gpu(vram_gb)
        cost_per_hour = GPU_PRICING.get(gpu_type, 1.10)
        est_minutes = _estimate_training_minutes(model_id, body.num_samples, body.num_epochs)
        est_cost = (est_minutes / 60) * cost_per_hour

        models.append(
            TrainingModelEstimate(
                model_id=model_id,
                label=info["label"],
                description=info["description"],
                gpu_type=gpu_type,
                vram_gb=round(vram_gb),
                estimated_time_minutes=round(est_minutes, 1),
                estimated_cost=round(est_cost, 4),
                cost_per_gpu_hour=cost_per_hour,
            )
        )

    return EstimateTrainingCostResponse(
        num_samples=body.num_samples,
        num_epochs=body.num_epochs,
        models=models,
    )


@router.post(
    "/training/method",
    response_model=EstimateTrainingMethodCostResponse,
    operation_id="estimate_training_method_cost",
    summary="Estimate cost for a specific model across training methods (LoRA, QLoRA, Full SFT).",
)
async def estimate_training_method_cost(
    body: EstimateTrainingMethodCostRequest,
) -> EstimateTrainingMethodCostResponse:
    model_info = TRAINING_MODELS.get(body.model_id, TRAINING_MODELS["qwen3-0.6b"])
    base_minutes = _estimate_training_minutes(body.model_id, body.num_samples, body.num_epochs)

    method_configs = [
        ("lora_sft", "LoRA SFT", "Trains adapter weights only — fastest and cheapest", 1.0, True),
        ("qlora_sft", "QLoRA SFT", "4-bit quantized base + LoRA — lower VRAM", 1.2, False),
        ("full_sft", "Full SFT", "Updates all model weights — highest quality", 3.5, False),
    ]

    methods = []
    for method, label, desc, time_mult, recommended in method_configs:
        vram_gb = _estimate_memory_gb(body.model_id, method)
        gpu_type = _pick_gpu(vram_gb)
        cost_per_hour = GPU_PRICING.get(gpu_type, 1.10)
        est_minutes = base_minutes * time_mult
        est_cost = (est_minutes / 60) * cost_per_hour

        methods.append(
            TrainingMethodEstimate(
                method=method,
                label=label,
                description=desc,
                gpu_type=gpu_type,
                vram_gb=round(vram_gb, 1),
                estimated_time_minutes=round(est_minutes, 1),
                estimated_cost=round(est_cost, 2),
                relative_time="1x" if time_mult == 1.0 else f"~{time_mult}x",
                recommended=recommended,
            )
        )

    recommended_method = next((m for m in methods if m.recommended), methods[0])
    frontier_total = body.num_samples * FRONTIER_TRAINING_COST_PER_SAMPLE
    savings_amount = frontier_total - recommended_method.estimated_cost
    savings_pct = (savings_amount / frontier_total * 100) if frontier_total > 0 else 0

    return EstimateTrainingMethodCostResponse(
        model_id=body.model_id,
        model_label=model_info["label"],
        num_samples=body.num_samples,
        num_epochs=body.num_epochs,
        methods=methods,
        comparison=TrainingMethodComparison(
            automated_label=f"{recommended_method.label} Training",
            automated_cost=recommended_method.estimated_cost,
            manual_training_total=round(frontier_total, 2),
            savings_amount=round(savings_amount, 2),
            savings_percent=round(savings_pct, 1),
        ),
    )


@router.post(
    "/eval",
    response_model=EstimateEvalCostResponse,
    operation_id="estimate_eval_cost",
    summary="Estimate evaluation cost using an LLM judge model.",
)
async def estimate_eval_cost(
    body: EstimateEvalCostRequest,
) -> EstimateEvalCostResponse:
    live_pricing = await _fetch_openrouter_pricing()
    input_cost_per_1k, output_cost_per_1k = _get_pricing(body.judge_model, live_pricing)

    total_input_tokens = body.num_samples * EVAL_INPUT_TOKENS_PER_SAMPLE
    total_output_tokens = body.num_samples * EVAL_OUTPUT_TOKENS_PER_SAMPLE

    input_cost = (total_input_tokens / 1000) * input_cost_per_1k
    output_cost = (total_output_tokens / 1000) * output_cost_per_1k
    eval_total = input_cost + output_cost
    cost_per_sample = eval_total / body.num_samples if body.num_samples > 0 else 0

    judge_options: list[tuple[str, str, str]] = []
    if body.models:
        for m in body.models:
            judge_options.append(
                (
                    m.model_id,
                    m.label or _resolve_model_label(m.model_id),
                    m.description or "",
                )
            )
    if not judge_options:
        judge_options = [("openai/gpt-4o-mini", "GPT-4o Mini", "Default judge model")]
    comparison = []
    for mid, label, desc in judge_options:
        inp_1k, out_1k = _get_pricing(mid, live_pricing)
        t_inp = body.num_samples * EVAL_INPUT_TOKENS_PER_SAMPLE
        t_out = body.num_samples * EVAL_OUTPUT_TOKENS_PER_SAMPLE
        total = (t_inp / 1000) * inp_1k + (t_out / 1000) * out_1k
        comparison.append(
            EvalJudgeOption(
                model_id=mid,
                label=label,
                description=desc,
                total_cost=round(total, 2),
                per_sample_cost=round(total / body.num_samples if body.num_samples > 0 else 0, 2),
                recommended=False,
            )
        )

    sorted_comparison = sorted(comparison, key=lambda m: m.total_cost)
    if sorted_comparison:
        sorted_comparison[0].recommended = True

    review_total = body.num_samples * MANUAL_EVAL_COST_PER_SAMPLE
    eval_savings = review_total - eval_total
    eval_savings_pct = (eval_savings / review_total * 100) if review_total > 0 else 0

    return EstimateEvalCostResponse(
        judge_model=body.judge_model,
        judge_model_label=_resolve_model_label(body.judge_model),
        num_samples=body.num_samples,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        input_cost=round(input_cost, 2),
        output_cost=round(output_cost, 2),
        total_cost=round(eval_total, 2),
        cost_per_sample=round(cost_per_sample, 2),
        comparison=sorted_comparison,
        manual_comparison=EvalManualComparison(
            manual_evaluation_total=round(review_total, 2),
            savings_amount=round(eval_savings, 2),
            savings_percent=round(eval_savings_pct, 1),
        ),
    )


# ---------------------------------------------------------------------------
# Training resource estimation (from HuggingFace model metadata)
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
