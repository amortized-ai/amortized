"""Cost estimation endpoints for SDG and training workflows."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter
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
    "vertex_ai/claude-sonnet-4-20250514": "Claude Sonnet",
    "anthropic/claude-haiku-4-5-20251001": "Claude Haiku",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-4o-mini": "GPT-4o Mini",
}

INPUT_TOKENS_PER_SAMPLE = 500
OUTPUT_TOKENS_PER_SAMPLE = 300
MANUAL_LABELING_COST_PER_SAMPLE = 3.50
MANUAL_TRAINING_COST_PER_SAMPLE = 6.00
MANUAL_EVAL_COST_PER_SAMPLE = 5.00

EVAL_INPUT_TOKENS_PER_SAMPLE = 800
EVAL_OUTPUT_TOKENS_PER_SAMPLE = 200

# ---------------------------------------------------------------------------
# Training cost constants
# ---------------------------------------------------------------------------

TRAINING_MODELS: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {
        "label": "Qwen3 0.6B",
        "description": "Ultra-lightweight, fastest inference, great for prototyping",
        "gpu_type": "T4",
        "cost_per_gpu_hour": 0.35,
        "base_minutes_per_100_samples": 8,
        "vram_gb": 4,
    },
    "qwen2.5-1.5b": {
        "label": "Qwen 2.5 1.5B",
        "description": "Small but capable, good balance of speed and quality",
        "gpu_type": "T4",
        "cost_per_gpu_hour": 0.35,
        "base_minutes_per_100_samples": 15,
        "vram_gb": 6,
    },
    "qwen3-4b": {
        "label": "Qwen3 4B",
        "description": "Larger model, better accuracy, still efficient",
        "gpu_type": "A10G",
        "cost_per_gpu_hour": 1.10,
        "base_minutes_per_100_samples": 25,
        "vram_gb": 12,
    },
    "llama-3.1-8b": {
        "label": "Llama 3.1 8B",
        "description": "Most capable, highest quality, requires more resources",
        "gpu_type": "A100",
        "cost_per_gpu_hour": 3.50,
        "base_minutes_per_100_samples": 35,
        "vram_gb": 24,
    },
}

# ---------------------------------------------------------------------------
# OpenRouter live pricing (cached)
# ---------------------------------------------------------------------------

_openrouter_cache: dict[str, tuple[float, float]] | None = None
_openrouter_cache_time: float = 0
_OPENROUTER_CACHE_TTL = 3600


async def _fetch_openrouter_pricing() -> dict[str, tuple[float, float]]:
    global _openrouter_cache, _openrouter_cache_time
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


class CompareSdgModelsRequest(BaseModel):
    num_samples: int = Field(100, description="Number of training samples to generate")


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


class EstimateEvalCostRequest(BaseModel):
    num_samples: int = Field(100, description="Number of evaluation samples")
    judge_model: str = Field("openai/gpt-4o-mini", description="Judge model ID")


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

    manual_total = body.num_samples * MANUAL_LABELING_COST_PER_SAMPLE
    savings_amount = manual_total - sdg_total
    savings_pct = (savings_amount / manual_total * 100) if manual_total > 0 else 0

    return EstimateSdgCostResponse(
        model=body.model,
        model_label=MODEL_LABELS.get(body.model, body.model),
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

    sdg_models = [
        ("vertex_ai/claude-haiku-4-5-20251001", "Claude Haiku", "Fast and affordable"),
        ("vertex_ai/claude-sonnet-4-20250514", "Claude Sonnet", "Higher quality output"),
        ("openai/gpt-4o", "GPT-4o", "Strong reasoning ability"),
    ]

    models = []
    for model_id, label, desc in sdg_models:
        inp_1k, out_1k = _get_pricing(model_id, live_pricing)
        total_input = body.num_samples * INPUT_TOKENS_PER_SAMPLE
        total_output = body.num_samples * OUTPUT_TOKENS_PER_SAMPLE
        cost = (total_input / 1000) * inp_1k + (total_output / 1000) * out_1k
        per_sample = cost / body.num_samples if body.num_samples > 0 else 0
        models.append(SdgModelComparison(
            model_id=model_id,
            label=label,
            description=desc,
            total_cost=round(cost, 4),
            per_sample_cost=round(per_sample, 6),
        ))

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
        base_minutes: float = info["base_minutes_per_100_samples"]
        estimated_minutes = (body.num_samples / 100) * base_minutes * (body.num_epochs / 3)
        estimated_hours = estimated_minutes / 60
        estimated_cost = estimated_hours * info["cost_per_gpu_hour"]

        models.append(TrainingModelEstimate(
            model_id=model_id,
            label=info["label"],
            description=info["description"],
            gpu_type=info["gpu_type"],
            vram_gb=info["vram_gb"],
            estimated_time_minutes=round(estimated_minutes, 1),
            estimated_cost=round(estimated_cost, 4),
            cost_per_gpu_hour=info["cost_per_gpu_hour"],
        ))

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
    base_minutes: float = model_info["base_minutes_per_100_samples"]
    base_time = (body.num_samples / 100) * base_minutes * (body.num_epochs / 3)
    base_hours = base_time / 60
    base_cost = base_hours * model_info["cost_per_gpu_hour"]

    methods = [
        TrainingMethodEstimate(
            method="lora_sft",
            label="LoRA SFT",
            description="Trains adapter weights only — fastest and cheapest",
            gpu_type=model_info["gpu_type"],
            vram_gb=model_info["vram_gb"],
            estimated_time_minutes=round(base_time, 1),
            estimated_cost=round(base_cost, 2),
            relative_time="1x",
            recommended=True,
        ),
        TrainingMethodEstimate(
            method="qlora_sft",
            label="QLoRA SFT",
            description="4-bit quantized base + LoRA — lower VRAM, slightly slower",
            gpu_type=model_info["gpu_type"],
            vram_gb=round(model_info["vram_gb"] * 0.5, 1),
            estimated_time_minutes=round(base_time * 1.2, 1),
            estimated_cost=round(base_cost * 1.2, 2),
            relative_time="~1.2x",
            recommended=False,
        ),
        TrainingMethodEstimate(
            method="full_sft",
            label="Full SFT",
            description="Updates all model weights — highest quality, most expensive",
            gpu_type="A100" if model_info["vram_gb"] > 8 else "A10G",
            vram_gb=model_info["vram_gb"] * 4,
            estimated_time_minutes=round(base_time * 3.5, 1),
            estimated_cost=round(base_cost * 3.5 * (3.5 / model_info["cost_per_gpu_hour"]), 2),
            relative_time="~3.5x",
            recommended=False,
        ),
    ]

    recommended = next((m for m in methods if m.recommended), methods[0])
    manual_total = body.num_samples * MANUAL_TRAINING_COST_PER_SAMPLE
    savings_amount = manual_total - recommended.estimated_cost
    savings_pct = (savings_amount / manual_total * 100) if manual_total > 0 else 0

    return EstimateTrainingMethodCostResponse(
        model_id=body.model_id,
        model_label=model_info["label"],
        num_samples=body.num_samples,
        num_epochs=body.num_epochs,
        methods=methods,
        comparison=TrainingMethodComparison(
            automated_label=f"{recommended.label} Training",
            automated_cost=recommended.estimated_cost,
            manual_training_total=round(manual_total, 2),
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

    judge_options = [
        ("openai/gpt-4o-mini", "GPT-4o Mini", "Cheapest, good for simple tasks"),
        ("anthropic/claude-haiku-4-5-20251001", "Claude Haiku", "Balanced cost and quality"),
        ("openai/gpt-4o", "GPT-4o", "Higher quality judging"),
        ("anthropic/claude-sonnet-4-20250514", "Claude Sonnet", "Highest quality, most expensive"),
    ]
    comparison = []
    for mid, label, desc in judge_options:
        inp_1k, out_1k = _get_pricing(mid, live_pricing)
        t_inp = body.num_samples * EVAL_INPUT_TOKENS_PER_SAMPLE
        t_out = body.num_samples * EVAL_OUTPUT_TOKENS_PER_SAMPLE
        total = (t_inp / 1000) * inp_1k + (t_out / 1000) * out_1k
        comparison.append(EvalJudgeOption(
            model_id=mid,
            label=label,
            description=desc,
            total_cost=round(total, 2),
            per_sample_cost=round(total / body.num_samples if body.num_samples > 0 else 0, 2),
            recommended=False,
        ))

    sorted_comparison = sorted(comparison, key=lambda m: m.total_cost)
    if sorted_comparison:
        sorted_comparison[0].recommended = True

    manual_eval_total = body.num_samples * MANUAL_EVAL_COST_PER_SAMPLE
    eval_savings = manual_eval_total - eval_total
    eval_savings_pct = (eval_savings / manual_eval_total * 100) if manual_eval_total > 0 else 0

    return EstimateEvalCostResponse(
        judge_model=body.judge_model,
        judge_model_label=MODEL_LABELS.get(body.judge_model, body.judge_model),
        num_samples=body.num_samples,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        input_cost=round(input_cost, 2),
        output_cost=round(output_cost, 2),
        total_cost=round(eval_total, 2),
        cost_per_sample=round(cost_per_sample, 2),
        comparison=sorted_comparison,
        manual_comparison=EvalManualComparison(
            manual_evaluation_total=round(manual_eval_total, 2),
            savings_amount=round(eval_savings, 2),
            savings_percent=round(eval_savings_pct, 1),
        ),
    )
