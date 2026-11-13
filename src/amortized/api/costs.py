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
from amortized.core.training_estimate import TrainingEstimateError, run_training_estimates

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
# The estimate comes from training-hub's memory estimator, run as a short job in the
# training image (see core/training_estimate.py and jobs/assets/estimate_vram.py).
# ---------------------------------------------------------------------------

# training.py defaults when the job config leaves these unset (micro_batch_size 2 x 4).
_DEFAULT_EFFECTIVE_BATCH_SIZE = 8
_DEFAULT_MAX_TOKENS_PER_GPU = 4096
_DEFAULT_MAX_SEQ_LEN = 2048

_GPU_COUNTS = (1, 2, 4, 8)

_METHODS = {
    "sft": "sft",
    "osft": "osft",
    "lora": "lora",
    "lora_sft": "lora",
    "qlora": "qlora",
    "qlora_sft": "qlora",
}


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
    enable_model_splitting: bool = Field(
        True,
        description="lora/qlora only: same value as the job config (default true). If false, "
        "the job runs on one GPU, so every GPU count needs the single-GPU VRAM",
    )
    unfreeze_rank_ratio: float = Field(
        0.2, ge=0.0, le=1.0, description="OSFT unfreeze ratio (osft only)"
    )


class TrainingEstimateResponse(BaseModel):
    model: str
    model_size: str = Field(..., description="Parameter count, e.g. '4.7B'")
    arch_source: str = Field(
        ..., description="Where the estimate came from, e.g. 'training-hub 0.12.0'"
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
        "for 1/2/4/8 GPUs, to suggest nproc_per_node. To compare several models or methods, "
        "use estimate_training_resources_batch (one call for the whole card)."
    ),
)
async def estimate_training_resources(
    body: TrainingEstimateRequest,
) -> TrainingEstimateResponse:
    if body.method.lower() not in _METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown training method: {body.method}")
    batch = await estimate_training_resources_batch(TrainingEstimateBatchRequest(requests=[body]))
    (item,) = batch.estimates
    if item.estimate is None:
        raise HTTPException(status_code=502, detail=item.error)
    return item.estimate


class TrainingEstimateBatchRequest(BaseModel):
    requests: list[TrainingEstimateRequest] = Field(..., min_length=1, max_length=32)


class TrainingEstimateBatchItem(BaseModel):
    estimate: TrainingEstimateResponse | None = None
    error: str | None = Field(None, description="Why this entry could not be estimated")


class TrainingEstimateBatchResponse(BaseModel):
    estimates: list[TrainingEstimateBatchItem] = Field(..., description="In request order")


@router.post(
    "/training/estimate/batch",
    response_model=TrainingEstimateBatchResponse,
    operation_id="estimate_training_resources_batch",
    summary=(
        "Estimate several training jobs in one call (e.g. every candidate model, or every "
        "method for the chosen model) — same fields per entry as estimate_training_resources. "
        "A failed entry reports an error without failing the others."
    ),
)
async def estimate_training_resources_batch(
    body: TrainingEstimateBatchRequest,
) -> TrainingEstimateBatchResponse:
    items = [TrainingEstimateBatchItem() for _ in body.requests]
    to_run: dict[int, dict[str, Any]] = {}
    for i, r in enumerate(body.requests):
        if r.method.lower() not in _METHODS:
            items[i].error = f"Unknown training method: {r.method}"
        else:
            to_run[i] = _build_request(r)
    if to_run:
        try:
            results = await run_training_estimates(list(to_run.values()))
        except TrainingEstimateError as exc:
            results = [{"error": str(exc)}] * len(to_run)
        for i, result in zip(to_run, results, strict=True):
            if "error" in result:
                name = body.requests[i].model_name_or_path
                items[i].error = f"Could not estimate {name!r} with training-hub: {result['error']}"
            else:
                items[i].estimate = _to_response(body.requests[i], result)
    return TrainingEstimateBatchResponse(estimates=items)


def _build_request(body: TrainingEstimateRequest) -> dict[str, Any]:
    method = _METHODS[body.method.lower()]

    # lora/qlora run one process that takes effective_batch_size // gradient_accumulation_steps
    # samples per step (training-hub lora.py); sft/osft are bounded by max_tokens_per_gpu.
    batch_size = body.effective_batch_size
    if method in ("lora", "qlora"):
        batch_size = max(1, batch_size // body.gradient_accumulation_steps)
    return {
        "model": body.model_name_or_path.strip(),
        "method": method,
        "batch_size": batch_size,
        "max_seq_len": body.max_seq_len or _DEFAULT_MAX_SEQ_LEN,
        "max_tokens_per_gpu": body.max_tokens_per_gpu or _DEFAULT_MAX_TOKENS_PER_GPU,
        "lora_r": body.lora_r,
        "unfreeze_rank_ratio": body.unfreeze_rank_ratio,
        "enable_model_splitting": body.enable_model_splitting,
        "gpu_counts": list(_GPU_COUNTS),
    }


def _to_response(body: TrainingEstimateRequest, result: dict[str, Any]) -> TrainingEstimateResponse:
    to_gb = 1024**3
    low, mid, high = result["single_gpu"]
    return TrainingEstimateResponse(
        model=body.model_name_or_path,
        model_size=f"{result['num_params'] / 1e9:.1f}B",
        arch_source=f"training-hub {result['training_hub_version']}",
        method=body.method,
        vram_gb={
            "low": round(low / to_gb, 1),
            "expected": round(mid / to_gb, 1),
            "high": round(high / to_gb, 1),
        },
        per_gpu_vram_gb_by_gpu_count={n: round(v / to_gb, 1) for n, v in result["per_gpu"].items()},
        gpu_budget=settings.gpu_budget,
    )
