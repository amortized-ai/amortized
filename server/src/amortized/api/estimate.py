"""Memory estimation endpoint."""

import logging

from fastapi import APIRouter

from amortized.models import MemoryEstimateRequest, MemoryEstimateResponse

logger = logging.getLogger("amortized.api.estimate")

router = APIRouter(prefix="/api/v1/estimate", tags=["estimate"])


def _estimate_vram(req: MemoryEstimateRequest) -> float:
    """Estimate GPU VRAM requirements.

    Uses Training Hub's LoRAEstimator/QLoRAEstimator when available,
    falls back to a rough heuristic.
    """
    try:
        if req.load_in_4bit:
            from training_hub import QLoRAEstimator

            estimator = QLoRAEstimator(
                model_path=req.model_name_or_path,
                lora_r=req.lora_r,
                batch_size=req.batch_size,
                max_seq_len=req.max_length,
            )
        else:
            from training_hub import LoRAEstimator

            estimator = LoRAEstimator(
                model_path=req.model_name_or_path,
                lora_r=req.lora_r,
                batch_size=req.batch_size,
                max_seq_len=req.max_length,
            )
        return float(estimator.estimate())
    except ImportError:
        logger.debug("training_hub not installed, using heuristic estimation")
        return _heuristic_estimate(req)


def _heuristic_estimate(req: MemoryEstimateRequest) -> float:
    """Rough VRAM estimate based on model name heuristics.

    This is a fallback when training_hub is not installed.
    Real estimates should use LoRAEstimator/QLoRAEstimator.
    """
    # Extract approximate parameter count from model path
    model_lower = req.model_name_or_path.lower()
    param_billions = 1.5  # default
    for token in model_lower.replace("-", ".").replace("_", ".").split("/")[-1].split("."):
        if token.endswith("b"):
            try:
                param_billions = float(token[:-1])
                break
            except ValueError:
                continue

    # Base VRAM: ~2 bytes per param for bf16, ~0.5 bytes for 4-bit
    bytes_per_param = 0.5 if req.load_in_4bit else 2.0
    base_gb = param_billions * bytes_per_param

    # LoRA overhead scales with rank
    lora_overhead = req.lora_r * 0.01  # rough

    # Batch/sequence overhead
    batch_overhead = req.batch_size * req.max_length * 2 / (1024**3)

    return round(base_gb + lora_overhead + batch_overhead, 2)


@router.post("", response_model=MemoryEstimateResponse)
async def estimate_memory(req: MemoryEstimateRequest) -> MemoryEstimateResponse:
    """Estimate GPU VRAM requirements for a LoRA training configuration."""
    vram = _estimate_vram(req)
    return MemoryEstimateResponse(
        model_name_or_path=req.model_name_or_path,
        lora_r=req.lora_r,
        batch_size=req.batch_size,
        max_length=req.max_length,
        estimated_vram_gb=vram,
        load_in_4bit=req.load_in_4bit,
    )
