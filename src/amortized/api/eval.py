"""Eval endpoint suggestions — gateway models and model-name defaults."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient

logger = logging.getLogger("amortized.api.eval")

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])


@router.get(
    "/endpoint-suggestions",
    operation_id="get_eval_endpoint_suggestions",
    summary=(
        "Suggest eval endpoints: model-name defaults for a training job"
        " (base from its config, tuned from its registered model) and"
        " gateway models. Eval jobs with a training_job_id or"
        " model_name_or_path serve the model themselves."
    ),
)
async def get_eval_endpoint_suggestions(
    training_job_id: str = Query(
        "", description="Training job ID to derive base/tuned model names from (optional)"
    ),
) -> dict[str, Any]:

    suggestions: dict[str, Any] = {
        "training_job_id": training_job_id,
        "model": "",
        "known_endpoints": [],
    }

    # --- Known serving endpoints: models behind the MLflow AI Gateway ---
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    gateway_url = config_mod.settings.gateway_url
    if tracking_uri and gateway_url:
        try:
            client = MLflowClient(tracking_uri)
            gateway_models = await client.list_gateway_models()
            suggestions["known_endpoints"] = [
                {
                    "name": m.get("name", ""),
                    "provider": m.get("provider", ""),
                    "model_name": m.get("model_name", ""),
                    "base_url": gateway_url,
                    "note": "MLflow AI Gateway",
                }
                for m in gateway_models
            ]
        except Exception:
            logger.warning("Failed to list gateway models for eval suggestions", exc_info=True)

    # --- Model-name defaults from the training job ---
    if not training_job_id:
        return suggestions

    from amortized.db.connection import get_pool
    from amortized.db.repository import Repository

    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        training = await repo.get_job(training_job_id)

    if training is None:
        suggestions["message"] = f"Training job {training_job_id} not found"
        return suggestions

    config = training.get("config", {})
    if isinstance(config, str):
        import json

        try:
            config = json.loads(config)
        except ValueError:
            config = {}

    base_model = str(config.get("model_name_or_path") or config.get("model_id") or "")

    tuned_model = ""
    mlflow_run_id = training.get("mlflow_run_id", "")
    if tracking_uri and mlflow_run_id:
        try:
            client = MLflowClient(tracking_uri)
            run = await client.get_run(mlflow_run_id)
            tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
            tuned_model = tags.get("model_display_name", "")
        except Exception:
            logger.warning(
                "Failed to read model_display_name tag for job %s", training_job_id, exc_info=True
            )
    if not tuned_model:
        # Fallback: the registration name pattern from jobs/training.py on_success
        algorithm = str(config.get("algorithm", "sft"))
        short = base_model.split("/")[-1] if base_model else "model"
        tuned_model = f"{short}-{algorithm}-{training_job_id[:8]}"
    suggestions["model"] = tuned_model

    return suggestions


async def _model_size_gb(training_job_id: str, model_name_or_path: str) -> float | None:
    """Weight-file size of the model to serve, in GB, when computable.

    Tuned models: sum MLflow artifact file sizes under model/hf_format (no
    download). HF ids: the hub API's file metadata. Unknown -> None (the
    caller falls back to free-memory-only reporting).
    """
    if training_job_id:
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository

        async with get_pool().acquire() as conn:
            parent = await Repository(conn).get_job(training_job_id)
        if not parent:
            return None
        run_id = parent.get("mlflow_run_id", "")
        if not run_id:
            return None
        client = MLflowClient(config_mod.settings.mlflow_tracking_uri)
        total = 0

        async def walk(path: str) -> None:
            nonlocal total
            for f in await client.list_artifacts(run_id, path):
                if f.get("is_dir"):
                    await walk(f["path"])
                else:
                    total += int(f.get("file_size") or 0)

        try:
            await walk("model/hf_format")
        except Exception:
            logger.warning("gpu-check artifact listing failed", exc_info=True)
            return None
        return round(total / 1e9, 2) or None

    if model_name_or_path:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(model_name_or_path, files_metadata=True)
            total = 0
            for sibling in info.siblings or []:
                if sibling.rfilename.endswith((".safetensors", ".bin")):
                    total += sibling.size or 0
            return round(total / 1e9, 2) or None
        except Exception:
            logger.warning("gpu-check hub lookup failed", exc_info=True)
            return None
    return None


@router.get(
    "/gpu-check",
    operation_id="check_eval_gpu",
    summary=(
        "Pre-flight GPU memory check for an eval job that serves its own"
        " model (training_job_id or model_name_or_path set). Returns the"
        " model's weight size, the GPU the eval would be assigned, its"
        " free memory, what jobs currently hold each GPU, and whether the"
        " model fits. Call before creating the eval job."
    ),
)
async def check_eval_gpu(
    training_job_id: str = Query(
        "", description="Training job ID of the tuned model to evaluate (optional)"
    ),
    model_name_or_path: str = Query(
        "", description="HF hub id of the model to evaluate (optional)"
    ),
) -> dict[str, Any]:
    from amortized.core.gpu_inventory import describe_gpus, read_inventory
    from amortized.jobs.base import JobBuildError

    described = await describe_gpus(config_mod.settings.compute_namespace)
    size_gb = await _model_size_gb(training_job_id, model_name_or_path)

    # The GPU the eval would pin: reuse what the user's pods already pin
    # (most free first), else the freest unheld GPU — same policy as the
    # job builder's assign_serve_gpu.
    mine = [g for g in described["gpus"] if g["mine"]]
    unheld = [g for g in described["gpus"] if not g["held_by"] and not g["busy"]]
    candidates = sorted(mine or unheld, key=lambda g: -g["memory_free_mb"])
    assigned = candidates[0] if candidates else None

    fits: bool | None = None
    if assigned and size_gb is not None:
        # Same 0.9 margin the job builder's auto gpu-memory-utilization
        # leaves; weights must fit in that slice.
        budget_gb = round(assigned["memory_free_mb"] / 1024 * 0.9, 2)
        fits = size_gb <= budget_gb

    result: dict[str, Any] = {
        "model_size_gb": size_gb,
        "assigned_gpu": assigned,
        "fits": fits,
        "gpus": described["gpus"],
        "updated": described["updated"],
    }
    if assigned is None:
        result["error"] = (
            "no GPU is available to serve the model on — every GPU is busy"
            " or held by another user"
        )
    elif fits is False:
        result["error"] = (
            f"the model needs ~{size_gb} GB for weights but the assigned GPU"
            f" ({assigned['node']} #{assigned['index']}) has only"
            f" ~{assigned['memory_free_mb'] / 1024:.0f} GB free"
        )
    return result
