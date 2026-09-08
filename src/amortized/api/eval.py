"""Eval endpoint suggestions — known serving endpoints and model-name defaults."""

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
        " (base from its config, tuned from its registered model) and the"
        " currently known serving endpoints (gateway models)."
    ),
)
async def get_eval_endpoint_suggestions(
    training_job_id: str = Query(
        "", description="Training job ID to derive base/tuned model names from (optional)"
    ),
) -> dict[str, Any]:

    suggestions: dict[str, Any] = {
        "training_job_id": training_job_id,
        "base_model": "",
        "tuned_model": "",
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
    suggestions["base_model"] = base_model

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
    suggestions["tuned_model"] = tuned_model

    return suggestions
