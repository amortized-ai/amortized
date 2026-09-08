"""Eval endpoint suggestions — known serving endpoints and model-name defaults."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient
from amortized.models import JobStatus

logger = logging.getLogger("amortized.api.eval")

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])


def _serve_base_url(job: dict[str, Any]) -> str:
    """In-cluster base URL of a serve job's vLLM endpoint ("" if unknown)."""
    k8s_job_name = job.get("k8s_job_name", "")
    namespace = job.get("k8s_namespace", "") or config_mod.settings.compute_namespace
    port = 8000
    config = job.get("config", {})
    if isinstance(config, str):
        import json

        try:
            config = json.loads(config)
        except ValueError:
            config = {}
    if isinstance(config, dict):
        try:
            port = int(config.get("port", 8000))
        except (TypeError, ValueError):
            port = 8000
    if not k8s_job_name:
        return ""
    return f"http://{k8s_job_name}.{namespace}.svc.cluster.local:{port}/v1"


async def _serve_endpoints() -> list[dict[str, Any]]:
    """Running serve jobs as selectable endpoints, with a live health check."""
    from amortized.db.connection import get_pool
    from amortized.db.repository import Repository

    endpoints: list[dict[str, Any]] = []
    try:
        from amortized.models import JobType

        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            jobs = await repo.list_jobs(status=JobStatus.running, job_type=JobType.serve)
    except Exception:
        logger.warning("Failed to list serve jobs for suggestions", exc_info=True)
        return endpoints

    import httpx

    async with httpx.AsyncClient(timeout=2.0) as client:
        for job in jobs:
            base_url = _serve_base_url(job)
            if not base_url:
                continue
            config = job.get("config", {})
            if isinstance(config, str):
                try:
                    import json

                    config = json.loads(config)
                except ValueError:
                    config = {}
            healthy = False
            try:
                resp = await client.get(base_url.replace("/v1", "") + "/health")
                healthy = resp.status_code == 200
            except Exception:
                pass
            endpoints.append(
                {
                    "job_id": job["id"],
                    "name": str(config.get("served_model_name", "")),
                    "model_name": str(config.get("served_model_name", "")),
                    "base_url": base_url,
                    "healthy": healthy,
                    "source": "serve job"
                    + (" (tuned)" if config.get("training_job_id") else ""),
                    "note": f"Serve job {job['id'][:8]}"
                    + (" — ready" if healthy else " — starting up, not ready yet"),
                }
            )
    return endpoints


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
        "serve_endpoints": [],
    }

    # --- Serve jobs the platform itself is running (vLLM endpoints) ---
    suggestions["serve_endpoints"] = await _serve_endpoints()

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
