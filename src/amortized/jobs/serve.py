"""Serve job builder — bring up a persistent vLLM inference endpoint.

Serves either a tuned model from a completed training job (downloading the
merged HF export from MLflow) or any model by name/path (HF hub id). The job
stays running until cancelled; a Kubernetes Service exposes the endpoint
in-cluster for eval jobs.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

from amortized.backends import Resources
from amortized.jobs.base import JobBuildError, JobBuildResult

logger = logging.getLogger("amortized.jobs.serve")

IMAGE = "ghcr.io/amortized-ai/training:latest"

DEFAULT_PORT = 8000


async def _resolve_training_model(config: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Resolve a tuned model from a training job's MLflow artifacts.

    Returns (served_model_name, model_path, pre_commands). model_path is the
    in-container directory the final merged checkpoint will be downloaded to.
    """
    from amortized.core.mlflow_client import MLflowClient

    import amortized.config as config_mod

    training_job_id = str(config.get("training_job_id", "")).strip()
    if not training_job_id:
        raise JobBuildError("training_job_id is required when serving a tuned model")

    from amortized.db.connection import get_pool

    async with get_pool().acquire() as conn:
        from amortized.db.repository import Repository

        parent = await Repository(conn).get_job(training_job_id)
    if not parent or parent["type"] != "training":
        raise JobBuildError(f"training_job_id {training_job_id!r} is not a training job")
    if parent["status"] != "succeeded":
        raise JobBuildError(
            f"training job {training_job_id[:8]} has status {parent['status']!r}"
            " — only succeeded training jobs can be served"
        )
    run_id = parent.get("mlflow_run_id", "")
    if not run_id:
        raise JobBuildError(f"training job {training_job_id[:8]} has no MLflow run")

    # Default served name: the training run's display name tag (e.g. mdl-brawny-jay-896)
    served_name = str(config.get("served_model_name", "")).strip()
    if not served_name:
        served_name = str((parent.get("config") or {}).get("model_display_name", "")).strip()
        if not served_name:
            raise JobBuildError(
                "served_model_name is required (training job has no model_display_name)"
            )

    local_dir = "/amortized/work/served_model"
    pre_commands = [
        # Download the merged HF export, then pick the highest-step checkpoint
        f"mlflow artifacts download -r {shlex.quote(run_id)} -a model/hf_format"
        f" -d {shlex.quote(local_dir)}",
        f"SERVE_MODEL_DIR=$(find {shlex.quote(local_dir)}/hf_format -mindepth 1 -maxdepth 1"
        " -type d | sort -V | tail -1)",
    ]
    model_path = "$SERVE_MODEL_DIR"
    return served_name, model_path, pre_commands


def _resolve_named_model(config: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Resolve a model by HF hub id or local path (vLLM downloads if needed)."""
    model_name = str(config.get("model_name_or_path", "")).strip()
    if not model_name:
        raise JobBuildError(
            "serve jobs require either training_job_id (tuned model)"
            " or model_name_or_path (HF model id or local path)"
        )
    served_name = str(config.get("served_model_name", "")).strip() or model_name
    return served_name, model_name, []


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    port = int(config.get("port", DEFAULT_PORT))
    gpus = int(config.get("nproc_per_node", 1))

    if config.get("training_job_id"):
        served_name, model_path, pre_commands = await _resolve_training_model(config)
    else:
        served_name, model_path, pre_commands = _resolve_named_model(config)

    # Built as a single sh -c string so $SERVE_MODEL_DIR set by a pre-command
    # expands (the worker's _wrap_command shlex.quotes plain arg lists).
    serve_cmd = f'vllm serve "{model_path}"'
    serve_cmd += f" --served-model-name {shlex.quote(served_name)}"
    serve_cmd += f" --port {int(port)}"
    max_model_len = config.get("max_model_len")
    if max_model_len:
        serve_cmd += f" --max-model-len {int(max_model_len)}"
    for extra in config.get("vllm_args", []) or []:
        if extra:
            serve_cmd += f" {shlex.quote(str(extra))}"
    if model_path.startswith("$"):
        serve_cmd = f'test -n "{model_path}" && {serve_cmd}'
    cmd = ["sh", "-c", serve_cmd]

    resolved_config = dict(config)
    resolved_config["served_model_name"] = served_name
    resolved_config["port"] = port

    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        pre_commands=pre_commands,
        # Serve jobs run until cancelled — no post commands, no completion
        post_commands=[],
        resources=Resources(gpus=gpus, memory_gb=config.get("memory_gb") or None),
        image=IMAGE,
        ports={port: port},
        resolved_config=resolved_config,
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    # Serve jobs never complete on their own — nothing to do on success.
    return None
