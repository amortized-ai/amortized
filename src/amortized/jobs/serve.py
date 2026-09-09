"""Serve job builder — bring up a persistent vLLM inference endpoint.

Serves either a tuned model from a completed training job (downloading the
HF export from MLflow) or any model by name/path (HF hub id). The job stays
running until cancelled; a Kubernetes Service exposes the endpoint
in-cluster for eval jobs.

Training Hub exports of VLM text backbones (e.g. Qwen3.5) carry a bare text
config (model_type "qwen3_5_text", no architectures, weights keyed like a
CausalLM) that stock vLLM cannot serve. Those exports are served through
serve_vllm.py — a generic wrapper that registers vLLM's unregistered
text-backbone architectures and tolerates non-persistent buffer weights —
after patch_model_config.py fills in the architectures key. One model per
job: base and tuned models are served by separate serve jobs.

Serve pods do NOT request nvidia.com/gpu. They pin to a GPU chosen by
core.gpu_inventory.assign_serve_gpu via NVIDIA_VISIBLE_DEVICES=<uuid>, so
several of a user's deployments share one GPU within their quota (the
in-pod check_gpu_memory pre-flight fails fast when free memory runs out).
"""

from __future__ import annotations

import logging
import os
import shlex
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.jobs.base import JobBuildError, JobBuildResult

logger = logging.getLogger("amortized.jobs.serve")

IMAGE = "ghcr.io/amortized-ai/training:latest"

DEFAULT_PORT = 8000

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


async def _training_display_name(parent: dict[str, Any], run_id: str) -> str:
    """Registered model name for a training run: MLflow tag, else the
    registration-name pattern ({short}-{algo}-{job8}) training's on_success uses."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    display = ""
    if tracking_uri:
        try:
            from amortized.core.mlflow_client import MLflowClient

            client = MLflowClient(tracking_uri)
            run = await client.get_run(run_id)
            tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
            display = tags.get("model_display_name", "")
        except Exception:
            logger.debug("Failed to read model_display_name for run %s", run_id, exc_info=True)
    if not display:
        parent_config = parent.get("config", {}) or {}
        algorithm = str(parent_config.get("algorithm", "sft"))
        base = str(parent_config.get("model_name_or_path", "") or "model")
        short = base.split("/")[-1]
        display = f"{short}-{algorithm}-{parent['id'][:8]}"
    return display


async def _resolve_training_model(
    config: dict[str, Any], config_files: dict[str, str]
) -> tuple[str, str, list[str]]:
    """Resolve a tuned model from a training job's MLflow artifacts.

    Returns (served_model_name, model_path, pre_commands). model_path is the
    in-container directory the servable checkpoint ends up in.
    """
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

    # Default served name: the training run's registered model tag (e.g.
    # mdl-brawny-jay-896), falling back to the registration-name pattern
    served_name = str(config.get("served_model_name", "")).strip()
    if not served_name:
        served_name = await _training_display_name(parent, run_id)

    local_dir = "/amortized/work/served_model"
    pre_commands = [
        # Download the HF export, then pick the highest-step checkpoint
        f"mlflow artifacts download -r {shlex.quote(run_id)} -a model/hf_format"
        f" -d {shlex.quote(local_dir)}",
        f"SERVE_MODEL_DIR=$(find {shlex.quote(local_dir)}/hf_format -mindepth 1 -maxdepth 1"
        " -type d | sort -V | tail -1)",
        # Training exports may be bare text towers (no architectures key) —
        # patch_model_config.py fills it in via transformers' own registry
        # and is a no-op for standard exports.
        "python3 /amortized/patch_model_config.py $SERVE_MODEL_DIR",
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

    # Pin to a GPU instead of requesting nvidia.com/gpu: all of the user's
    # serve deployments share one GPU (their quota budget), reusing the GPU
    # their running serve pods already hold. Falls back to a build error
    # when nothing is available.
    from amortized import config as config_module

    try:
        from amortized.core.gpu_inventory import assign_serve_gpu

        gpu_uuids = await assign_serve_gpu(
            config_module.settings.compute_namespace, gpus
        )
    except Exception as exc:
        # Non-kubernetes backends (local) have no inventory — request GPUs
        # the regular way there.
        if config_module.settings.compute_backend == "kubernetes":
            raise
        gpu_uuids = []

    if config.get("training_job_id"):
        served_name, model_path, pre_commands = await _resolve_training_model(
            config, config_files
        )
    else:
        served_name, model_path, pre_commands = _resolve_named_model(config)

    # $SERVE_MODEL_DIR is our own pre-command variable (find output, no
    # spaces) — shlex-quoting it would suppress expansion, so it goes in raw
    # while user-provided paths (HF ids) stay quoted.
    quoted_path = model_path if model_path.startswith("$") else shlex.quote(model_path)

    # Pre-flight: fail fast (before vLLM's slow engine init) when the model
    # weights won't fit in the GPU memory available to this job. Uses the
    # same model ref as the serve command (checkpoint dir or HF id).
    import contextlib

    gpu_memory_utilization = 0.9
    for extra in config.get("vllm_args", []) or []:
        if str(extra).startswith("--gpu-memory-utilization"):
            with contextlib.suppress(ValueError):
                gpu_memory_utilization = float(str(extra).split("=", 1)[-1].split()[-1])
    check_gpus = 1 if gpu_uuids else int(gpus)
    pre_commands.append(
        "python3 /amortized/check_gpu_memory.py"
        f" {quoted_path} {int(check_gpus)} {gpu_memory_utilization}"
    )

    serve_cmd = f"python3 /amortized/serve_vllm.py serve {quoted_path}"
    serve_cmd += f" --served-model-name {shlex.quote(served_name)}"
    serve_cmd += f" --port {int(port)}"
    for extra in config.get("vllm_args", []) or []:
        if extra:
            serve_cmd += f" {shlex.quote(str(extra))}"

    for asset in ("serve_vllm.py", "patch_model_config.py", "check_gpu_memory.py"):
        with open(os.path.join(_ASSETS_DIR, asset)) as f:
            config_files[asset] = f.read()

    cmd = ["sh", "-c", serve_cmd]

    resolved_config = dict(config)
    resolved_config["served_model_name"] = served_name
    resolved_config["port"] = port

    env: dict[str, str] = {}
    if gpu_uuids:
        # nvidia-container-runtime honors this env var (it is the node's
        # default runtime) — the pod sees exactly these GPUs, which the
        # device plugin does not account for, so the quota stays free for
        # training jobs.
        env["NVIDIA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
        resolved_config["gpu_uuids"] = gpu_uuids
        resource_gpus = 0
    else:
        resource_gpus = gpus

    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        pre_commands=pre_commands,
        env=env,
        # Serve jobs run until cancelled — no post commands, no completion
        post_commands=[],
        resources=Resources(gpus=resource_gpus, memory_gb=config.get("memory_gb") or None),
        image=IMAGE,
        ports={port: port},
        resolved_config=resolved_config,
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    # Serve jobs never complete on their own — nothing to do on success.
    return None
