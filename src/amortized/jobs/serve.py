"""Serve job builder — bring up a persistent vLLM inference endpoint.

Serves either a tuned model from a completed training job (downloading the
merged HF export from MLflow) or any model by name/path (HF hub id). The job
stays running until cancelled; a Kubernetes Service exposes the endpoint
in-cluster for eval jobs.
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

_MERGE_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "assets", "merge_text_export.py")


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

    parent_config = parent.get("config", {}) or {}
    base_model = str(parent_config.get("model_name_or_path", "")).strip()

    # Default served name: the training run's registered model tag (e.g.
    # mdl-brawny-jay-896), falling back to the registration-name pattern
    served_name = str(config.get("served_model_name", "")).strip()
    if not served_name:
        served_name = await _training_display_name(parent, run_id)

    local_dir = "/amortized/work/served_model"
    pre_commands = [
        # Download the merged HF export, then pick the highest-step checkpoint
        f"mlflow artifacts download -r {shlex.quote(run_id)} -a model/hf_format"
        f" -d {shlex.quote(local_dir)}",
        f"SERVE_MODEL_DIR=$(find {shlex.quote(local_dir)}/hf_format -mindepth 1 -maxdepth 1"
        " -type d | sort -V | tail -1)",
    ]

    # Training Hub exports Qwen3.5 checkpoints as the bare text tower
    # (model_type qwen3_5_text), which vLLM cannot serve. When the export is
    # text-only, graft it back onto the base multimodal model (script ships
    # as a config file; it copies through untouched for servable exports).
    co_serve_base = bool(config.get("serve_base", True)) and bool(base_model)
    if base_model:
        merge_dir = "/amortized/work/merged_model"
        pre_commands.append(
            "python3 /amortized/merge_text_export.py"
            f" $SERVE_MODEL_DIR {shlex.quote(base_model)} {shlex.quote(merge_dir)}"
        )
        with open(_MERGE_SCRIPT_PATH) as f:
            config_files["merge_text_export.py"] = f.read()
        model_path = merge_dir
    else:
        model_path = "$SERVE_MODEL_DIR"

    base_model_out = base_model if co_serve_base else ""
    return served_name, model_path, pre_commands, base_model_out


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

    base_model = ""
    if config.get("training_job_id"):
        served_name, model_path, pre_commands, base_model = await _resolve_training_model(
            config, config_files
        )
    else:
        served_name, model_path, pre_commands = _resolve_named_model(config)

    # Built as a single sh -c string so the worker's _wrap_command can chain
    # the pre-commands (download, merge) in front of it in one shell.
    max_model_len = config.get("max_model_len")
    if base_model and not max_model_len:
        # Two memory-capped instances share one GPU — the default 256K
        # context of e.g. Qwen3.5 needs more KV cache than the cap allows
        max_model_len = 32768
    tuned_cmd = f"vllm serve {shlex.quote(model_path)}"
    tuned_cmd += f" --served-model-name {shlex.quote(served_name)}"
    tuned_cmd += f" --port {int(port)}"
    for extra in config.get("vllm_args", []) or []:
        if extra:
            tuned_cmd += f" {shlex.quote(str(extra))}"

    ports = {port: port}
    if base_model:
        # Co-serve the base model alongside the tuned one (same GPU — both
        # fit; per-user GPU quota is 1): two vLLM processes, memory-capped.
        # Grouped with { ...; } so the worker's pre-command && chain does not
        # pull the background jobs into its own subshell. POSIX-sh safe wait
        # (dash has no wait -n); exit code is nonzero if either died.
        # max-num-seqs is capped because hybrid-attention models (Qwen3.5)
        # need per-sequence state blocks that don't fit at 0.45 utilization.
        base_port = port + 1
        shared_flags = (
            f" --gpu-memory-utilization 0.45 --max-model-len {int(max_model_len)}"
            " --max-num-seqs 64"
        )
        base_cmd = f"vllm serve {shlex.quote(base_model)}"
        base_cmd += f" --served-model-name {shlex.quote(base_model)}"
        base_cmd += f" --port {int(base_port)}{shared_flags}"
        serve_cmd = (
            f"{{ {base_cmd} & P1=$!;"
            f" {tuned_cmd}{shared_flags} & P2=$!;"
            " wait $P1; S1=$?; wait $P2; exit $((S1+$?)); }"
        )
        ports[base_port] = base_port
    else:
        serve_cmd = tuned_cmd

    cmd = ["sh", "-c", serve_cmd]

    resolved_config = dict(config)
    resolved_config["served_model_name"] = served_name
    resolved_config["port"] = port
    if base_model:
        resolved_config["base_model"] = base_model
        resolved_config["base_port"] = port + 1

    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        pre_commands=pre_commands,
        # Serve jobs run until cancelled — no post commands, no completion
        post_commands=[],
        resources=Resources(gpus=gpus, memory_gb=config.get("memory_gb") or None),
        image=IMAGE,
        ports=ports,
        resolved_config=resolved_config,
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    # Serve jobs never complete on their own — nothing to do on success.
    return None
