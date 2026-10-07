"""Shared helpers used by multiple job builders."""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient

logger = logging.getLogger("amortized.jobs")

ArtifactResult = tuple[dict[str, Any], list[str]]


async def set_mlflow_run_tag(mlflow_run_id: str, key: str, value: str) -> None:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return

    try:
        client = MLflowClient(tracking_uri)
        await client.set_tag(mlflow_run_id, key, value)
    except Exception:
        logger.warning(
            "Failed to set MLflow tag %s=%s on run %s",
            key,
            value,
            mlflow_run_id,
            exc_info=True,
        )


async def fetch_document_chunks(document_id: str) -> list[str]:
    """Fetch pre-chunked document content from MLflow artifact store."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not document_id:
        return []

    client = MLflowClient(tracking_uri)
    metadata_text = await client.get_artifact_text(document_id, "chunks/metadata.json")
    if not metadata_text:
        return []

    metadata = json.loads(metadata_text)
    chunks: list[str] = []
    for i in range(len(metadata)):
        text = await client.get_artifact_text(document_id, f"chunks/chunk_{i:03d}.md")
        if text:
            chunks.append(text)
    return chunks


async def is_lora_export(run_id: str) -> bool:
    """True when the training run's model artifacts are a bare PEFT adapter.

    lora_sft saves adapter_config.json/adapter_model.safetensors at the top
    of model/ and no merged hf_format export; every other algorithm exports
    model/hf_format/<step>/. Detection is by artifact listing, not the
    parent job's algorithm field, so any peft-style export is handled.
    """
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri:
        return False
    try:
        from amortized.core.mlflow_client import MLflowClient

        client = MLflowClient(tracking_uri)
        files = await client.list_artifacts(run_id, "model")
        return any(f.get("path") == "model/adapter_config.json" for f in files)
    except Exception:
        logger.debug("artifact listing failed for run %s", run_id, exc_info=True)
        return False


async def training_display_name(parent: dict[str, Any], run_id: str) -> str:
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


async def resolve_training_model(
    config: dict[str, Any],
) -> tuple[str, str, list[str]]:
    """Resolve a tuned model from a training job's MLflow artifacts.

    Returns (served_model_name, model_path, pre_commands). model_path is the
    in-container directory the servable checkpoint ends up in.
    """
    from amortized.jobs.base import JobBuildError

    training_job_id = str(config.get("training_job_id", "")).strip()
    if not training_job_id:
        raise JobBuildError("training_job_id is required when evaluating a tuned model")

    from amortized.db.connection import get_pool

    async with get_pool().acquire() as conn:
        from amortized.db.repository import Repository

        parent = await Repository(conn).get_job(training_job_id)
    if not parent or parent["type"] != "training":
        raise JobBuildError(f"training_job_id {training_job_id!r} is not a training job")
    if parent["status"] != "succeeded":
        raise JobBuildError(
            f"training job {training_job_id[:8]} has status {parent['status']!r}"
            " — only succeeded training jobs can be evaluated"
        )
    run_id = parent.get("mlflow_run_id", "")
    if not run_id:
        raise JobBuildError(f"training job {training_job_id[:8]} has no MLflow run")

    served_name = str(config.get("served_model_name", "")).strip()
    if not served_name:
        served_name = await training_display_name(parent, run_id)

    local_dir = "/amortized/work/served_model"
    if await is_lora_export(run_id):
        pre_commands = [
            f"mlflow artifacts download -r {shlex.quote(run_id)} -a model"
            f" -d {shlex.quote(local_dir)}",
            "python3 /amortized/merge_lora.py"
            f" {shlex.quote(local_dir)}/model {shlex.quote(local_dir)}/merged",
            f"python3 /amortized/patch_model_config.py {shlex.quote(local_dir)}/merged",
        ]
        model_path = f"{local_dir}/merged"
    else:
        pre_commands = [
            f"mlflow artifacts download -r {shlex.quote(run_id)} -a model/hf_format"
            f" -d {shlex.quote(local_dir)}",
            f"SERVE_MODEL_DIR=$(find {shlex.quote(local_dir)}/hf_format -mindepth 1 -maxdepth 1"
            " -type d | sort -V | tail -1)",
            "python3 /amortized/patch_model_config.py $SERVE_MODEL_DIR",
        ]
        model_path = "$SERVE_MODEL_DIR"

    return served_name, model_path, pre_commands


async def resolve_parent_artifacts(
    job: dict[str, Any],
    config: dict[str, Any],
) -> ArtifactResult:
    """Resolve parent job artifacts and inject into config for chaining.

    Returns (config, pre_commands) where pre_commands are shell strings that
    download artifacts via ``mlflow.artifacts.download_artifacts()``.
    The caller must place pre_commands into ``JobBuildResult.pre_commands``.
    """
    parent_job_id = job.get("parent_job_id", "") or config.get("parent_job_id", "")

    from amortized.db.connection import get_pool
    from amortized.db.repository import Repository
    from amortized.models import JobType

    # A direct dataset run reference (uploaded dataset or split) works
    # without a parent job — mirror the parent-based download below.
    if not parent_job_id:
        data_run_id = str(config.get("data_run_id", "") or "")
        if job["type"] == JobType.training.value and data_run_id and not config.get("data_path"):
            local_dir = "/amortized/work/data"
            config = dict(config)
            config["data_path"] = f"{local_dir}/generated_data"
            download_cmd = (
                f"mlflow artifacts download"
                f" -r {shlex.quote(data_run_id)}"
                f" -a generated_data"
                f" -d {shlex.quote(local_dir)}"
            )
            logger.info(
                "Will download training data from MLflow run %s to %s",
                data_run_id,
                local_dir,
            )
            return config, [download_cmd]
        return config, []

    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        parent = await repo.get_job(parent_job_id)

    if not parent:
        logger.warning("Parent job %s not found", parent_job_id)
        return config, []

    parent_run_id = parent.get("mlflow_run_id", "")
    if not parent_run_id:
        logger.warning("Parent job %s has no mlflow_run_id", parent_job_id)
        return config, []

    pre_commands: list[str] = []
    config = dict(config)

    if job["type"] == JobType.training.value and parent["type"] in ("sdg", "upload"):
        existing = config.get("data_path", "")
        if not existing or not existing.startswith("s3://"):
            local_dir = "/amortized/work/data"
            pre_cmd = (
                f"mlflow artifacts download"
                f" -r {shlex.quote(parent_run_id)}"
                f" -a generated_data"
                f" -d {shlex.quote(local_dir)}"
            )
            pre_commands.append(pre_cmd)
            config["data_path"] = f"{local_dir}/generated_data"
            logger.info(
                "Will download artifacts from MLflow run %s to %s",
                parent_run_id,
                local_dir,
            )
    elif job["type"] == JobType.eval.value and parent["type"] in ("sdg", "upload"):
        local_dir = "/amortized/work/eval_data"
        pre_cmd = (
            f"mlflow artifacts download"
            f" -r {shlex.quote(parent_run_id)}"
            f" -a generated_data"
            f" -d {shlex.quote(local_dir)}"
        )
        pre_commands.append(pre_cmd)
        config["eval_data_path"] = f"{local_dir}/generated_data"
        logger.info(
            "Will download eval data from MLflow run %s to %s",
            parent_run_id,
            local_dir,
        )

    return config, pre_commands
