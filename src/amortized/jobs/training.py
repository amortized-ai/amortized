"""Training job builder — LoRA SFT, OSFT, DPO, GKD via Training Hub."""

from __future__ import annotations

import logging
import os
import shlex
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.core.mlflow_client import MLflowClient
from amortized.jobs.base import JobBuildResult
from amortized.jobs.common import (
    resolve_training_model,
    set_mlflow_run_tag,
    training_display_name,
    validate_training_parent,
)

logger = logging.getLogger("amortized.jobs.training")

_TRAINING_HUB_FIELD_MAP: dict[str, str] = {
    "model_name_or_path": "model_path",
    "num_train_epochs": "num_epochs",
    "per_device_train_batch_size": "micro_batch_size",
    "max_length": "max_seq_len",
    "output_dir": "ckpt_output_dir",
}

ALGO_ALIASES: dict[str, str] = {
    "lora": "lora_sft",
    "qlora": "lora_sft",
    "qlora_sft": "lora_sft",
    "classifier": "embedding_sft",
    "embedding": "embedding_sft",
}

_TRAINING_HUB_SKIP_KEYS = {
    "algorithm",
    "engine",
    "use_peft",
    "qlora",
    "bnb_4bit_quant_type",
    "bnb_4bit_compute_dtype",
    "lora_target_modules",
    "model_id",
    "model",
    "num_samples",
    "compute",
    "task_description",
    "method",
    "dataset_job_id",
    "topic",
    "model_job_id",
    "training_job_id",
    "data_run_id",
    "resume_model_name",
}


def _training_hub_config_yaml(algorithm: str, config: dict[str, Any]) -> str:
    import yaml

    thub_config: dict[str, Any] = {}
    for key, value in config.items():
        if key in _TRAINING_HUB_SKIP_KEYS or value is None:
            continue
        if key == "output_dir" and algorithm == "gepa":
            thub_config["output_dir"] = value
            continue
        th_key = _TRAINING_HUB_FIELD_MAP.get(key, key)
        thub_config[th_key] = value

    output_dir = config.get("output_dir", "/amortized/work/output")
    if algorithm == "gepa":
        thub_config.setdefault("output_dir", output_dir)
    else:
        thub_config.setdefault("ckpt_output_dir", output_dir)
        thub_config.setdefault("data_output_dir", output_dir + "/processed_data")

    if algorithm in ("sft", "lora_sft"):
        batch = thub_config.pop("micro_batch_size", 2)
        thub_config.setdefault("effective_batch_size", batch * 4)
        thub_config.setdefault("max_seq_len", 2048)
        thub_config.setdefault("max_batch_len", 60000)
    elif algorithm == "osft":
        batch = thub_config.pop("micro_batch_size", 2)
        thub_config.setdefault("effective_batch_size", batch * 4)
        thub_config.setdefault("max_seq_len", 2048)
        thub_config.setdefault("max_tokens_per_gpu", 4096)
        thub_config.setdefault("learning_rate", 2e-5)
    elif algorithm == "embedding_sft":
        # Contrastive embedding fine-tuning (classifiers/routers). It uses
        # ``batch_size`` (not micro_batch_size) and has none of the SFT/OSFT
        # sequence-packing knobs, so drop the fields the generic map produced
        # that embedding_sft would only warn-and-ignore.
        batch = thub_config.pop("micro_batch_size", None)
        if batch is not None:
            thub_config.setdefault("batch_size", batch)
        for irrelevant in ("max_seq_len", "data_output_dir"):
            thub_config.pop(irrelevant, None)

    result: str = yaml.dump(thub_config, default_flow_style=False, sort_keys=False)
    return result


async def _resolve_parent_base_model(training_job_id: str) -> str:
    """Get the original base model name from the parent training job."""
    from amortized.db.connection import get_pool
    from amortized.db.repository import Repository

    async with get_pool().acquire() as conn:
        parent = await Repository(conn).get_job(training_job_id)
    if not parent:
        return "unknown"
    parent_config = parent.get("config") or {}
    return str(
        parent_config.get("model_name_or_path", "")
        or parent_config.get("model_id", "unknown")
    )


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    algorithm = config.get("algorithm", "sft")
    algorithm = ALGO_ALIASES.get(algorithm, algorithm)

    model_pre_commands: list[str] = []
    resume_parent: dict[str, Any] | None = None
    resume_run_id = ""
    if config.get("training_job_id"):
        resume_parent, resume_run_id = await validate_training_parent(config)
        _, model_path, model_pre_commands = await resolve_training_model(
            config, skip_display_name=True
        )
        config = dict(config)
        if any("merge_lora.py" in cmd for cmd in model_pre_commands):
            logger.warning(
                "Resuming from a LoRA parent: the adapter will be merged"
                " into the base model and training will continue on full"
                " weights (merge-then-full-train). To continue training"
                " the adapter instead, start a new lora_sft job with the"
                " same base model."
            )
        if model_path.startswith("$"):
            resolved_path = "/amortized/work/served_model/checkpoint"
            model_pre_commands.append(
                f'ln -sfn "{model_path}" {resolved_path}'
            )
            model_path = resolved_path
        config["model_name_or_path"] = model_path
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        for asset in ("merge_lora.py", "patch_model_config.py"):
            with open(os.path.join(assets_dir, asset)) as f:
                config_files[asset] = f.read()

    config_files["config.yaml"] = _training_hub_config_yaml(algorithm, config)

    if resume_parent is not None:
        parent_config = resume_parent.get("config") or {}
        config["model_name_or_path"] = str(
            parent_config.get("model_name_or_path", "")
            or parent_config.get("model_id", "unknown")
        )
        config["resume_model_name"] = await training_display_name(
            resume_parent, resume_run_id
        )
    thub_subcommand = algorithm.replace("_", "-")
    cmd = ["thub", thub_subcommand, "--config", "/amortized/config.yaml"]

    output_dir = config.get("output_dir", "/amortized/work/output")
    post_cmd = (
        f"mlflow artifacts log-artifacts -l {shlex.quote(output_dir)} -r $MLFLOW_RUN_ID -a model"
    )

    # embedding_sft trains in-process via SentenceTransformerTrainer (single device, no torchrun;
    # CUDA only if available), so it's CPU by DEFAULT (0 GPUs → no nvidia nodeSelector/runtimeclass,
    # schedules anywhere, frees the quota) with an explicit 1-GPU opt-in via train_on_gpu (a GPU is
    # ~100x faster — matters for many epochs / large datasets; see issue #486). nproc_per_node is
    # ignored here (single-device). All other algorithms scale with nproc_per_node. Keep in sync
    # with estimate_gpu_demand() in core/jobs.py.
    if algorithm == "embedding_sft":
        gpus = 1 if config.get("train_on_gpu") else 0
    else:
        gpus = config.get("nproc_per_node", 1)
    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        pre_commands=model_pre_commands,
        post_commands=[post_cmd],
        resources=Resources(gpus=gpus),
        image=config_mod.job_image("training", config_mod.settings.resolved_training_image_tag),
        resolved_config=dict(config),
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return

    config = job.get("config", {})
    algorithm = config.get("algorithm", "sft")
    job_id = job["id"]

    training_job_id = str(config.get("training_job_id", "") or "").strip()
    resume_model_name: str | None = None
    if training_job_id:
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.jobs.common import training_display_name

        async with get_pool().acquire() as conn:
            parent = await Repository(conn).get_job(training_job_id)
        if parent:
            parent_config = parent.get("config") or {}
            base_model = str(
                parent_config.get("model_name_or_path", "")
                or parent_config.get("model_id", "unknown")
            )
            parent_run_id = parent.get("mlflow_run_id", "")
            if parent_run_id:
                resume_model_name = await training_display_name(parent, parent_run_id)
        else:
            base_model = "unknown"
    else:
        base_model = config.get("model_name_or_path", config.get("model_id", "unknown"))
    short_name = base_model.split("/")[-1]
    model_name = f"{short_name}-{algorithm}-{job_id[:8]}"

    try:
        client = MLflowClient(tracking_uri)
        if resume_model_name:
            description = (
                f"Fine-tuned {base_model} via {algorithm} (job {job_id[:8]},"
                f" resumed from {resume_model_name})"
            )
        else:
            description = f"Fine-tuned {base_model} via {algorithm} (job {job_id[:8]})"
        registered = await client.register_model(model_name, mlflow_run_id, description)
        if not registered:
            logger.warning("Job %s succeeded but model registration failed", job_id)
            return

        run = await client.get_run(mlflow_run_id)
        run_name = run["info"].get("run_name", job_id[:8])
        display_name = f"mdl-{run_name}"
        await set_mlflow_run_tag(mlflow_run_id, "model_display_name", display_name)
        await client.set_registered_model_tag(model_name, "model_display_name", display_name)

        # Tag the model kind so eval steers embedding models to classification
        # eval (a generative/judge eval would fail on a sentence-transformers model).
        model_type = (
            "embedding"
            if algorithm in ("embedding_sft", "classifier", "embedding")
            else "generative"
        )
        await set_mlflow_run_tag(mlflow_run_id, "model_type", model_type)
        await client.set_registered_model_tag(model_name, "model_type", model_type)

        topic = config.get("topic", "")
        if not topic:
            parent_id = job.get("parent_job_id", "")
            if parent_id:
                from amortized.db.connection import get_pool
                async with get_pool().acquire() as conn:
                    parent = await conn.fetchrow(
                        "SELECT mlflow_run_id FROM jobs WHERE id = $1", parent_id,
                    )
                if parent and parent["mlflow_run_id"]:
                    try:
                        parent_run = await client.get_run(parent["mlflow_run_id"])
                        parent_tags = {
                            t["key"]: t["value"]
                            for t in parent_run["data"].get("tags", [])
                        }
                        topic = parent_tags.get("dataset_topic", "")
                    except Exception:
                        logger.debug(
                            "Could not resolve topic from parent job %s",
                            parent_id, exc_info=True,
                        )
        if topic:
            await set_mlflow_run_tag(mlflow_run_id, "model_topic", topic)
            await client.set_registered_model_tag(model_name, "model_topic", topic)
    except Exception:
        logger.warning("Failed to register model %s", model_name, exc_info=True)
