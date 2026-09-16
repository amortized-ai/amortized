"""Training job builder — LoRA SFT, OSFT, DPO, GKD via Training Hub."""

from __future__ import annotations

import logging
import shlex
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.core.mlflow_client import MLflowClient
from amortized.core.model_catalog import training_model_cpu_memory_gb
from amortized.jobs.base import JobBuildResult
from amortized.jobs.common import set_mlflow_run_tag

logger = logging.getLogger("amortized.jobs.training")

_TRAINING_HUB_FIELD_MAP: dict[str, str] = {
    "model_name_or_path": "model_path",
    "num_train_epochs": "num_epochs",
    "per_device_train_batch_size": "micro_batch_size",
    "max_length": "max_seq_len",
    "output_dir": "ckpt_output_dir",
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
    "device",
    "timeout_seconds",
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

    result: str = yaml.dump(thub_config, default_flow_style=False, sort_keys=False)
    return result


IMAGE_TAG = "latest"
CPU_CPUS = 4
CPU_TIMEOUT_SECONDS = 3600


def _image(device: str) -> str:
    """Compose the training image from settings.image_registry (upload.py pattern).

    GPU keeps the historical ``training:latest``; CPU uses a separate
    ``training-cpu`` image with a versioned tag from day one.
    """
    registry = config_mod.settings.image_registry
    if device == "cpu":
        return f"{registry}/training-cpu:{config_mod.settings.training_cpu_image_tag}"
    return f"{registry}/training:{IMAGE_TAG}"


def _cpu_timeout_seconds(raw: Any) -> int:
    """Fail-open validation of the optional ``timeout_seconds`` config key.

    Only a positive integer is honored; anything else (non-numeric string,
    0, negative) falls back to the default with a warning rather than
    raising or producing an invalid ``activeDeadlineSeconds``.
    """
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        if raw is not None:
            logger.warning(
                "Ignoring invalid timeout_seconds=%r (must be a positive integer); "
                "using default %ss",
                raw,
                CPU_TIMEOUT_SECONDS,
            )
        return CPU_TIMEOUT_SECONDS
    return raw


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    algo_aliases = {"lora": "lora_sft", "qlora": "lora_sft", "qlora_sft": "lora_sft"}
    algorithm = config.get("algorithm", "sft")
    algorithm = algo_aliases.get(algorithm, algorithm)

    env: dict[str, str] = {}
    timeout: int | None = None
    if config.get("device") == "cpu":
        # CPU-safe defaults enforced here, not agent-reliant: cap processes at 1
        # and run fp32 (bf16 has no effect on CPU anyway).
        config = {**config, "bf16": False, "nproc_per_node": 1}
        memory_gb = training_model_cpu_memory_gb(config.get("model_name_or_path", ""))
        resources = Resources(gpus=0, cpus=CPU_CPUS, memory_gb=memory_gb)
        # Thread-thrash guard: match thread pools to the CPU request.
        env = {
            "OMP_NUM_THREADS": str(CPU_CPUS),
            "MKL_NUM_THREADS": str(CPU_CPUS),
            "TOKENIZERS_PARALLELISM": "false",
        }
        timeout = _cpu_timeout_seconds(config.get("timeout_seconds"))
    else:
        resources = Resources(gpus=config.get("nproc_per_node", 1))

    config_files["config.yaml"] = _training_hub_config_yaml(algorithm, config)
    if config.get("device") == "cpu":
        # The CPU image cannot run the `thub` CLI: lora_sft's only backend
        # (unsloth) is excluded by design, and sft's only backend
        # (instructlab-training) unconditionally calls torch.cuda.set_device().
        # Dispatch the TRL-based runner baked into the image instead — it
        # consumes the same thub-format config.yaml generated above. It lives
        # at /usr/local/bin (not /amortized) because the K8s ConfigMap mounts
        # /amortized read-only and would shadow image files placed there.
        cmd = ["python3", "/usr/local/bin/cpu_sft.py", "--config", "/amortized/config.yaml"]
    else:
        thub_subcommand = algorithm.replace("_", "-")
        cmd = ["thub", thub_subcommand, "--config", "/amortized/config.yaml"]

    output_dir = config.get("output_dir", "/amortized/work/output")
    post_cmd = (
        f"mlflow artifacts log-artifacts -l {shlex.quote(output_dir)} -r $MLFLOW_RUN_ID -a model"
    )

    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        post_commands=[post_cmd],
        env=env,
        resources=resources,
        image=_image(config.get("device", "gpu")),
        timeout=timeout,
        resolved_config=dict(config),
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return

    config = job.get("config", {})
    base_model = config.get("model_name_or_path", config.get("model_id", "unknown"))
    short_name = base_model.split("/")[-1]
    algorithm = config.get("algorithm", "sft")
    job_id = job["id"]
    model_name = f"{short_name}-{algorithm}-{job_id[:8]}"

    try:
        client = MLflowClient(tracking_uri)
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
