"""Training job builder — LoRA SFT, OSFT, DPO, GKD via Training Hub."""

from __future__ import annotations

import logging
import shlex
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.core.mlflow_client import MLflowClient
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

    # Keep each safetensors shard under 500MB so the MLflow --serve-artifacts
    # proxy (3Gi memory cap) doesn't OOM on large model uploads.
    thub_config.setdefault("max_shard_size", "500MB")

    result: str = yaml.dump(thub_config, default_flow_style=False, sort_keys=False)
    return result


IMAGE = "ghcr.io/amortized-ai/training:latest"


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    algo_aliases = {"lora": "lora_sft", "qlora": "lora_sft", "qlora_sft": "lora_sft"}
    algorithm = config.get("algorithm", "sft")
    algorithm = algo_aliases.get(algorithm, algorithm)

    config_files["config.yaml"] = _training_hub_config_yaml(algorithm, config)
    thub_subcommand = algorithm.replace("_", "-")
    cmd = ["thub", thub_subcommand, "--config", "/amortized/config.yaml"]

    output_dir = config.get("output_dir", "/amortized/work/output")
    max_shard = config.get("max_shard_size", "500MB")
    config_files["reshard.py"] = "\n".join([
        "import glob, os",
        f"output_dir = '{output_dir}'",
        "hf_dirs = sorted(glob.glob(os.path.join(output_dir, 'hf_format', '*')), key=os.path.getmtime)",
        "model_dir = hf_dirs[-1] if hf_dirs else output_dir",
        f"max_shard_size = '{max_shard}'",
        "print(f'Re-sharding model in {model_dir} with max_shard_size={max_shard_size}')",
        "from transformers import AutoModelForCausalLM",
        "m = AutoModelForCausalLM.from_pretrained(model_dir)",
        "m.save_pretrained(model_dir, max_shard_size=max_shard_size)",
        "orig = os.path.join(model_dir, 'model.safetensors')",
        "if os.path.exists(orig) and len([f for f in os.listdir(model_dir) if f.startswith('model-') and f.endswith('.safetensors')]) > 0:",
        "    os.remove(orig)",
        "    print(f'Removed original {orig}')",
        "print('Re-sharding complete')",
        "for f in sorted(os.listdir(model_dir)):",
        "    size = os.path.getsize(os.path.join(model_dir, f))",
        "    if size > 1024:",
        "        print(f'  {f}: {size / 1024 / 1024:.1f} MB')",
    ])
    reshard_cmd = "python3 /amortized/reshard.py"
    upload_cmd = (
        f"mlflow artifacts log-artifacts -l {shlex.quote(output_dir)} -r $MLFLOW_RUN_ID -a model"
    )

    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        post_commands=[reshard_cmd, upload_cmd],
        resources=Resources(gpus=config.get("nproc_per_node", 1)),
        image=IMAGE,
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
