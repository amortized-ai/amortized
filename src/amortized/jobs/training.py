"""Training job builder — LoRA SFT, OSFT, DPO, GKD via Training Hub."""

from __future__ import annotations

import logging
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.core.mlflow_client import MLflowClient
from amortized.jobs.base import JobBuildResult

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

    return yaml.dump(thub_config, default_flow_style=False, sort_keys=False)


IMAGE = "ghcr.io/amortized-ai/training:latest"


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    env: dict[str, str] = {}

    if config_mod.settings.mlflow_tracking_uri:
        config.setdefault("report_to", "mlflow")
        env["HF_MLFLOW_LOG_ARTIFACTS"] = "true"

    algo_aliases = {"lora": "lora_sft", "qlora": "lora_sft", "qlora_sft": "lora_sft"}
    algorithm = config.get("algorithm", "sft")
    algorithm = algo_aliases.get(algorithm, algorithm)

    config_files["config.yaml"] = _training_hub_config_yaml(algorithm, config)
    thub_subcommand = algorithm.replace("_", "-")
    cmd = ["thub", thub_subcommand, "--config", "/amortized/config.yaml"]

    # After training, upload output dir to the MLflow run that TRL created.
    # Uses MlflowClient (not fluent API) — runs in a separate process with
    # no active run context.
    _upload_script = (
        "import os, mlflow; "
        "tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', ''); "
        "exp_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', ''); "
        "output = '/amortized/work/output'; "
        "c = mlflow.MlflowClient(tracking_uri) if tracking_uri else None; "
        "exp = c.get_experiment_by_name(exp_name) if c and exp_name else None; "
        "runs = c.search_runs("
        "[exp.experiment_id], order_by=['start_time DESC'], max_results=1"
        ") if exp else []; "
        "run_id = runs[0].info.run_id if runs else ''; "
        "run_id and os.path.isdir(output) "
        "and c.log_artifacts(run_id, output, 'model'); "
        "print(f'AMORTIZED_MLFLOW_RUN_ID={run_id}') if run_id else None"
    )
    post_cmd = f'python3 -c "{_upload_script}"'

    return JobBuildResult(
        command=cmd,
        config_files=config_files,
        env=env,
        resources=Resources(gpus=config.get("nproc_per_node", 1)),
        image=IMAGE,
        resolved_config=dict(config),
        post_commands=[post_cmd],
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return

    model_id = job.get("config", {}).get("model_id", "unknown")
    algorithm = job.get("config", {}).get("algorithm", "sft")
    job_id = job["id"]
    model_name = f"{model_id}-{algorithm}-{job_id[:8]}"

    try:
        client = MLflowClient(tracking_uri)
        description = f"Fine-tuned {model_id} via {algorithm} (job {job_id[:8]})"
        registered = await client.register_model(model_name, mlflow_run_id, description)
        if not registered:
            logger.warning("Job %s succeeded but model registration failed", job_id)
    except Exception:
        logger.warning("Failed to register model %s", model_name, exc_info=True)
