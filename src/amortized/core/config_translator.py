"""Translate amortized job configs into tool-native YAML for TRL."""

from __future__ import annotations

from typing import Any

import yaml

import amortized.config as config_mod

_TRL_ALGO_MAP: dict[str, str] = {
    "sft": "sft",
    "lora_sft": "sft",
    "osft": "sft",
    "grpo": "grpo",
    "lora_grpo": "grpo",
    "dpo": "dpo",
    "kto": "kto",
    "gkd": "gkd",
    # TODO: gepa has no TRL CLI equivalent yet
}

_TRL_FIELD_MAP: dict[str, str] = {
    "model_path": "model_name_or_path",
    "model_name_or_path": "model_name_or_path",
    "num_epochs": "num_train_epochs",
    "learning_rate": "learning_rate",
    "lora_r": "lora_r",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "micro_batch_size": "per_device_train_batch_size",
    "max_seq_len": "max_length",
    "bf16": "bf16",
}


def _trl_config_yaml(algorithm: str, config: dict[str, Any]) -> str:
    """Translate amortized training config -> TRL CLI YAML config."""
    trl_config: dict[str, Any] = {
        "model_name_or_path": config.get("model_name_or_path", config.get("model_path", "")),
        "output_dir": config.get("output_dir", "/amortized/work/output"),
        "report_to": config.get(
            "report_to", "mlflow" if config_mod.settings.mlflow_tracking_uri else "none"
        ),
    }

    data_path = config.get("data_path", config.get("dataset", ""))
    if data_path.startswith("s3://"):
        local_name = data_path.split("/")[-1]
        local_path = f"/amortized/work/{local_name}"
    elif data_path.endswith((".jsonl", ".json", ".csv", ".parquet")):
        local_path = data_path
    else:
        local_path = None

    if local_path:
        ext = local_path.rsplit(".", 1)[-1]
        builder = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}.get(
            ext, "json"
        )
        trl_config["dataset_name"] = builder
        trl_config["dataset_kwargs"] = {"data_files": local_path}
    else:
        trl_config["dataset_name"] = data_path

    if algorithm == "gkd" and config.get("teacher_model_name_or_path"):
        trl_config["teacher_model_name_or_path"] = config["teacher_model_name_or_path"]

    skip_keys = {
        "algorithm",
        "engine",
        "data_path",
        "dataset",
        "model_name_or_path",
        "model_path",
        "teacher_model_name_or_path",
    }
    for key, value in config.items():
        if key in skip_keys or value is None:
            continue
        mapped = _TRL_FIELD_MAP.get(key, key)
        trl_config[mapped] = value

    if config.get("lora_target_modules") and isinstance(config["lora_target_modules"], list):
        trl_config["lora_target_modules"] = " ".join(config["lora_target_modules"])

    if config.get("qlora"):
        trl_config["load_in_4bit"] = True

    result: str = yaml.dump(trl_config, default_flow_style=False, sort_keys=False)
    return result
