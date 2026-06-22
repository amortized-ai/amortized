"""Translate amortized job configs into tool-native YAML for TRL, vLLM, and asynth."""

from __future__ import annotations

import logging
from typing import Any

import yaml

import amortized.config as config_mod

logger = logging.getLogger("amortized.worker")

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


_SERVE_FIELD_MAP: dict[str, str] = {
    "model_name_or_path": "model",
    "served_model_name": "served-model-name",
    "tensor_parallel_size": "tensor-parallel-size",
}

_SERVE_SKIP_KEYS = {"adapter_path", "output_dir", "gpu_ids"}


def _serve_config_yaml(config: dict[str, Any]) -> str:
    vllm_config: dict[str, Any] = {"host": "0.0.0.0"}

    served_name = config.get("served_model_name", "default")
    adapter_path = config.get("adapter_path")

    if adapter_path:
        vllm_config["enable-lora"] = True
        vllm_config["lora-modules"] = f"{served_name}={adapter_path}"

    for key, value in config.items():
        if key in _SERVE_SKIP_KEYS or value is None:
            continue
        vllm_field = _SERVE_FIELD_MAP.get(key)
        if vllm_field:
            vllm_config[vllm_field] = value
        elif key not in _SERVE_FIELD_MAP:
            vllm_config[key] = value

    result: str = yaml.dump(vllm_config, default_flow_style=False, sort_keys=False)
    return result


def _generate_container_config(
    job_type: str, config: dict[str, Any], *, s3_output_path: str = ""
) -> dict[str, Any]:
    """Build an asynth-compatible config dict for container execution."""
    if job_type == "sdg":
        return _build_synth_config(config, s3_output_path=s3_output_path)
    raise ValueError(f"No container config for job type: {job_type}")


def _build_synth_config(config: dict[str, Any], *, s3_output_path: str = "") -> dict[str, Any]:
    """Build an asynth-compatible synthesis config dict for CLI execution."""
    inference_config: dict[str, Any] = {
        "model": config["model"],
        "temperature": config.get("temperature", 0.7),
        "max_concurrency": config.get("max_concurrency", 16),
        "num_retries": config.get("num_retries", 3),
    }
    for optional in ("max_tokens", "top_p", "seed", "api_base", "api_key"):
        if config.get(optional) is not None:
            inference_config[optional] = config[optional]

    strategy_params = config.get("strategy_params", {})
    if isinstance(strategy_params, dict):
        strategy_params = dict(strategy_params)
        if config.get("input_data") and "input_data" not in strategy_params:
            strategy_params["input_data"] = config["input_data"]
        if config.get("input_documents") and "input_documents" not in strategy_params:
            strategy_params["input_documents"] = config["input_documents"]

    output_path = s3_output_path or "output/generated_data.jsonl"
    return {
        "inference_config": inference_config,
        "num_samples": config.get("num_samples", 100),
        "output_path": output_path,
        "strategy_params": strategy_params,
    }


def _resolve_judge_template(config: dict[str, Any]) -> dict[str, Any]:
    """If the judge config references a template, load it and merge the prompt."""
    judge = config.get("judge")
    if not judge or not isinstance(judge, dict):
        return config
    template_name = judge.get("template")
    if not template_name:
        return config
    from amortized.core.judge_templates import load_judge_template

    try:
        tmpl = load_judge_template(template_name)
    except FileNotFoundError:
        logger.warning("Judge template '%s' not found, skipping", template_name)
        return config
    tmpl_config = tmpl.get("config", tmpl)
    tmpl_judge = tmpl_config.get("judge", {})
    merged_judge = dict(judge)
    if "prompt" not in merged_judge:
        merged_judge["prompt"] = tmpl_judge.get("prompt") or tmpl_config.get("judge_prompt", "")
    if tmpl_config.get("system_instruction"):
        merged_judge["system_instruction"] = tmpl_config["system_instruction"]
    for key in ("judgment_type", "response_format", "include_explanation"):
        if tmpl_config.get(key) is not None and key not in merged_judge:
            merged_judge[key] = tmpl_config[key]
    config = {**config, "judge": merged_judge}
    logger.info("Resolved judge template '%s'", template_name)
    return config


def _build_judge_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build an asynth-compatible judge config dict for CLI execution."""
    judge = config.get("judge", {})
    result: dict[str, Any] = {
        "judge_params": {
            "prompt_template": judge.get("prompt", "Evaluate this response: {response}"),
            "response_format": judge.get("response_format", "json"),
            "judgment_type": judge.get("judgment_type", "bool"),
            "include_explanation": judge.get("include_explanation", True),
        },
        "inference_config": {
            "model": judge.get("model", "openai/gpt-4o-mini"),
            "temperature": judge.get("temperature", 0.0),
        },
    }
    if judge.get("system_instruction"):
        result["judge_params"]["system_instruction"] = judge["system_instruction"]
    return result


def _eval_config_yaml(config: dict[str, Any]) -> str:
    """Translate amortized eval config -> asynth judge YAML config."""
    judge_cfg = _build_judge_config(config)

    eval_config: dict[str, Any] = {
        **judge_cfg,
        "dataset": config.get("dataset", ""),
        "output_path": "/amortized/work/eval_results.json",
    }

    if config.get("model_endpoint"):
        eval_config["model_endpoint"] = config["model_endpoint"]
    if config.get("model_name"):
        eval_config["model_name"] = config["model_name"]
    if config.get("max_samples"):
        eval_config["max_samples"] = config["max_samples"]
    if config.get("temperature") is not None:
        eval_config["temperature"] = config["temperature"]
    if config.get("deterministic_checks"):
        eval_config["deterministic_checks"] = config["deterministic_checks"]

    result: str = yaml.dump(eval_config, default_flow_style=False, sort_keys=False)
    return result
