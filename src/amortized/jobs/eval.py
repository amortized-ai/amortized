"""Eval job builder — compare base and tuned model endpoints on an eval dataset."""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any

from amortized.backends import Resources
from amortized.jobs.base import JobBuildError, JobBuildResult
from amortized.jobs.common import set_mlflow_run_tag

logger = logging.getLogger("amortized.jobs.eval")

IMAGE = "ghcr.io/amortized-ai/eval:latest"

_JUDGE_METRICS = {"judge_win_rate"}
_ENDPOINT_KEYS = ("endpoint_base", "endpoint_tuned", "judge")


def _endpoint_spec(
    config: dict[str, Any], key: str, env_name: str, env: dict[str, str]
) -> dict[str, str]:
    endpoint = config.get(key)
    if not isinstance(endpoint, dict) or not endpoint.get("base_url") or not endpoint.get("model"):
        raise JobBuildError(f"{key}: base_url and model are required")
    api_key = endpoint.get("api_key", "")
    if api_key:
        env[env_name] = api_key
    return {
        "base_url": str(endpoint["base_url"]).rstrip("/"),
        "model": str(endpoint["model"]),
        "api_key_env": env_name,
    }


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    env: dict[str, str] = {}

    endpoints = {
        "base": _endpoint_spec(config, "endpoint_base", "EVAL_BASE_API_KEY", env),
        "tuned": _endpoint_spec(config, "endpoint_tuned", "EVAL_TUNED_API_KEY", env),
    }

    metrics = [m for m in (config.get("metrics") or []) if m]
    if any(m in _JUDGE_METRICS for m in metrics) and not config.get("judge"):
        raise JobBuildError("judge endpoint is required when metrics include 'judge_win_rate'")
    if config.get("judge"):
        endpoints["judge"] = _endpoint_spec(config, "judge", "EVAL_JUDGE_API_KEY", env)
        if "judge_win_rate" not in metrics:
            metrics.append("judge_win_rate")

    eval_data_path = config.get("eval_data_path", "")
    eval_data_run_id = config.get("eval_data_run_id", "")
    pre_commands: list[str] = []
    if not eval_data_path and eval_data_run_id:
        local_dir = "/amortized/work/eval_data"
        pre_commands.append(
            f"mlflow artifacts download"
            f" -r {shlex.quote(eval_data_run_id)}"
            f" -a generated_data"
            f" -d {shlex.quote(local_dir)}"
        )
        eval_data_path = f"{local_dir}/generated_data"
    if not eval_data_path:
        raise JobBuildError(
            "eval jobs require either parent_job_id (a completed SDG/upload job"
            " with a dataset) or eval_data_run_id (an MLflow run with a"
            " generated_data artifact)"
        )

    runner_config = {
        "eval_data_path": eval_data_path,
        "endpoints": endpoints,
        "metrics": metrics,
        "max_samples": config.get("max_samples", 200),
        "judge_max_samples": config.get("judge_max_samples", 100),
        "temperature": config.get("temperature", 0.0),
        "output_dir": "/amortized/work/results",
    }
    config_files["config.json"] = json.dumps(runner_config)

    post_cmd = (
        "mlflow artifacts log-artifacts"
        " -l /amortized/work/results"
        " -r $MLFLOW_RUN_ID"
        " -a eval_results"
    )

    resolved_config = dict(config)
    for key in _ENDPOINT_KEYS:
        endpoint = resolved_config.get(key)
        if isinstance(endpoint, dict):
            scrubbed = dict(endpoint)
            scrubbed.pop("api_key", None)
            resolved_config[key] = scrubbed
    resolved_config["eval_data_path"] = eval_data_path

    return JobBuildResult(
        command=["python3", "/app/run_eval.py", "--config", "/amortized/config.json"],
        config_files=config_files,
        env=env,
        pre_commands=pre_commands,
        post_commands=[post_cmd],
        resources=Resources(gpus=0, cpus=2, memory_gb=4),
        image=IMAGE,
        resolved_config=resolved_config,
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    job_config = job.get("config", {})
    if isinstance(job_config, str):
        job_config = json.loads(job_config)

    base_model = (job_config.get("endpoint_base") or {}).get("model", "")
    tuned_model = (job_config.get("endpoint_tuned") or {}).get("model", "")
    if base_model:
        await set_mlflow_run_tag(mlflow_run_id, "eval_base_model", base_model)
    if tuned_model:
        await set_mlflow_run_tag(mlflow_run_id, "eval_tuned_model", tuned_model)

    topic = job_config.get("topic", "")
    if topic:
        await set_mlflow_run_tag(mlflow_run_id, "eval_topic", topic)

    try:
        import amortized.config as config_mod
        from amortized.core.mlflow_client import MLflowClient

        tracking_uri = config_mod.settings.mlflow_tracking_uri
        if not tracking_uri:
            return
        client = MLflowClient(tracking_uri)
        metrics_text = await client.get_artifact_text(mlflow_run_id, "eval_results/metrics.json")
        if not metrics_text:
            return
        metrics = json.loads(metrics_text).get("results", {})

        for name in ("base", "tuned"):
            em = metrics.get(name, {}).get("exact_match")
            if em is not None:
                await set_mlflow_run_tag(mlflow_run_id, f"eval_exact_match_{name}", str(em))
        win_rate = metrics.get("judge", {}).get("win_rate")
        if win_rate is not None:
            await set_mlflow_run_tag(mlflow_run_id, "eval_win_rate", str(win_rate))
        num_samples = metrics.get("base", {}).get("num_samples")
        if num_samples is not None:
            await set_mlflow_run_tag(mlflow_run_id, "num_samples", str(num_samples))
    except Exception:
        logger.debug("Failed to tag eval metrics on run %s", mlflow_run_id, exc_info=True)
