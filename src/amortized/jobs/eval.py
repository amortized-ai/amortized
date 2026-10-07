"""Eval job builder — evaluate one model on an eval dataset.

Two modes:

- **Embedded serving** (the common path): the eval config names the model
  to evaluate (``training_job_id`` for a tuned model, or
  ``model_name_or_path`` for an HF id). The eval job pod serves the model
  itself — vLLM starts in the background, the script waits for the
  endpoint to become healthy, runs the eval against localhost, and
  exits. The serve lifetime is the eval lifetime: scores done → pod done.
  GPU allocation matches training jobs: the pod requests
  ``nvidia.com/gpu`` (``nproc_per_node``) and is bounded by the user's
  namespace quota — GPUs are assumed free, so no occupancy checking.
- **External endpoint**: ``endpoint`` points at an OpenAI-compatible URL
  (e.g. a gateway model). CPU-only, no GPU.

Training Hub exports of VLM text backbones (e.g. Qwen3.5) carry a bare
text config that stock vLLM cannot serve; those are handled through
serve_vllm.py + patch_model_config.py (delivered as config files), same
as the removed standalone serve jobs did. lora_sft exports carry only
the PEFT adapter; merge_lora.py merges it into the base model before
serving.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from typing import Any

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.core.model_catalog import inject_enabled_provider_keys, resolve_provider
from amortized.jobs.base import JobBuildError, JobBuildResult
from amortized.jobs.common import resolve_training_model, set_mlflow_run_tag

logger = logging.getLogger("amortized.jobs.eval")

_ENDPOINT_KEYS = ("endpoint", "endpoint_base", "endpoint_tuned", "judge")

DEFAULT_SERVE_PORT = 8000

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Model loading inside the eval pod can take several minutes on a shared
# GPU; the health-wait loop gives up after this many seconds.
SERVE_STARTUP_TIMEOUT = 900


def _endpoint_spec(
    config: dict[str, Any], key: str, env_name: str, env: dict[str, str]
) -> dict[str, str]:
    endpoint = config.get(key)
    if not isinstance(endpoint, dict) or not endpoint.get("base_url") or not endpoint.get("model"):
        raise JobBuildError(f"{key}: base_url and model are required")
    base_url = str(endpoint["base_url"])
    api_key = endpoint.get("api_key", "")
    # A base_url that names an enabled provider (e.g. "openai") resolves to that
    # provider's real endpoint + its injected key env-var — the same catalog SDG
    # uses — so a BYOK-stripped judge/endpoint still authenticates with the
    # forwarded provider key (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...). The
    # provider_type (openai|anthropic) selects the runner's calling paradigm so the
    # judge supports exactly what the SDG teacher does (incl. native Anthropic).
    provider_type = "openai"
    provider = resolve_provider(base_url)
    if provider:
        base_url = provider.get("endpoint", base_url)
        provider_type = provider.get("provider_type") or "openai"
        if not api_key:
            env_name = provider.get("api_key", env_name)
    if api_key:
        env[env_name] = api_key
    return {
        "base_url": base_url.rstrip("/"),
        "model": str(endpoint["model"]),
        "api_key_env": env_name,
        "provider_type": provider_type,
    }


def _resolve_named_model(config: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Resolve a model by HF hub id or local path (vLLM downloads if needed)."""
    model_name = str(config.get("model_name_or_path", "")).strip()
    if not model_name:
        raise JobBuildError(
            "eval jobs require one of: training_job_id (tuned model),"
            " model_name_or_path (HF model id), or endpoint (OpenAI-compatible URL)"
        )
    served_name = str(config.get("served_model_name", "")).strip() or model_name
    return served_name, model_name, []


async def _build_embedded_serving(
    config: dict[str, Any],
    env: dict[str, str],
) -> tuple[list[str], str, dict[str, Any]]:
    """Prepare embedded serving for the model named in the config.

    Returns (pre_commands, main_script, resolved_extras) where
    resolved_extras carries config fields to persist (served name, gpu
    count, utilization).
    """
    if config.get("training_job_id"):
        served_name, model_path, pre_commands = await resolve_training_model(config)
    else:
        served_name, model_path, pre_commands = _resolve_named_model(config)

    # Same GPU budget scheme as training jobs: the pod requests
    # nvidia.com/gpu and the namespace ResourceQuota bounds it. GPUs are
    # assumed free, so nothing is pinned or occupancy-checked here.
    gpus = int(config.get("nproc_per_node", 1))

    # $SERVE_MODEL_DIR is our own pre-command variable (find output, no
    # spaces) — shlex-quoting it would suppress expansion, so that exact
    # sentinel goes in raw. Everything else is user-provided and stays
    # quoted: a value starting with '$' (e.g. "$(rm -rf ~)") would
    # otherwise run as command substitution in the pod's shell.
    quoted_path = (
        model_path
        if model_path == "$SERVE_MODEL_DIR"
        else shlex.quote(model_path)
    )

    import contextlib as _cl

    explicit_util: float | None = None
    vllm_args = [str(a) for a in (config.get("vllm_args") or [])]
    for idx, extra in enumerate(vllm_args):
        if extra == "--gpu-memory-utilization" and idx + 1 < len(vllm_args):
            # Separate-token form: --gpu-memory-utilization 0.5
            with _cl.suppress(ValueError):
                explicit_util = float(vllm_args[idx + 1])
        elif extra.startswith("--gpu-memory-utilization"):
            # Equals form: --gpu-memory-utilization=0.5
            with _cl.suppress(ValueError):
                explicit_util = float(extra.split("=", 1)[-1].split()[-1])

    gpu_memory_utilization = explicit_util if explicit_util is not None else 0.9
    pre_commands.append(
        "python3 /amortized/check_gpu_memory.py"
        f" {quoted_path} {int(gpus)} {gpu_memory_utilization}"
    )

    port = int(config.get("port", DEFAULT_SERVE_PORT))
    serve_cmd = f"python3 /amortized/serve_vllm.py serve {quoted_path}"
    serve_cmd += f" --served-model-name {shlex.quote(served_name)}"
    serve_cmd += f" --port {int(port)}"
    serve_cmd += f" --gpu-memory-utilization {gpu_memory_utilization}"
    for extra in config.get("vllm_args", []) or []:
        if extra:
            serve_cmd += f" {shlex.quote(str(extra))}"

    # The eval stages are visible in the job logs (and the studio monitor
    # card parses the EVAL-STAGE markers to show where the job is). The
    # script is a sh brace group (see build) so a failed pre-command
    # aborts all of it, and it never ends in a bare `exit` so the
    # worker-appended post-command (results upload) runs on success.
    script = "\n".join(
        [
            'echo "=== EVAL-STAGE: serving ==="',
            f"{serve_cmd} &",
            "SERVE_PID=$!",
            'echo "=== EVAL-STAGE: waiting-for-endpoint ==="',
            # Fails fast when the server process dies (OOM) instead of
            # polling the full timeout.
            f"python3 /amortized/wait_for_endpoint.py {int(port)} $SERVE_PID"
            f" {SERVE_STARTUP_TIMEOUT}",
            "WAIT_RC=$?",
            'if [ "$WAIT_RC" -ne 0 ]; then',
            '  echo "=== EVAL-STAGE: serve-failed ==="',
            "  kill $SERVE_PID 2>/dev/null || true",
            "  wait $SERVE_PID 2>/dev/null || true",
            '  echo "eval aborted: the model server never became healthy"',
            "  exit 1",
            "fi",
            'echo "=== EVAL-STAGE: evaluating ==="',
            "python3 /app/run_eval.py --config /amortized/config.json",
            "EVAL_RC=$?",
            "kill $SERVE_PID 2>/dev/null || true",
            "wait $SERVE_PID 2>/dev/null || true",
            'if [ "$EVAL_RC" -ne 0 ]; then exit 1; fi',
        ]
    )

    extras = {
        "served_model_name": served_name,
        "port": port,
        "gpu_memory_utilization": gpu_memory_utilization,
        "gpus": gpus,
    }
    return pre_commands, script, extras


async def _classification_training_run_id(training_job_id: str) -> str:
    """Resolve the MLflow run id of a completed training job."""
    from amortized.db.connection import get_pool
    from amortized.db.repository import Repository
    from amortized.models import JobStatus

    async with get_pool().acquire() as conn:
        parent = await Repository(conn).get_job(training_job_id)
    if not parent:
        raise JobBuildError(f"training_job_id {training_job_id!r} not found")
    if parent.get("type") != "training":
        raise JobBuildError(f"training_job_id {training_job_id!r} is not a training job")
    status = str(parent.get("status", "") or "")
    if status != JobStatus.succeeded.value:
        raise JobBuildError(
            f"training job {training_job_id[:8]} has not succeeded (status: "
            f"{status or 'unknown'}) — cannot evaluate its model"
        )
    run_id = str(parent.get("mlflow_run_id", "") or "")
    if not run_id:
        raise JobBuildError(f"training job {training_job_id[:8]} has no MLflow run")
    return run_id


async def _build_classification_eval(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    """Evaluate an embedding classifier/router — no serving, no judge.

    Loads the tuned sentence-transformers model directly, splits the held-out
    labeled dataset into per-class anchors + query examples, routes each query
    to its nearest class, and reports accuracy / macro-F1 / confusion. Runs in
    the training image (which ships sentence-transformers).
    """
    # Training image has sentence-transformers + datasets + training_hub.
    from amortized.jobs.training import IMAGE as TRAINING_IMAGE

    pre_commands: list[str] = []

    # 1. Model to evaluate: a tuned model (training_job_id) or a base ST model.
    training_job_id = str(config.get("training_job_id", "")).strip()
    model_name_or_path = str(config.get("model_name_or_path", "")).strip()
    if training_job_id:
        run_id = await _classification_training_run_id(training_job_id)
        local_dir = "/amortized/work/eval_model"
        pre_commands.append(
            f"mlflow artifacts download -r {shlex.quote(run_id)}"
            f" -a model -d {shlex.quote(local_dir)}"
        )
        model_path = f"{local_dir}/model"
    elif model_name_or_path:
        model_path = model_name_or_path  # base model baseline (HF id / local path)
    else:
        raise JobBuildError(
            "classification eval requires training_job_id (tuned model) or"
            " model_name_or_path (base model baseline)"
        )

    # 2. Held-out labeled dataset. resolve_parent_artifacts already sets
    #    eval_data_path for a parent sdg/upload job; else download by run id.
    eval_data_path = config.get("eval_data_path", "")
    eval_data_run_id = str(config.get("eval_data_run_id", "") or "")
    if not eval_data_path and eval_data_run_id:
        local_dir = "/amortized/work/eval_data"
        pre_commands.append(
            f"mlflow artifacts download -r {shlex.quote(eval_data_run_id)}"
            f" -a generated_data -d {shlex.quote(local_dir)}"
        )
        eval_data_path = f"{local_dir}/generated_data"
    if not eval_data_path:
        raise JobBuildError(
            "classification eval requires parent_job_id (a completed SDG/upload"
            " job with a labeled dataset) or eval_data_run_id"
        )

    runner_config = {
        "model_path": model_path,
        "eval_data_path": eval_data_path,
        "text_column": config.get("text_column", "text"),
        "label_column": config.get("label_column", "label"),
        "class_labels": config.get("class_labels"),
        "anchors_per_class": int(config.get("anchors_per_class", 16)),
        "top_k": int(config.get("top_k", 3)),
        "tau": float(config.get("tau", 0.0)),
        "seed": int(config.get("seed", 42)),
        "output_dir": "/amortized/work/results",
    }
    config_files["config.json"] = json.dumps(runner_config)
    with open(os.path.join(_ASSETS_DIR, "classify_eval.py")) as f:
        config_files["classify_eval.py"] = f.read()

    post_cmd = (
        "mlflow artifacts log-artifacts"
        " -l /amortized/work/results"
        " -r $MLFLOW_RUN_ID"
        " -a eval_results"
    )

    resolved_config = dict(config)
    resolved_config["eval_data_path"] = eval_data_path
    resolved_config["model_path"] = model_path

    return JobBuildResult(
        command=["python3", "/amortized/classify_eval.py", "--config", "/amortized/config.json"],
        config_files=config_files,
        pre_commands=pre_commands,
        post_commands=[post_cmd],
        # Encoding a small eval set is fast; GPU optional (0 = CPU, no quota).
        resources=Resources(gpus=int(config.get("nproc_per_node", 0)), cpus=4, memory_gb=8),
        image=TRAINING_IMAGE,
        resolved_config=resolved_config,
    )


async def build(
    job: dict[str, Any],
    config: dict[str, Any],
    config_files: dict[str, str],
) -> JobBuildResult:
    if config.get("eval_mode") == "classification":
        return await _build_classification_eval(job, config, config_files)

    env: dict[str, str] = {}
    # Forward every configured provider key into the job (same as SDG), so the
    # judge (and any external endpoint) can authenticate with OPENAI_API_KEY /
    # ANTHROPIC_API_KEY / ... regardless of which provider it uses.
    inject_enabled_provider_keys(env)

    # Eval job image — the tag is configurable (settings.eval_image_tag, default "latest") so a
    # specific commit can be pinned to test or roll back the eval backend without moving :latest.
    image = f"{config_mod.settings.image_registry}/eval:{config_mod.settings.eval_image_tag}"

    # --- Model under evaluation: embedded serving or external endpoint ---
    has_model_source = bool(
        str(config.get("training_job_id", "")).strip()
        or str(config.get("model_name_or_path", "")).strip()
    )
    embedded = has_model_source and not config.get("endpoint")

    if embedded:
        pre_commands, main_script, extras = await _build_embedded_serving(config, env)
        model_endpoint = {
            "base_url": f"http://localhost:{extras['port']}/v1",
            "model": extras["served_model_name"],
            "api_key_env": "EVAL_MODEL_API_KEY",
            # Embedded vLLM is OpenAI-compatible.
            "provider_type": "openai",
        }
        # GPU budget like a training job: the pod requests nvidia.com/gpu
        # and the namespace ResourceQuota bounds it.
        resources = Resources(gpus=int(extras.get("gpus", 1)), cpus=4, memory_gb=16)
        # The brace group makes the multi-line script a single compound
        # command — without it the worker's "pre && main" chain would only
        # guard the first line and a failed pre-check would not stop the
        # serve/eval lines below it.
        command: list[str] = ["sh", "-c", "{\n" + main_script + "\n}"]
        # serve_vllm.py / patch_model_config.py / merge_lora.py /
        # check_gpu_memory.py are shipped as config files (mounted at
        # /amortized) — same mechanism the standalone serve jobs used.
        for asset in (
            "serve_vllm.py",
            "patch_model_config.py",
            "merge_lora.py",
            "check_gpu_memory.py",
            "wait_for_endpoint.py",
        ):
            with open(os.path.join(_ASSETS_DIR, asset)) as f:
                config_files[asset] = f.read()
    else:
        model_key = (
            "endpoint"
            if config.get("endpoint")
            else "endpoint_tuned"
            if config.get("endpoint_tuned")
            else "endpoint_base"
        )
        model_endpoint = _endpoint_spec(config, model_key, "EVAL_MODEL_API_KEY", env)
        pre_commands = []
        extras = {}
        resources = Resources(gpus=0, cpus=2, memory_gb=4)
        command = ["python3", "/app/run_eval.py", "--config", "/amortized/config.json"]

    endpoints = {"model": model_endpoint}

    rubric = [
        c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")
    ]
    judge_cfg = config.get("judge")
    if bool(rubric) and not judge_cfg:
        raise JobBuildError(
            "a judge endpoint is required to score a rubric — the eval config"
            " must set `judge` explicitly (there is no default judge)"
        )
    if judge_cfg:
        endpoints["judge"] = _endpoint_spec(
            {"judge": judge_cfg}, "judge", "EVAL_JUDGE_API_KEY", env
        )

    eval_data_path = config.get("eval_data_path", "")
    eval_data_run_id = config.get("eval_data_run_id", "")
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
        "rubric": rubric,
        "max_samples": config.get("max_samples", 0),
        "judge_max_samples": config.get("judge_max_samples", 0),
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
    resolved_config.update(extras)

    return JobBuildResult(
        command=command,
        config_files=config_files,
        env=env,
        pre_commands=pre_commands,
        post_commands=[post_cmd],
        resources=resources,
        image=image,
        resolved_config=resolved_config,
    )


async def on_success(job: dict[str, Any], mlflow_run_id: str) -> None:
    job_config = job.get("config", {})
    if isinstance(job_config, str):
        job_config = json.loads(job_config)

    model_name = (
        job_config.get("served_model_name")
        or (job_config.get("endpoint") or {}).get("model", "")
        or (job_config.get("endpoint_tuned") or {}).get("model", "")
        or (job_config.get("endpoint_base") or {}).get("model", "")
    )
    if model_name:
        await set_mlflow_run_tag(mlflow_run_id, "eval_model", model_name)

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

        model_metrics = metrics.get("model", {})
        for criterion, score in (metrics.get("scores") or {}).items():
            if score is not None:
                await set_mlflow_run_tag(mlflow_run_id, f"eval_score_{criterion}", str(score))
        num_samples = model_metrics.get("num_samples")
        if num_samples is not None:
            await set_mlflow_run_tag(mlflow_run_id, "num_samples", str(num_samples))
        # Transparency: how many samples the judge actually scored and
        # which model judged — a partial judge run otherwise looks like a
        # full one in the comparison table.
        num_scored = metrics.get("num_scored")
        if num_scored is not None:
            await set_mlflow_run_tag(mlflow_run_id, "num_scored", str(num_scored))
        judge = (job_config.get("judge") or {})
        judge_model = str(judge.get("model") or "")
        if judge_model:
            await set_mlflow_run_tag(mlflow_run_id, "judge_model", judge_model)
    except Exception:
        logger.debug("Failed to tag eval metrics on run %s", mlflow_run_id, exc_info=True)
