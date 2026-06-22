"""Background worker that picks up queued jobs and runs them via ComputeBackend."""

import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

import amortized.config as config_mod
from amortized.backends import BackendHandle, BackendStatus, Capability, JobSpec
from amortized.core.compute import MissingCapabilityError, check_capabilities, get_backend
from amortized.core.events import emit_event
from amortized.db.repository import Repository, _row_to_job
from amortized.models import JobStatus, JobType

logger = logging.getLogger("amortized.worker")

# Keep references to background monitor tasks so they aren't garbage-collected
_monitor_tasks: set[asyncio.Task[None]] = set()

_JOB_TYPE_IMAGES: dict[str, str] = {
    "training": "ghcr.io/amortized-ai/trl:1.5.0",
    "sdg": "ghcr.io/amortized-ai/asynth:latest",
    "eval": "ghcr.io/amortized-ai/asynth:latest",
    "serve": "docker.io/vllm/vllm-openai",
}

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


async def _get_db() -> aiosqlite.Connection:
    """Open a standalone database connection for the worker."""
    config_mod.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(config_mod.settings.db_path))
    db.row_factory = aiosqlite.Row
    return db


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

_SCRIPT_ALGO_TRAINERS: dict[str, tuple[str, str]] = {
    "sft": ("SFTTrainer", "SFTConfig"),
    "gkd": ("GKDTrainer", "GKDConfig"),
    "dpo": ("DPOTrainer", "DPOConfig"),
    "kto": ("KTOTrainer", "KTOConfig"),
    "grpo": ("GRPOTrainer", "GRPOConfig"),
}


def _trl_config_yaml(algorithm: str, config: dict[str, Any]) -> str:
    """Translate amortized training config -> TRL CLI YAML config."""
    import yaml

    trl_config: dict[str, Any] = {
        "model_name_or_path": config.get("model_name_or_path", config.get("model_path", "")),
        "output_dir": config.get("output_dir", "/amortized/work/output"),
        "report_to": config.get("report_to", "none"),
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

    # GKD needs teacher model
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
    import yaml

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
    if job_type == JobType.sdg.value:
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
    import yaml

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


def _serialize_handle(handle: BackendHandle) -> str:
    return json.dumps(
        {
            "backend_name": handle.backend_name,
            "job_id": handle.job_id,
            "remote_pid": handle.remote_pid,
            "remote_dir": handle.remote_dir,
            "container_id": handle.container_id,
            "scheduler_id": handle.scheduler_id,
            "secret_names": handle.secret_names,
        }
    )


def _deserialize_handle(raw: str | None) -> BackendHandle | None:
    if not raw:
        return None
    d = json.loads(raw)
    raw_secrets = d.get("secret_names")
    secret_names = [tuple(s) for s in raw_secrets] if raw_secrets else None
    return BackendHandle(
        backend_name=d["backend_name"],
        job_id=d["job_id"],
        remote_pid=d.get("remote_pid"),
        remote_dir=d.get("remote_dir"),
        container_id=d.get("container_id"),
        scheduler_id=d.get("scheduler_id"),
        secret_names=secret_names,
    )


async def _update_job(
    job_id: str,
    *,
    status: JobStatus,
    started_at: str | None = None,
    completed_at: str | None = None,
    error: str | None = None,
    pid: int | None = None,
    backend_handle: str | None = None,
    mlflow_run_id: str | None = None,
) -> None:
    """Update job status in the database and emit a state_change event."""
    db = await _get_db()
    try:
        now = datetime.now(UTC).isoformat()
        fields = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status.value, now]

        if started_at is not None:
            fields.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            fields.append("completed_at = ?")
            params.append(completed_at)
        if error is not None:
            fields.append("error = ?")
            params.append(error)
        if pid is not None:
            fields.append("pid = ?")
            params.append(pid)
        if backend_handle is not None:
            fields.append("backend_handle = ?")
            params.append(backend_handle)
        if mlflow_run_id is not None:
            fields.append("mlflow_run_id = ?")
            params.append(mlflow_run_id)

        params.append(job_id)
        await db.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        await db.commit()

        repo = Repository(db)
        event_data: dict[str, Any] = {"status": status.value}
        if error is not None:
            event_data["error"] = error
        await emit_event(repo, job_id, "state_change", event_data)
    finally:
        await db.close()


async def _pick_pending_job() -> dict[str, Any] | None:
    """Pick the oldest queued job from the database."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
            (JobStatus.queued.value,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_job(row)
    finally:
        await db.close()


async def _fetch_remote_outputs(handle: BackendHandle, output_dir: str) -> None:
    """Download job outputs from a remote node to the local output directory via SFTP."""
    backend = get_backend(handle.backend_name)
    if not hasattr(backend, "_connect"):
        return
    conn = await backend._connect()
    try:
        async with conn.start_sftp_client() as sftp:
            remote_dir = handle.remote_dir or ""
            if not remote_dir:
                return
            if remote_dir.startswith("~"):
                try:
                    result = await conn.run("echo $HOME", check=True)
                    home = result.stdout.strip()
                    remote_dir = remote_dir.replace("~", home, 1)
                except Exception:
                    pass
            await _sftp_download_recursive(sftp, remote_dir, output_dir)
    finally:
        conn.close()


async def _sftp_download_recursive(sftp: Any, remote_path: str, local_path: str) -> None:
    """Recursively download all files from a remote directory via SFTP."""
    Path(local_path).mkdir(parents=True, exist_ok=True)
    entries = await sftp.listdir(remote_path)
    for name in entries:
        if name in (".", ".."):
            continue
        remote_full = f"{remote_path}/{name}"
        local_full = os.path.join(local_path, name)
        if await sftp.isdir(remote_full):
            await _sftp_download_recursive(sftp, remote_full, local_full)
        else:
            await sftp.get(remote_full, local_full)
            logger.debug("Fetched %s -> %s", remote_full, local_full)
    logger.info("Fetched %d entries from %s", len(entries), remote_path)


async def _resolve_artifact_refs(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``artifact:<id>`` references to file paths."""
    updated = dict(config)
    for key, value in config.items():
        if not isinstance(value, str) or not value.startswith("artifact:"):
            continue
        artifact_id = value[len("artifact:") :]
        db = await _get_db()
        try:
            cursor = await db.execute(
                "SELECT a.path, j.backend_handle, j.output_dir "
                "FROM artifacts a LEFT JOIN jobs j ON a.job_id = j.id "
                "WHERE a.id = ?",
                (artifact_id,),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
        if not row:
            logger.warning("Artifact %s not found", artifact_id)
            continue
        local_path, handle_json, local_output_dir = row
        # For remote backends, resolve to the remote path
        if handle_json and local_output_dir and local_path.startswith(local_output_dir):
            handle_data = json.loads(handle_json)
            remote_dir = handle_data.get("remote_dir", "")
            if remote_dir:
                rel = local_path[len(local_output_dir) :].lstrip("/")
                if remote_dir.startswith("~"):
                    backend = get_backend(handle_data.get("backend_name", ""))
                    if hasattr(backend, "_connect"):
                        conn = await backend._connect()
                        try:
                            result = await conn.run("echo $HOME", check=True)
                            remote_dir = remote_dir.replace("~", result.stdout.strip(), 1)
                        finally:
                            conn.close()
                updated[key] = f"{remote_dir}/{rel}"
                logger.info("Resolved %s -> %s (remote)", key, updated[key])
                continue
        updated[key] = local_path
        logger.info("Resolved %s -> %s (local)", key, local_path)
    return updated


async def _run_job(job: dict[str, Any]) -> None:
    """Dispatch a job via ComputeBackend and poll until completion."""
    job_id = job["id"]
    now = datetime.now(UTC).isoformat()

    output_dir_names = {
        JobType.training.value: "training_output",
        JobType.sdg.value: "sdg_output",
        JobType.eval.value: "eval_output",
        JobType.serve.value: "serve_output",
    }
    dir_name = output_dir_names.get(job["type"], f"{job['type']}_output")
    base_dir = job.get("output_dir") or str(config_mod.settings.data_dir / dir_name)
    output_dir = os.path.abspath(os.path.expanduser(os.path.join(base_dir, job_id)))

    db = await _get_db()
    try:
        await db.execute(
            "UPDATE jobs SET output_dir = ? WHERE id = ?",
            (output_dir, job_id),
        )
        await db.commit()
    finally:
        await db.close()

    config = job["config"]
    if job["type"] == JobType.training.value or "output_dir" not in config:
        config = {**config, "output_dir": output_dir}

    for key, value in list(config.items()):
        if isinstance(value, str) and value.startswith("~"):
            config = {**config, key: os.path.expanduser(value)}

    config = await _resolve_artifact_refs(config)

    if job["type"] == JobType.serve.value:
        port = int(config.get("port", 8000))
        spec_ports = {port: port}
    else:
        spec_ports = {}

    cmd: list[str] = []

    backend_name = config_mod.settings.resolved_default_backend
    if isinstance(job.get("metadata"), dict):
        backend_name = job["metadata"].get("backend", backend_name)

    try:
        backend = get_backend(backend_name)
    except KeyError:
        await _update_job(
            job_id,
            status=JobStatus.failed,
            completed_at=datetime.now(UTC).isoformat(),
            error=f"Unknown compute backend: {backend_name!r}",
        )
        return

    required_caps: set[Capability] = set()
    compute = job.get("metadata", {}) if isinstance(job.get("metadata"), dict) else {}
    gpus = compute.get("gpus", 0)
    if isinstance(gpus, int) and gpus > 0:
        required_caps.add(Capability.GPU)
    if config.get("resume_from_checkpoint"):
        required_caps.add(Capability.RESUME)

    if required_caps:
        try:
            check_capabilities(backend, required_caps)
        except MissingCapabilityError as exc:
            await _update_job(
                job_id,
                status=JobStatus.failed,
                completed_at=datetime.now(UTC).isoformat(),
                error=str(exc),
            )
            return

    spec_env: dict[str, str] = {}
    for env_name in config_mod.settings.forward_env:
        value = os.environ.get(env_name)
        if value:
            spec_env[env_name] = value

    _PROVIDER_ENV_MAP = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "huggingface": "HF_TOKEN",
        "openrouter": "OPENROUTER_API_KEY",
    }
    key_db = await _get_db()
    try:
        key_repo = Repository(key_db)
        for provider, env_name in _PROVIDER_ENV_MAP.items():
            if env_name not in spec_env:
                key_row = await key_repo.get_api_key_for_provider(provider)
                if key_row:
                    from amortized.core.crypto import decrypt_value

                    spec_env[env_name] = decrypt_value(key_row["key_value"])
    finally:
        await key_db.close()

    if config_mod.settings.mlflow_tracking_uri:
        spec_env["MLFLOW_TRACKING_URI"] = config_mod.settings.mlflow_tracking_uri
        spec_env["MLFLOW_EXPERIMENT_NAME"] = f"amortized/{job['type']}/{job_id[:8]}"
        if job["type"] == JobType.training.value:
            spec_env["HF_MLFLOW_LOG_ARTIFACTS"] = "true"

    image = _JOB_TYPE_IMAGES.get(job["type"])

    if job["type"] == JobType.eval.value:
        config = _resolve_judge_template(config)

    spec_env["_config"] = json.dumps(config)

    if image and job["type"] == JobType.training.value:
        algorithm = config.get("algorithm", "sft")
        trl_algo = _TRL_ALGO_MAP.get(algorithm)
        if trl_algo is None:
            raise ValueError(f"Unknown training algorithm: {algorithm}")
        data_path = config.get("data_path", config.get("dataset", ""))
        if data_path.startswith("s3://"):
            spec_env["_s3_data_path"] = data_path
        spec_env["_run_config"] = _trl_config_yaml(trl_algo, config)
        cmd = ["trl", trl_algo, "--config", "/amortized/config.yaml"]
    elif image and job["type"] == JobType.serve.value:
        spec_env["_run_config"] = _serve_config_yaml(config)
        if config.get("gpu_ids"):
            spec_env["CUDA_VISIBLE_DEVICES"] = str(config["gpu_ids"])
        cmd = ["--config", "/amortized/config.yaml"]
    elif image and job["type"] == JobType.sdg.value:
        import yaml

        s3_output = ""
        bucket = os.environ.get("AWS_S3_BUCKET") or config_mod.settings.storage_bucket
        if bucket:
            s3_output = f"s3://{bucket}/artifacts/{job_id}/output/generated_data.jsonl"

        synth_config = _generate_container_config(job["type"], config, s3_output_path=s3_output)
        spec_env["_synth_config"] = yaml.dump(synth_config, default_flow_style=False)
        cmd = ["asynth", "synthesize", "--config", "/amortized/synth_config.yaml", "--verbose"]
    elif image:
        spec_env["_run_config"] = _eval_config_yaml(config)
        cmd = ["asynth", "judge", "--config", "/amortized/config.yaml"]

    spec = JobSpec(
        job_id=job_id,
        command=cmd,
        env=spec_env,
        work_dir=output_dir,
        image=image,
        ports=spec_ports,
    )

    logger.info("Submitting job %s to backend %r", job_id, backend_name)

    try:
        handle = await backend.submit(spec)

        handle_json = _serialize_handle(handle)
        await _update_job(
            job_id,
            status=JobStatus.running,
            started_at=now,
            pid=handle.remote_pid,
            backend_handle=handle_json,
        )

        if job["type"] == JobType.serve.value:
            if backend_name != "kubernetes":
                task = asyncio.create_task(_monitor_serve_job(job_id, handle, backend))
                _monitor_tasks.add(task)
                task.add_done_callback(_monitor_tasks.discard)
            logger.info("Serve job %s started — monitoring in background", job_id)
            return

        poll_interval = 2.0
        while True:
            status = await backend.status(handle)
            if not status.running:
                break
            await asyncio.sleep(poll_interval)

        completed_at = datetime.now(UTC).isoformat()

        if handle.remote_dir and hasattr(backend, "_connect"):
            try:
                await _fetch_remote_outputs(handle, output_dir)
            except Exception:
                logger.warning("Failed to fetch remote outputs for job %s", job_id, exc_info=True)

        if handle.secret_names and hasattr(backend, "cleanup_secrets"):
            try:
                await backend.cleanup_secrets(handle)
            except Exception:
                logger.warning("Failed to clean up secrets for job %s", job_id, exc_info=True)

        if status.exit_code == 0:
            await _update_job(
                job_id,
                status=JobStatus.succeeded,
                completed_at=completed_at,
            )
            logger.info("Job %s succeeded", job_id)
        elif status.exit_code is not None and status.exit_code < 0:
            await _update_job(
                job_id,
                status=JobStatus.cancelled,
                completed_at=completed_at,
                error="Job was cancelled",
            )
            logger.info("Job %s was cancelled", job_id)
        else:
            stderr_output = ""
            stderr_path = os.path.join(output_dir, "stderr.log")
            try:
                with open(stderr_path) as f:
                    stderr_output = f.read()[-2000:]
            except OSError:
                pass
            error_msg = status.error or f"Process exited with code {status.exit_code}"
            if stderr_output:
                from amortized.core.redact import redact_text

                stderr_output = redact_text(stderr_output)
                error_msg = f"{error_msg}: {stderr_output}"
            await _update_job(
                job_id,
                status=JobStatus.failed,
                completed_at=completed_at,
                error=error_msg,
            )
            logger.error("Job %s failed with code %s", job_id, status.exit_code)

    except Exception as exc:
        await _update_job(
            job_id,
            status=JobStatus.failed,
            completed_at=datetime.now(UTC).isoformat(),
            error=str(exc),
        )
        logger.exception("Job %s failed with exception", job_id)


async def cancel_job_via_backend(job_id: str, handle_json: str | None) -> bool:
    """Cancel a job via its stored BackendHandle. Returns True if cancelled."""
    handle = _deserialize_handle(handle_json)
    if handle is None:
        return False
    try:
        backend = get_backend(handle.backend_name)
        await backend.cancel(handle)
        return True
    except (KeyError, OSError):
        return False


async def kill_job_process(pid: int, timeout: float = 5.0) -> bool:
    """Kill a job subprocess by PID. Fallback when no backend handle is stored."""
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    loop = asyncio.get_event_loop()
    start = loop.time()

    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return True

        elapsed = loop.time() - start
        if elapsed >= timeout:
            import contextlib

            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
            return True

        await asyncio.sleep(0.1)


async def cleanup_orphaned_jobs() -> None:
    """Handle 'running' jobs on startup: use backend handles first, fall back to PID checks."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT id, pid, type, output_dir, config, backend_handle FROM jobs WHERE status = ?",
            (JobStatus.running.value,),
        )
        rows = await cursor.fetchall()
        now = datetime.now(UTC).isoformat()

        for row in rows:
            job_id = row["id"]
            pid = row["pid"]
            job_type = row["type"]
            output_dir = row["output_dir"]
            handle_json = row["backend_handle"]

            if not output_dir:
                orphan_dir_names = {
                    JobType.training.value: "training_output",
                    JobType.sdg.value: "sdg_output",
                    JobType.eval.value: "eval_output",
                }
                dir_name = orphan_dir_names.get(job_type, f"{job_type}_output")
                output_dir = str(config_mod.settings.data_dir / dir_name / job_id)

            alive = False

            handle = _deserialize_handle(handle_json)
            if handle is not None:
                try:
                    backend = get_backend(handle.backend_name)
                    bs: BackendStatus = await backend.status(handle)
                    alive = bs.running
                except KeyError:
                    pass

            if not alive and pid is not None:
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    pass

            if alive:
                logger.info("Re-adopted running job %s (pid=%s)", job_id, pid)
            else:
                await db.execute(
                    """UPDATE jobs SET status = ?, updated_at = ?, completed_at = ?,
                       error = ? WHERE id = ?""",
                    (
                        JobStatus.failed.value,
                        now,
                        now,
                        "Orphaned job — process no longer running",
                        job_id,
                    ),
                )
                logger.warning("Marked orphaned job %s (pid=%s) as failed", job_id, pid)

        await db.commit()
    finally:
        await db.close()


async def _monitor_serve_job(
    job_id: str,
    handle: BackendHandle,
    backend: Any,
) -> None:
    """Background monitor for a long-running serve job.

    Polls the backend periodically. If the container dies unexpectedly,
    marks the job as failed. Normal shutdown happens via cancel.
    """
    poll_interval = 10.0
    while True:
        try:
            status = await backend.status(handle)
            if not status.running:
                completed_at = datetime.now(UTC).isoformat()
                if status.exit_code is not None and status.exit_code < 0:
                    await _update_job(
                        job_id,
                        status=JobStatus.cancelled,
                        completed_at=completed_at,
                        error="Serve container stopped",
                    )
                else:
                    error = status.error or f"Serve container exited with code {status.exit_code}"
                    await _update_job(
                        job_id,
                        status=JobStatus.failed,
                        completed_at=completed_at,
                        error=error,
                    )
                if handle.secret_names and hasattr(backend, "cleanup_secrets"):
                    try:
                        await backend.cleanup_secrets(handle)
                    except Exception:
                        logger.warning("Failed to clean up secrets for serve job %s", job_id)
                break
        except Exception:
            logger.warning("Error monitoring serve job %s", job_id, exc_info=True)
        await asyncio.sleep(poll_interval)


async def _monitor_heartbeats(poll_interval: float = 60.0, timeout: float = 300.0) -> None:
    """Check running jobs for stale heartbeats and probe backend on timeout."""
    while True:
        try:
            db = await _get_db()
            try:
                repo = Repository(db)
                running_jobs = await repo.list_jobs(status=JobStatus.running)
                for job in running_jobs:
                    latest_event = await repo.get_latest_event(job["id"])
                    if latest_event is None:
                        continue
                    ts = latest_event.get("timestamp", "")
                    try:
                        event_dt = datetime.fromisoformat(ts)
                        age = time.time() - event_dt.timestamp()
                    except (ValueError, TypeError):
                        continue
                    if age <= timeout:
                        continue
                    handle = _deserialize_handle(job.get("backend_handle"))
                    if handle is None:
                        continue
                    try:
                        backend = get_backend(handle.backend_name)
                        bs = await backend.status(handle)
                    except KeyError:
                        continue
                    if not bs.running:
                        now = datetime.now(UTC).isoformat()
                        error_msg = f"Process died silently (exit_code={bs.exit_code})"
                        await db.execute(
                            """UPDATE jobs SET status = ?, updated_at = ?, completed_at = ?,
                               error = ? WHERE id = ?""",
                            (JobStatus.failed.value, now, now, error_msg, job["id"]),
                        )
                        await db.commit()
                        await emit_event(
                            repo,
                            job["id"],
                            "state_change",
                            {"status": JobStatus.failed.value, "error": error_msg},
                        )
                        logger.warning(
                            "Heartbeat timeout: job %s marked failed (exit_code=%s)",
                            job["id"],
                            bs.exit_code,
                        )
            finally:
                await db.close()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Heartbeat monitor error")
        await asyncio.sleep(poll_interval)


async def worker_loop(poll_interval: float = 2.0) -> None:
    """Main worker loop — polls for queued jobs and runs them."""
    logger.info("Worker started (poll interval: %.1fs)", poll_interval)

    while True:
        try:
            job = await _pick_pending_job()
            if job is not None:
                await _run_job(job)
            else:
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            logger.info("Worker shutting down")
            break
        except Exception:
            logger.exception("Worker error — retrying in %ss", poll_interval)
            await asyncio.sleep(poll_interval)
