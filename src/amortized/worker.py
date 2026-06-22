"""Background worker that picks up queued jobs and runs them via ComputeBackend."""

import asyncio
import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

import amortized.config as config_mod
from amortized.backends import BackendHandle, BackendStatus, Capability, JobSpec, S3Download
from amortized.core.compute import MissingCapabilityError, check_capabilities, get_backend
from amortized.core.config_translator import (
    _TRL_ALGO_MAP,
    _eval_config_yaml,
    _generate_container_config,
    _resolve_judge_template,
    _serve_config_yaml,
    _trl_config_yaml,
)
from amortized.core.events import emit_event
from amortized.core.jobs import _deserialize_handle
from amortized.db.repository import Repository
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


async def _get_db() -> aiosqlite.Connection:
    """Open a standalone database connection for the worker."""
    config_mod.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(config_mod.settings.db_path))
    db.row_factory = aiosqlite.Row
    return db


async def _get_repo() -> tuple[aiosqlite.Connection, Repository]:
    db = await _get_db()
    return db, Repository(db)


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
    """Update job status via Repository and emit a state_change event."""
    db, repo = await _get_repo()
    try:
        now = datetime.now(UTC).isoformat()
        await repo.update_job_status(
            job_id,
            status=status,
            updated_at=now,
            started_at=started_at,
            completed_at=completed_at,
            error=error,
            pid=pid,
            backend_handle=backend_handle,
            mlflow_run_id=mlflow_run_id,
        )
        event_data: dict[str, Any] = {"status": status.value}
        if error is not None:
            event_data["error"] = error
        await emit_event(repo, job_id, "state_change", event_data)
    finally:
        await db.close()


async def _pick_pending_job() -> dict[str, Any] | None:
    """Pick the oldest queued job from the database."""
    db, repo = await _get_repo()
    try:
        return await repo.pick_pending_job()
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
        db, repo = await _get_repo()
        try:
            row = await repo.get_artifact_with_job_context(artifact_id)
        finally:
            await db.close()
        if not row:
            logger.warning("Artifact %s not found", artifact_id)
            continue
        local_path, handle_json, local_output_dir = (
            row["path"],
            row["backend_handle"],
            row["output_dir"],
        )
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


async def _get_training_job_for_serve(training_job_id: str) -> dict[str, Any]:
    """Look up a completed training job for serve model resolution."""
    db, repo = await _get_repo()
    try:
        job = await repo.get_job(training_job_id)
        if not job:
            raise ValueError(f"Training job not found: {training_job_id}")
        if job["status"] != "succeeded":
            raise ValueError(f"Training job has not succeeded (status: {job['status']})")
        return job
    finally:
        await db.close()


async def _resolve_mlflow_artifact_uri(mlflow_run_id: str) -> str:
    """Query MLflow for the artifact URI of a training run."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return ""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{tracking_uri}/api/2.0/mlflow/runs/get",
                params={"run_id": mlflow_run_id},
            )
            resp.raise_for_status()
            run = resp.json()["run"]
            return run["info"]["artifact_uri"]
    except Exception:
        logger.warning(
            "Failed to resolve MLflow artifact URI for run %s", mlflow_run_id, exc_info=True
        )
        return ""


async def _extract_mlflow_run_id(backend: Any, handle: BackendHandle) -> str:
    """Extract MLflow run ID from job logs if available."""
    try:
        log_lines: list[str] = []
        async for line in backend.logs(handle):
            log_lines.append(line)
            if len(log_lines) > 200:
                log_lines = log_lines[-200:]
        log_text = "\n".join(log_lines)
        match = re.search(r"/runs/([a-f0-9]{32})", log_text)
        return match.group(1) if match else ""
    except Exception:
        return ""


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

    db, repo = await _get_repo()
    try:
        now_for_dir = datetime.now(UTC).isoformat()
        await repo.update_job_status(
            job_id,
            status=JobStatus(job["status"]),
            updated_at=now_for_dir,
            output_dir=output_dir,
        )
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
    key_db, key_repo = await _get_repo()
    try:
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

    config_files: dict[str, str] = {
        "config.json": json.dumps({"config": config, "artifacts": {}}),
    }
    s3_downloads: list[S3Download] = []

    if image and job["type"] == JobType.training.value:
        algorithm = config.get("algorithm", "sft")
        trl_algo = _TRL_ALGO_MAP.get(algorithm)
        if trl_algo is None:
            raise ValueError(f"Unknown training algorithm: {algorithm}")
        data_path = config.get("data_path", config.get("dataset", ""))
        if data_path.startswith("s3://"):
            local_name = data_path.split("/")[-1]
            s3_downloads.append(
                S3Download(
                    s3_uri=data_path,
                    local_path=f"/amortized/work/{local_name}",
                )
            )
        config_files["config.yaml"] = _trl_config_yaml(trl_algo, config)
        cmd = ["trl", trl_algo, "--config", "/amortized/config.yaml"]
    elif image and job["type"] == JobType.serve.value:
        if config.get("training_job_id"):
            try:
                training_job = await _get_training_job_for_serve(config["training_job_id"])
                training_config = training_job.get("config", {})
                if isinstance(training_config, str):
                    training_config = json.loads(training_config)
                if not config.get("model_name_or_path"):
                    config["model_name_or_path"] = training_config.get("model_name_or_path", "")
                config["adapter_path"] = "/amortized/work/model"
                mlflow_run_id = training_job.get("mlflow_run_id", "")
                if mlflow_run_id:
                    artifact_uri = await _resolve_mlflow_artifact_uri(mlflow_run_id)
                    if artifact_uri:
                        s3_downloads.append(
                            S3Download(
                                s3_uri=artifact_uri,
                                local_path="/amortized/work/model",
                                is_directory=True,
                            )
                        )
                        logger.info(
                            "Resolved model from training job %s: %s",
                            config["training_job_id"],
                            artifact_uri,
                        )
            except ValueError as exc:
                await _update_job(
                    job_id,
                    status=JobStatus.failed,
                    completed_at=datetime.now(UTC).isoformat(),
                    error=str(exc),
                )
                return

        config_files["config.yaml"] = _serve_config_yaml(config)
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
        config_files["synth_config.yaml"] = yaml.dump(synth_config, default_flow_style=False)
        cmd = ["asynth", "synthesize", "--config", "/amortized/synth_config.yaml", "--verbose"]
    elif image:
        config_files["config.yaml"] = _eval_config_yaml(config)
        cmd = ["asynth", "judge", "--config", "/amortized/config.yaml"]

    spec = JobSpec(
        job_id=job_id,
        command=cmd,
        env=spec_env,
        work_dir=output_dir,
        image=image,
        ports=spec_ports,
        config_files=config_files,
        s3_downloads=s3_downloads,
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
            mlflow_run_id = ""
            if job["type"] == JobType.training.value:
                mlflow_run_id = await _extract_mlflow_run_id(backend, handle)
            await _update_job(
                job_id,
                status=JobStatus.succeeded,
                completed_at=completed_at,
                mlflow_run_id=mlflow_run_id,
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


async def cleanup_orphaned_jobs() -> None:
    """Handle 'running' jobs on startup: use backend handles first, fall back to PID checks."""
    db, repo = await _get_repo()
    try:
        running_jobs = await repo.list_jobs(status=JobStatus.running)
        now = datetime.now(UTC).isoformat()

        for job in running_jobs:
            job_id = job["id"]
            pid = job.get("pid")
            job_type = job["type"]
            output_dir = job.get("output_dir")
            handle_json = job.get("backend_handle")

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
                await repo.update_job_status(
                    job_id,
                    status=JobStatus.failed,
                    updated_at=now,
                    completed_at=now,
                    error="Orphaned job — process no longer running",
                )
                logger.warning("Marked orphaned job %s (pid=%s) as failed", job_id, pid)
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
            db, repo = await _get_repo()
            try:
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
                        await repo.update_job_status(
                            job["id"],
                            status=JobStatus.failed,
                            updated_at=now,
                            completed_at=now,
                            error=error_msg,
                        )
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
