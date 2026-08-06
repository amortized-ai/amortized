"""Background worker that picks up queued jobs and runs them via ComputeBackend."""

import asyncio
import json
import logging
import os
import re
import shlex
from datetime import UTC, datetime
from typing import Any

import amortized.config as config_mod
from amortized.backends import BackendHandle, Capability, JobSpec
from amortized.core.compute import MissingCapabilityError, check_capabilities, get_backend
from amortized.core.jobs import deserialize_handle
from amortized.db.repository import Repository
from amortized.jobs import get_builder
from amortized.jobs.base import JobBuildError
from amortized.jobs.common import (
    resolve_parent_artifacts as _resolve_parent_artifacts,
)
from amortized.jobs.common import (
    set_mlflow_run_tag,
)
from amortized.models import JobStatus, JobType
from amortized.watch import emit_job_event

logger = logging.getLogger("amortized.worker")

_background_tasks: set[asyncio.Task[None]] = set()


def _fire_event(job_id: str, status: str, job: dict[str, Any]) -> None:
    task = asyncio.create_task(_safe_emit(job_id, status, job))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _safe_emit(job_id: str, status: str, job: dict[str, Any]) -> None:
    try:
        await emit_job_event(job_id, status, job)
    except Exception:
        logger.warning("Failed to emit job event for %s", job_id, exc_info=True)


# ---------------------------------------------------------------------------
# Command wrapping
# ---------------------------------------------------------------------------


def _wrap_command(
    command: list[str],
    pre_commands: list[str],
    post_commands: list[str],
) -> list[str]:
    """Wrap a command with pre/post commands using shell chaining.

    Pre-commands use && (fail fast). Post-commands use ; (best-effort).
    """
    if not pre_commands and not post_commands:
        return command
    if command[:2] == ["sh", "-c"] and len(command) == 3:
        main_cmd = command[2]
    else:
        main_cmd = shlex.join(command)
    pre_chain = " && ".join([*pre_commands, main_cmd])
    if post_commands:
        post_chain = " ; ".join(post_commands)
        return ["sh", "-c", f"{pre_chain} && {{ {post_chain} ; true; }}"]
    return ["sh", "-c", pre_chain]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


async def _update_job(job_id: str, **kwargs: Any) -> None:
    from amortized.db.connection import get_pool

    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        await repo.update_job(job_id, **kwargs)


async def _pick_pending_job() -> dict[str, Any] | None:
    from amortized.db.connection import get_pool

    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        return await repo.pick_pending_job()


async def _resolve_mlflow_artifact_uri(mlflow_run_id: str) -> str:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return ""
    try:
        from amortized.core.mlflow_client import MLflowClient

        client = MLflowClient(tracking_uri)
        run = await client.get_run(mlflow_run_id)
        uri: str = run["info"]["artifact_uri"]
        return uri
    except Exception:
        logger.warning(
            "Failed to resolve MLflow artifact URI for run %s",
            mlflow_run_id,
            exc_info=True,
        )
        return ""


async def _extract_mlflow_run_id(backend: Any, handle: BackendHandle) -> str:
    try:
        log_lines: list[str] = []
        async for line in backend.logs(handle):
            log_lines.append(line)
            if len(log_lines) > 500:
                log_lines = log_lines[-500:]
        log_text = "\n".join(log_lines)
        explicit = re.search(r"AMORTIZED_MLFLOW_RUN_ID=([a-f0-9]{32})", log_text)
        if explicit:
            return explicit.group(1)
        match = re.search(r"/runs/([a-f0-9]{32})", log_text)
        return match.group(1) if match else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


async def _run_job(job: dict[str, Any]) -> None:
    job_id = job["id"]
    job_type = job["type"]
    now = datetime.now(UTC)
    config = dict(job["config"])
    config.pop("_secrets", None)

    backend_name = config_mod.settings.resolved_default_backend

    # --- Output directory ---
    output_dir_names = {
        JobType.training.value: "training_output",
        JobType.sdg.value: "sdg_output",
        JobType.upload.value: "upload_output",
    }
    dir_name = output_dir_names.get(job_type, f"{job_type}_output")
    base_dir = str(config_mod.settings.data_dir / dir_name)
    output_dir = os.path.abspath(os.path.expanduser(os.path.join(base_dir, job_id)))

    if job_type == JobType.training.value or "output_dir" not in config:
        config["output_dir"] = output_dir
    if backend_name == "kubernetes" and job_type == JobType.training.value:
        config["output_dir"] = "/amortized/work/output"

    for key, value in list(config.items()):
        if isinstance(value, str) and value.startswith("~"):
            config[key] = os.path.expanduser(value)

    # --- Backend ---
    try:
        backend = get_backend(backend_name)
    except KeyError:
        await _update_job(
            job_id,
            status=JobStatus.failed.value,
            completed_at=datetime.now(UTC),
            error=f"Unknown compute backend: {backend_name!r}",
        )
        return

    required_caps: set[Capability] = set()
    if required_caps:
        try:
            check_capabilities(backend, required_caps)
        except MissingCapabilityError as exc:
            await _update_job(
                job_id,
                status=JobStatus.failed.value,
                completed_at=datetime.now(UTC),
                error=f"Job '{job_id}' cannot run on backend '{backend_name}': {exc}",
            )
            return

    # --- Environment ---
    spec_env: dict[str, str] = {}
    for env_name in config_mod.settings.forward_env:
        value = os.environ.get(env_name)
        if value:
            spec_env[env_name] = value

    if config_mod.settings.mlflow_tracking_uri:
        mlflow_experiment = f"amortized/{job_type}/{job_id[:8]}"
        spec_env["MLFLOW_TRACKING_URI"] = config_mod.settings.mlflow_tracking_uri
        spec_env["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment
        s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
        if s3_endpoint:
            spec_env["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
            spec_env["FSSPEC_S3_ENDPOINT_URL"] = s3_endpoint
        for s3_var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_BUCKET"):
            val = os.environ.get(s3_var, "")
            if val:
                spec_env[s3_var] = val
        await _update_job(job_id, mlflow_experiment=mlflow_experiment)

    # --- Resolve parent artifacts ---
    config_files: dict[str, str] = {}
    config, parent_pre_commands = await _resolve_parent_artifacts(job, config)

    # --- Job-type-specific build ---
    builder = get_builder(job_type)
    if builder is None:
        await _update_job(
            job_id,
            status=JobStatus.failed.value,
            completed_at=datetime.now(UTC),
            error=f"Unsupported job type: {job_type!r}",
        )
        return

    try:
        result = await builder.build(job, config, config_files)
    except JobBuildError as exc:
        logger.error("Job %s: %s", job_id, exc)
        await _update_job(
            job_id,
            status=JobStatus.failed.value,
            completed_at=datetime.now(UTC),
            error=str(exc),
        )
        return

    # Merge builder env into spec_env
    spec_env.update(result.env)

    # Merge parent artifact pre_commands with builder pre_commands
    all_pre_commands = parent_pre_commands + result.pre_commands

    # Persist resolved config
    await _update_job(job_id, config=result.resolved_config)

    # --- Wrap command with pre/post commands ---
    final_command = _wrap_command(result.command, all_pre_commands, result.post_commands)

    # --- Submit ---
    spec = JobSpec(
        job_id=job_id,
        command=final_command,
        env=spec_env,
        work_dir=output_dir,
        image=result.image,
        config_files=result.config_files,
        job_type=job_type,
        user_id=job.get("user_id", ""),
        resources=result.resources,
    )

    logger.info("Submitting job %s to backend %r", job_id, backend_name)

    try:
        handle = await backend.submit(spec)
        handle_json = _serialize_handle(handle)

        k8s_job_name = handle.scheduler_id or ""
        await _update_job(
            job_id,
            status=JobStatus.provisioning.value,
            started_at=now,
            backend_handle=handle_json,
            k8s_job_name=k8s_job_name,
        )

        # --- Poll until completion ---
        poll_interval = 2.0
        transitioned_to_running = False
        while True:
            status = await backend.status(handle)
            if not status.running:
                break
            if not transitioned_to_running:
                await _update_job(job_id, status=JobStatus.running.value)
                transitioned_to_running = True
                _fire_event(job_id, "running", job)
            await asyncio.sleep(poll_interval)

        completed_at = datetime.now(UTC)

        if handle.secret_names and hasattr(backend, "cleanup_secrets"):
            try:
                await backend.cleanup_secrets(handle)
            except Exception:
                logger.warning("Failed to clean up secrets for job %s", job_id, exc_info=True)

        # --- Completion handling ---
        if status.exit_code == 0:
            mlflow_run_id = await _extract_mlflow_run_id(backend, handle)
            if mlflow_run_id:
                await set_mlflow_run_tag(mlflow_run_id, "job_type", job_type)
                await set_mlflow_run_tag(mlflow_run_id, "job_id", job_id)
                await builder.on_success(job, mlflow_run_id)

            await _update_job(
                job_id,
                status=JobStatus.succeeded.value,
                completed_at=completed_at,
                mlflow_run_id=mlflow_run_id,
            )
            logger.info("Job %s succeeded", job_id)
            _fire_event(job_id, "succeeded", job)
        elif status.exit_code is not None and status.exit_code < 0:
            await _update_job(
                job_id,
                status=JobStatus.cancelled.value,
                completed_at=completed_at,
                error="Job was cancelled",
            )
            logger.info("Job %s was cancelled", job_id)
            _fire_event(
                job_id,
                "cancelled",
                {**job, "error": "Job was cancelled"},
            )
        else:
            error_msg = status.error or (
                f"Job '{job_id}' failed on backend '{backend_name}'"
                f" with exit code {status.exit_code}. Check logs for details."
            )
            await _update_job(
                job_id,
                status=JobStatus.failed.value,
                completed_at=completed_at,
                error=error_msg,
            )
            logger.error("Job %s failed with code %s", job_id, status.exit_code)
            _fire_event(job_id, "failed", {**job, "error": error_msg})

    except Exception as exc:
        error_text = f"Job '{job_id}' failed during submission to backend '{backend_name}': {exc}"
        try:
            stderr_path = os.path.join(output_dir, "stderr.log")
            os.makedirs(output_dir, exist_ok=True)
            with open(stderr_path, "a") as f:
                f.write(f"[amortized] Job failed before starting: {error_text}\n")
            fallback_handle = _serialize_handle(
                BackendHandle(
                    backend_name=backend_name,
                    job_id=job_id,
                    remote_dir=output_dir,
                )
            )
        except Exception:
            fallback_handle = None
        await _update_job(
            job_id,
            status=JobStatus.failed.value,
            completed_at=datetime.now(UTC),
            error=error_text,
            backend_handle=fallback_handle,
        )
        logger.exception("Job %s failed with exception", job_id)
        _fire_event(job_id, "failed", {**job, "error": error_text})


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def cleanup_orphaned_jobs() -> None:
    from amortized.db.connection import get_pool

    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        running_jobs = await repo.list_jobs(status=JobStatus.running)

    now = datetime.now(UTC)

    for job in running_jobs:
        job_id = job["id"]
        handle_json = job.get("backend_handle")

        alive = False
        handle = deserialize_handle(handle_json)
        if handle is not None:
            try:
                backend = get_backend(handle.backend_name)
                bs = await backend.status(handle)
                alive = bs.running
            except KeyError:
                pass

        if alive:
            logger.info("Re-adopted running job %s", job_id)
        else:
            async with get_pool().acquire() as conn:
                result = await conn.execute(
                    """UPDATE jobs SET status = $1, completed_at = $2,
                       error = $3 WHERE id = $4 AND status = $5""",
                    JobStatus.failed.value,
                    now,
                    "Orphaned job — process no longer running",
                    job_id,
                    JobStatus.running.value,
                )
            if result == "UPDATE 1":
                logger.warning("Marked orphaned job %s as failed", job_id)


async def worker_loop(poll_interval: float = 2.0) -> None:
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
