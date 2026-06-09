"""Background worker that picks up queued jobs and runs them via ComputeBackend."""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

import amortized.config as config_mod
from amortized.backends import BackendHandle, BackendStatus, Capability, JobSpec
from amortized.core.artifacts import register_artifacts_for_job, register_log_artifacts
from amortized.core.compute import MissingCapabilityError, check_capabilities, get_backend
from amortized.core.events import emit_event
from amortized.db.repository import Repository, _row_to_job
from amortized.models import JobStatus, JobType

logger = logging.getLogger("amortized.worker")

# Keep references to background monitor tasks so they aren't garbage-collected
_monitor_tasks: set[asyncio.Task[None]] = set()

_JOB_TYPE_IMAGES: dict[str, str] = {
    "training": "ghcr.io/amortized-ai/training:latest",
    "sdg": "ghcr.io/amortized-ai/synth:latest",
    "inference": "ghcr.io/amortized-ai/inference:latest",
    "eval": "ghcr.io/amortized-ai/eval:latest",
}

_ALGORITHM_IMAGES: dict[str, str] = {
    "lora_sft": "ghcr.io/amortized-ai/training-lora:latest",
    "full_sft": "ghcr.io/amortized-ai/training-sft:latest",
    "sft": "ghcr.io/amortized-ai/training-sft:latest",
    "grpo": "ghcr.io/amortized-ai/training-grpo:latest",
    "lora_grpo": "ghcr.io/amortized-ai/training-grpo:latest",
    "gepa": "ghcr.io/amortized-ai/training-gepa:latest",
    "osft": "ghcr.io/amortized-ai/training-osft:latest",
}

_RUNNER_MODULES: dict[str, str] = {
    JobType.training.value: "amortized.runners.training_runner",
    JobType.sdg.value: "amortized.runners.sdg_runner",
    JobType.inference.value: "amortized.runners.inference_runner",
    JobType.eval.value: "amortized.runners.eval_runner",
}


async def _get_db() -> aiosqlite.Connection:
    """Open a standalone database connection for the worker."""
    config_mod.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(config_mod.settings.db_path))
    db.row_factory = aiosqlite.Row
    return db


def _build_runner_command(job: dict[str, Any]) -> list[str]:
    """Build the subprocess command for a job."""
    config = job["config"]
    job_type = job["type"]

    module = _RUNNER_MODULES.get(job_type)
    if module is None:
        raise ValueError(f"No runner module for job type: {job_type}")

    if job_type == JobType.sdg.value:
        if "output_dir" not in config and job.get("output_dir"):
            config = {**config, "output_dir": job["output_dir"]}
        elif "output_dir" not in config:
            output_dir = str(config_mod.settings.data_dir / "sdg_output" / job["id"])
            config = {**config, "output_dir": output_dir}
    elif job_type == JobType.eval.value:
        if "output_dir" not in config and job.get("output_dir"):
            config = {**config, "output_dir": job["output_dir"]}
        elif "output_dir" not in config:
            output_dir = str(config_mod.settings.data_dir / "eval_output" / job["id"])
            config = {**config, "output_dir": output_dir}

    return [sys.executable, "-m", module, json.dumps(config)]


def _serialize_handle(handle: BackendHandle) -> str:
    return json.dumps(
        {
            "backend_name": handle.backend_name,
            "job_id": handle.job_id,
            "remote_pid": handle.remote_pid,
            "remote_dir": handle.remote_dir,
            "container_id": handle.container_id,
            "scheduler_id": handle.scheduler_id,
        }
    )


def _deserialize_handle(raw: str | None) -> BackendHandle | None:
    if not raw:
        return None
    d = json.loads(raw)
    return BackendHandle(
        backend_name=d["backend_name"],
        job_id=d["job_id"],
        remote_pid=d.get("remote_pid"),
        remote_dir=d.get("remote_dir"),
        container_id=d.get("container_id"),
        scheduler_id=d.get("scheduler_id"),
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


async def _register_artifacts_for_job(job_id: str, output_dir: str) -> None:
    """Scan output directory and register found artifacts via core layer."""
    db = await _get_db()
    try:
        repo = Repository(db)
        await register_artifacts_for_job(repo, job_id, output_dir)
    finally:
        await db.close()


async def _register_log_artifacts(job_id: str, output_dir: str) -> None:
    """Register stdout.log and stderr.log as log-type artifacts via core layer."""
    db = await _get_db()
    try:
        repo = Repository(db)
        await register_log_artifacts(repo, job_id, output_dir)
    finally:
        await db.close()


async def _fetch_remote_outputs(handle: BackendHandle, output_dir: str) -> None:
    """Download job outputs from a remote node to the local output directory via SFTP."""
    backend = get_backend(handle.backend_name)
    conn = await backend._connect()
    try:
        async with conn.start_sftp_client() as sftp:
            remote_dir = handle.remote_dir
            # Resolve ~ to absolute path on the remote node
            if remote_dir and remote_dir.startswith("~"):
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


async def _run_job(job: dict[str, Any]) -> None:
    """Dispatch a job via ComputeBackend and poll until completion."""
    job_id = job["id"]
    now = datetime.now(UTC).isoformat()

    output_dir_names = {
        JobType.training.value: "training_output",
        JobType.sdg.value: "sdg_output",
        JobType.inference.value: "inference_output",
        JobType.eval.value: "eval_output",
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
    if job["type"] == JobType.training.value:
        config = {**config, "ckpt_output_dir": output_dir}
    elif job["type"] == JobType.inference.value:
        if "output_path" not in config or not config["output_path"]:
            config = {**config, "output_path": os.path.join(output_dir, "results.jsonl")}
    elif "output_dir" not in config:
        config = {**config, "output_dir": output_dir}

    path_keys = {
        "data_path",
        "ckpt_output_dir",
        "output_dir",
        "output_path",
        "resume_from_checkpoint",
    }
    for key in path_keys:
        if key in config and isinstance(config[key], str):
            config = {**config, key: os.path.expanduser(config[key])}

    cmd = _build_runner_command({**job, "config": config, "output_dir": output_dir})

    backend_name = config_mod.settings.default_backend or "local"
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
    if job["type"] in (JobType.sdg.value, JobType.eval.value) and not config.get("api_key"):
        model = config.get("model", "")
        provider = model.split("/")[0] if "/" in model else ""
        if provider:
            env_var = f"{provider.upper()}_API_KEY"
            key_db = await _get_db()
            try:
                key_repo = Repository(key_db)
                key_row = await key_repo.get_api_key_for_provider(provider)
                if key_row:
                    spec_env[env_var] = key_row["key_value"]
            finally:
                await key_db.close()

    image = _JOB_TYPE_IMAGES.get(job["type"])
    if job["type"] == JobType.training.value:
        algorithm = config.get("algorithm", "lora_sft")
        image = _ALGORITHM_IMAGES.get(algorithm, image)

    spec_env["_config"] = json.dumps(config)

    spec = JobSpec(
        job_id=job_id,
        command=cmd,
        env=spec_env,
        work_dir=output_dir,
        image=image,
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

        poll_interval = 2.0
        while True:
            status = await backend.status(handle)
            if not status.running:
                break
            await asyncio.sleep(poll_interval)

        completed_at = datetime.now(UTC).isoformat()

        if handle.remote_dir and handle.backend_name != "local":
            try:
                await _fetch_remote_outputs(handle, output_dir)
            except Exception:
                logger.warning("Failed to fetch remote outputs for job %s", job_id, exc_info=True)

        await _register_log_artifacts(job_id, output_dir)

        if status.exit_code == 0:
            await _update_job(
                job_id,
                status=JobStatus.succeeded,
                completed_at=completed_at,
            )
            await _register_artifacts_for_job(job_id, output_dir)
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
                    JobType.inference.value: "inference_output",
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
