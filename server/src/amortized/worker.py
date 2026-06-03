"""Background worker that picks up pending jobs and runs them as subprocesses."""

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

import amortized.config as config_mod
from amortized.core.artifacts import register_artifacts_for_job, register_log_artifacts
from amortized.core.events import emit_event
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType

logger = logging.getLogger("amortized.worker")

# Track the currently running subprocess so cancel can reach it
_current_process: subprocess.Popen[bytes] | None = None
_current_job_id: str | None = None

# Keep references to background monitor tasks so they aren't garbage-collected
_monitor_tasks: set[asyncio.Task[None]] = set()


def get_current_process() -> tuple[subprocess.Popen[bytes] | None, str | None]:
    """Return the currently running subprocess and its job ID."""
    return _current_process, _current_job_id


async def _get_db() -> aiosqlite.Connection:
    """Open a standalone database connection for the worker."""
    config_mod.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(config_mod.settings.db_path))
    db.row_factory = aiosqlite.Row
    return db


def _find_runner_pid(job_id: str) -> int | None:
    """Scan /proc for a running runner subprocess matching a job config containing job_id."""
    proc_path = Path("/proc")
    if not proc_path.exists():
        return None

    for entry in proc_path.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline_path = entry / "cmdline"
            cmdline = cmdline_path.read_bytes().decode("utf-8", errors="replace")
            # Runner commands contain the module name and job config as JSON
            if "amortized.runners." in cmdline and job_id in cmdline:
                return int(entry.name)
        except (OSError, ValueError):
            continue
    return None


async def _monitor_adopted_process(pid: int, job_id: str, output_dir: str) -> None:
    """Monitor an adopted process until it exits, then update job status."""
    global _current_process, _current_job_id

    logger.info("Monitoring adopted process pid=%d for job %s", pid, job_id)
    loop = asyncio.get_event_loop()

    def _wait_for_exit() -> int | None:
        """Poll until the process exits, return exit code or None."""
        while True:
            try:
                # Try to reap the process if we're the parent
                wpid, wstatus = os.waitpid(pid, os.WNOHANG)
                if wpid != 0:
                    if os.WIFEXITED(wstatus):
                        return os.WEXITSTATUS(wstatus)
                    return 1  # killed by signal
            except ChildProcessError:
                # Not our child — poll with kill(0)
                try:
                    os.kill(pid, 0)
                except OSError:
                    # Process is gone; we can't get the exit code
                    return None
                import time

                time.sleep(2)
                continue
            import time

            time.sleep(2)

    exit_code = await loop.run_in_executor(None, _wait_for_exit)

    _current_process = None
    _current_job_id = None

    completed_at = datetime.now(UTC).isoformat()

    if exit_code == 0:
        await _update_job(job_id, status=JobStatus.completed, completed_at=completed_at)
        await _register_artifacts_for_job(job_id, output_dir)
        logger.info("Adopted job %s completed successfully", job_id)
    elif exit_code is None:
        # Process disappeared, we don't know the exit code — assume success if artifacts exist
        output_path = Path(output_dir)
        has_artifacts = output_path.exists() and any(output_path.iterdir())
        if has_artifacts:
            await _update_job(job_id, status=JobStatus.completed, completed_at=completed_at)
            await _register_artifacts_for_job(job_id, output_dir)
            logger.info("Adopted job %s finished (artifacts found, assuming success)", job_id)
        else:
            await _update_job(
                job_id,
                status=JobStatus.failed,
                completed_at=completed_at,
                error="Adopted process exited with unknown status and no artifacts found",
            )
            logger.warning("Adopted job %s finished with unknown status", job_id)
    else:
        await _update_job(
            job_id,
            status=JobStatus.failed,
            completed_at=completed_at,
            error=f"Adopted process exited with code {exit_code}",
        )
        logger.error("Adopted job %s failed with code %d", job_id, exit_code)


async def cleanup_orphaned_jobs() -> None:
    """Handle 'running' jobs on startup: re-adopt live processes, fail dead ones."""
    global _current_process, _current_job_id

    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT id, pid, type, output_dir, config FROM jobs WHERE status = ?",
            (JobStatus.running.value,),
        )
        rows = await cursor.fetchall()
        now = datetime.now(UTC).isoformat()

        for row in rows:
            job_id = row["id"]
            pid = row["pid"]
            job_type = row["type"]
            output_dir = row["output_dir"]

            # Determine output_dir if not stored
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
            if pid is not None:
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False
            else:
                # No PID stored — try to find the process via /proc
                found_pid = _find_runner_pid(job_id)
                if found_pid is not None:
                    pid = found_pid
                    alive = True
                    await db.execute(
                        "UPDATE jobs SET pid = ?, updated_at = ? WHERE id = ?",
                        (pid, now, job_id),
                    )
                    logger.info(
                        "Found orphaned process pid=%d for job %s via /proc scan", pid, job_id
                    )

            if alive and pid is not None:
                # Re-adopt: set module-level tracking and spawn monitor
                _current_job_id = job_id
                _current_process = None  # We don't have a Popen object, but pid is tracked in DB
                logger.info("Re-adopted running job %s with pid %d", job_id, pid)
                task = asyncio.create_task(_monitor_adopted_process(pid, job_id, output_dir))
                _monitor_tasks.add(task)
                task.add_done_callback(_monitor_tasks.discard)
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


async def _pick_pending_job() -> dict[str, Any] | None:
    """Pick the oldest pending job from the database."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
            (JobStatus.pending.value,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["config"] = json.loads(d["config"]) if isinstance(d["config"], str) else d["config"]
        return d
    finally:
        await db.close()


async def _update_job(
    job_id: str,
    *,
    status: JobStatus,
    started_at: str | None = None,
    completed_at: str | None = None,
    error: str | None = None,
    pid: int | None = None,
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


async def _register_artifacts_for_job(job_id: str, output_dir: str) -> None:
    """Scan output directory and register found artifacts via core layer."""
    db = await _get_db()
    try:
        repo = Repository(db)
        await register_artifacts_for_job(repo, job_id, output_dir)
    finally:
        await db.close()


_RUNNER_MODULES: dict[str, str] = {
    JobType.training.value: "amortized.runners.training_runner",
    JobType.sdg.value: "amortized.runners.sdg_runner",
    JobType.inference.value: "amortized.runners.inference_runner",
    JobType.eval.value: "amortized.runners.eval_runner",
}


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


async def _register_log_artifacts(job_id: str, output_dir: str) -> None:
    """Register stdout.log and stderr.log as log-type artifacts via core layer."""
    db = await _get_db()
    try:
        repo = Repository(db)
        await register_log_artifacts(repo, job_id, output_dir)
    finally:
        await db.close()


async def _run_job(job: dict[str, Any]) -> None:
    """Spawn a subprocess to execute the job and monitor it."""
    global _current_process, _current_job_id

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
    output_dir = os.path.join(base_dir, job_id)

    # Ensure output_dir is set in DB
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE jobs SET output_dir = ? WHERE id = ?",
            (output_dir, job_id),
        )
        await db.commit()
    finally:
        await db.close()

    # Update config with the resolved output_dir so runners use the correct path
    config = job["config"]
    if job["type"] == JobType.training.value:
        config = {**config, "ckpt_output_dir": output_dir}
    elif job["type"] == JobType.inference.value:
        if "output_path" not in config or not config["output_path"]:
            config = {**config, "output_path": os.path.join(output_dir, "results.jsonl")}
    elif "output_dir" not in config:
        config = {**config, "output_dir": output_dir}

    cmd = _build_runner_command({**job, "config": config, "output_dir": output_dir})

    logger.info("Starting job %s: %s", job_id, " ".join(cmd[:3]))

    stdout_file = None
    stderr_file = None
    try:
        # Redirect stdout/stderr to log files instead of pipes to avoid deadlock.
        # When using subprocess.PIPE, the pipe buffer (typically 64KB) can fill up
        # if the subprocess produces large output. The subprocess then blocks on
        # write while proc.wait() blocks waiting for exit — a classic deadlock.
        # See: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
        os.makedirs(output_dir, exist_ok=True)
        stdout_path = os.path.join(output_dir, "stdout.log")
        stderr_path = os.path.join(output_dir, "stderr.log")
        stdout_file = open(stdout_path, "w")  # noqa: SIM115
        stderr_file = open(stderr_path, "w")  # noqa: SIM115

        proc = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        _current_process = proc
        _current_job_id = job_id

        # Mark as running with PID
        await _update_job(
            job_id,
            status=JobStatus.running,
            started_at=now,
            pid=proc.pid,
        )

        # Wait for subprocess completion in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        returncode = await loop.run_in_executor(None, proc.wait)

        _current_process = None
        _current_job_id = None

        # Close log files now that the process has exited
        stdout_file.close()
        stderr_file.close()
        stdout_file = None
        stderr_file = None

        completed_at = datetime.now(UTC).isoformat()

        # Register log files as artifacts
        await _register_log_artifacts(job_id, output_dir)

        if returncode == 0:
            await _update_job(
                job_id,
                status=JobStatus.completed,
                completed_at=completed_at,
            )
            await _register_artifacts_for_job(job_id, output_dir)
            logger.info("Job %s completed successfully", job_id)
        elif returncode == -signal.SIGTERM or returncode == -signal.SIGKILL:
            await _update_job(
                job_id,
                status=JobStatus.cancelled,
                completed_at=completed_at,
                error="Job was cancelled",
            )
            logger.info("Job %s was cancelled", job_id)
        else:
            stderr_output = ""
            try:
                with open(stderr_path) as f:
                    stderr_output = f.read()[-2000:]
            except OSError:
                pass
            await _update_job(
                job_id,
                status=JobStatus.failed,
                completed_at=completed_at,
                error=f"Process exited with code {returncode}: {stderr_output}",
            )
            logger.error("Job %s failed with code %d", job_id, returncode)

    except Exception as exc:
        _current_process = None
        _current_job_id = None
        if stdout_file is not None:
            stdout_file.close()
        if stderr_file is not None:
            stderr_file.close()
        await _update_job(
            job_id,
            status=JobStatus.failed,
            completed_at=datetime.now(UTC).isoformat(),
            error=str(exc),
        )
        logger.exception("Job %s failed with exception", job_id)


async def kill_job_process(pid: int, timeout: float = 5.0) -> bool:
    """Kill a job subprocess. Returns True if the process was terminated."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    # Wait for graceful shutdown
    loop = asyncio.get_event_loop()
    start = asyncio.get_event_loop().time()

    while True:
        try:
            os.kill(pid, 0)  # Check if still alive
        except OSError:
            return True

        elapsed = loop.time() - start
        if elapsed >= timeout:
            # Force kill
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
            return True

        await asyncio.sleep(0.1)


async def worker_loop(poll_interval: float = 2.0) -> None:
    """Main worker loop — polls for pending jobs and runs them."""
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
