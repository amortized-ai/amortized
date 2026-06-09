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


_ALGORITHM_IMPORTS: dict[str, tuple[str, str]] = {
    "lora_sft": ("training_hub", "lora_sft"),
    "full_sft": ("training_hub", "sft"),
    "sft": ("training_hub", "sft"),
    "grpo": ("training_hub", "lora_grpo"),
    "lora_grpo": ("training_hub", "lora_grpo"),
    "osft": ("training_hub", "osft"),
    "gepa": ("training_hub", "gepa"),
}


def _generate_container_script(job_type: str, config: dict[str, Any]) -> str:
    """Generate a self-contained Python script for container execution.

    The script reads config from /amortized/config.json (already mounted)
    and calls the library function directly. No custom framework needed —
    the container only needs the library installed.
    """
    if job_type == JobType.training.value:
        return _training_script(config)
    if job_type == JobType.sdg.value:
        return _sdg_script()
    if job_type == JobType.eval.value:
        return _eval_script()
    if job_type == JobType.inference.value:
        return _inference_script()
    raise ValueError(f"No container script for job type: {job_type}")


def _training_script(config: dict[str, Any]) -> str:
    algorithm = config.get("algorithm", "lora_sft")
    entry = _ALGORITHM_IMPORTS.get(algorithm)
    if not entry:
        raise ValueError(f"Unknown training algorithm: {algorithm}")
    module, func = entry
    return f"""\
import json

config = json.load(open("/amortized/config.json"))["config"]
from {module} import {func}

skip = {{"algorithm", "ckpt_output_dir", "output_dir"}}
kwargs = {{k: v for k, v in config.items() if k not in skip and v is not None}}
kwargs["ckpt_output_dir"] = "/amortized/work"
{func}(**kwargs)
"""


def _sdg_script() -> str:
    return """\
import json, os

config = json.load(open("/amortized/config.json"))["config"]

from asynth import LiteLLMInferenceConfig, SynthesisConfig, synthesize
from asynth.configs.params.synthesis_params import GeneralSynthesisParams

os.makedirs("/amortized/work/output", exist_ok=True)

inference_config = LiteLLMInferenceConfig(
    model=config["model"],
    temperature=config.get("temperature", 0.7),
    max_concurrency=config.get("max_concurrency", 16),
    max_tokens=config.get("max_tokens"),
    top_p=config.get("top_p"),
    seed=config.get("seed"),
    num_retries=config.get("num_retries", 3),
    api_base=config.get("api_base"),
    api_key=config.get("api_key"),
)

raw_strategy = config.get("strategy_params", {})
if raw_strategy and isinstance(raw_strategy, dict):
    merged = dict(raw_strategy)
    if config.get("input_data") and "input_data" not in merged:
        merged["input_data"] = config["input_data"]
    if config.get("input_documents") and "input_documents" not in merged:
        merged["input_documents"] = config["input_documents"]
    if hasattr(GeneralSynthesisParams, "from_dict"):
        strategy_params = GeneralSynthesisParams.from_dict(merged)
    else:
        strategy_params = GeneralSynthesisParams(**merged)
else:
    strategy_params = GeneralSynthesisParams()

synth_config = SynthesisConfig(
    num_samples=config.get("num_samples", 100),
    output_path="/amortized/work/output/generated_data.jsonl",
    inference_config=inference_config,
    strategy_params=strategy_params,
)

synthesize(synth_config)
"""


def _eval_script() -> str:
    return """\
import json, os

config = json.load(open("/amortized/config.json"))["config"]

from asynth import JudgeConfig, LiteLLMInferenceConfig, create_judge

dataset_path = config.get("dataset_path") or config.get("dataset")
if not dataset_path:
    raise ValueError("dataset or dataset_path is required for eval jobs")

data = [json.loads(line) for line in open(dataset_path) if line.strip()]

evaluator_type = config.get("evaluator_type", "llm")
judgment_type = config.get("judgment_type", "bool")
response_format = config.get("response_format", "json")

if evaluator_type == "rule_based":
    judge_config = JudgeConfig.from_dict({
        "rule_judge_params": {
            "rule_type": config.get("rule_config", {}).get("rule_type", "regex"),
            "input_fields": config.get("variables", []),
            "rule_config": config.get("rule_config", {}),
            "response_format": response_format,
            "judgment_type": judgment_type,
        },
    })
else:
    judge_params = {
        "prompt_template": config.get("judge_prompt", "Evaluate the following: {response}"),
        "response_format": response_format,
        "judgment_type": judgment_type,
    }
    if config.get("variables"):
        judge_params["prompt_template_placeholders"] = config["variables"]
    judge_config = JudgeConfig.from_dict({"judge_params": judge_params})

inference_config = None
if evaluator_type == "llm":
    model = config.get("judge_model") or config.get("model", "openai/gpt-4o-mini")
    inf_params = config.get("inference_params", {})
    inference_config = LiteLLMInferenceConfig(
        model=model,
        temperature=inf_params.get("temperature", 1.0),
        max_tokens=inf_params.get("max_tokens"),
        top_p=inf_params.get("top_p"),
        seed=inf_params.get("seed"),
        api_base=inf_params.get("api_base"),
        api_key=inf_params.get("api_key"),
    )

judge = create_judge(judge_config, inference_config=inference_config)
outputs = judge.judge(data)

os.makedirs("/amortized/work", exist_ok=True)
results = []
for i, output in enumerate(outputs):
    results.append({
        "index": i,
        "passed": output.passed,
        "score": output.score,
        "explanation": output.explanation,
        "raw_output": output.raw_output,
    })

total = len(results)
passed = sum(1 for r in results if r.get("passed", False))
scores = [float(r["score"]) for r in results if r.get("score") is not None]
avg_score = sum(scores) / len(scores) if scores else 0.0

output = {
    "results": results,
    "summary": {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_score": round(avg_score, 4),
    },
}

with open("/amortized/work/eval_results.json", "w") as f:
    json.dump(output, f, indent=2)
"""


def _inference_script() -> str:
    return """\
import json, os

config = json.load(open("/amortized/config.json"))["config"]

from vllm import LLM, SamplingParams

model_path = config["model_path"]
tp = config.get("tensor_parallel_size", 1)
llm = LLM(model=model_path, tensor_parallel_size=tp, trust_remote_code=True)

sampling = SamplingParams(
    temperature=config.get("temperature", 0.7),
    max_tokens=config.get("max_tokens", 512),
    top_p=config.get("top_p", 1.0),
)

prompts = config.get("prompts", [])
input_path = config.get("input_path")
if input_path and not prompts:
    with open(input_path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row.get("prompt", row.get("input", "")))

outputs = llm.generate(prompts, sampling)

os.makedirs("/amortized/work", exist_ok=True)
with open("/amortized/work/results.jsonl", "w") as f:
    for output in outputs:
        f.write(json.dumps({
            "prompt": output.prompt,
            "output": output.outputs[0].text,
            "finish_reason": output.outputs[0].finish_reason,
        }) + "\\n")
"""


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

    config = await _resolve_artifact_refs(config)

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
    for env_name in config_mod.settings.forward_env:
        value = os.environ.get(env_name)
        if value:
            spec_env[env_name] = value

    image = _JOB_TYPE_IMAGES.get(job["type"])

    spec_env["_config"] = json.dumps(config)

    if image:
        script = _generate_container_script(job["type"], config)
        spec_env["_run_script"] = script
        python_bin = "python3.11" if job["type"] == JobType.training.value else "python3"
        cmd = [python_bin, "/amortized/work/run.py"]

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

        if handle.secret_names and hasattr(backend, "cleanup_secrets"):
            try:
                await backend.cleanup_secrets(handle)
            except Exception:
                logger.warning("Failed to clean up secrets for job %s", job_id, exc_info=True)

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
