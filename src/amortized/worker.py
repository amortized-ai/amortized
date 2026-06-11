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
    "sdg": "ghcr.io/amortized-ai/asynth:latest",
    "eval": "ghcr.io/amortized-ai/asynth:latest",
    "serve": "docker.io/vllm/vllm-openai",
}

TRAINING_HUB_ALGOS = {"lora_sft", "sft", "osft", "grpo", "lora_grpo", "gepa"}
TRL_ALGOS = {"dpo", "kto", "gkd"}

_RUNNER_MODULES: dict[str, str] = {
    JobType.sdg.value: "amortized.runners.sdg_runner",
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

    if job_type == JobType.training.value:
        algorithm = config.get("algorithm", "sft")
        if algorithm in TRAINING_HUB_ALGOS:
            # training-hub algos are dispatched via container script;
            # this placeholder is overridden by the container dispatch path.
            return [sys.executable, "-m", "training_hub", algorithm, json.dumps(config)]
        subcommand = _get_trl_subcommand(algorithm)
        return ["trl", subcommand, "--config", json.dumps(config)]

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


_TRL_SUBCOMMANDS: dict[str, str] = {
    "dpo": "dpo",
    "kto": "kto",
    "gkd": "gkd",
}


def _get_trl_subcommand(algorithm: str) -> str:
    sub = _TRL_SUBCOMMANDS.get(algorithm)
    if sub is None:
        raise ValueError(f"Unknown training algorithm: {algorithm}")
    return sub


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

_TRL_SKIP_KEYS = {"algorithm", "ckpt_output_dir", "output_dir", "engine", "load_in_4bit"}


def _training_config_yaml(config: dict[str, Any]) -> str:
    import yaml

    trl_config: dict[str, Any] = {
        "output_dir": "/amortized/work",
        "report_to": "none",
    }

    for key, value in config.items():
        if key in _TRL_SKIP_KEYS or value is None:
            continue
        if key == "data_path":
            trl_config["datasets"] = [{"path": os.path.dirname(value)}]
            continue
        trl_field = _TRL_FIELD_MAP.get(key)
        if trl_field:
            trl_config[trl_field] = value
        elif key not in _TRL_FIELD_MAP:
            trl_config[key] = value

    if "lora_r" in trl_config:
        trl_config["use_peft"] = True

    result: str = yaml.dump(trl_config, default_flow_style=False, sort_keys=False)
    return result


_TRAINING_HUB_FIELD_MAP: dict[str, str] = {
    "model_name_or_path": "model_path",
    "num_train_epochs": "num_epochs",
    "per_device_train_batch_size": "micro_batch_size",
    "max_length": "max_seq_len",
    "output_dir": "ckpt_output_dir",
}

_TRAINING_HUB_NATIVE_KEYS = {
    "model_path",
    "data_path",
    "ckpt_output_dir",
    "num_epochs",
    "micro_batch_size",
    "max_seq_len",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "bf16",
    "gradient_checkpointing",
    "gradient_accumulation_steps",
    "sample_packing",
    "seed_candidate",
    "task_lm",
}

_TRAINING_HUB_SKIP_KEYS = {
    "algorithm",
    "engine",
    "use_peft",
    "qlora",
    "bnb_4bit_quant_type",
    "bnb_4bit_compute_dtype",
    "lora_target_modules",
}


def _training_hub_script(algorithm: str) -> str:
    return (
        "import json, os\n"
        "\n"
        "config = json.load(open('/amortized/config.json'))['config']\n"
        f"from training_hub import {algorithm}\n"
        "\n"
        "FIELD_MAP = " + repr(_TRAINING_HUB_FIELD_MAP) + "\n"
        "SKIP_KEYS = " + repr(_TRAINING_HUB_SKIP_KEYS) + "\n"
        "\n"
        "kwargs = {}\n"
        "for key, value in config.items():\n"
        "    if key in SKIP_KEYS or value is None:\n"
        "        continue\n"
        "    if key == 'load_in_4bit' and value:\n"
        "        kwargs['load_in_4bit'] = True\n"
        "        continue\n"
        "    mapped = FIELD_MAP.get(key)\n"
        "    if mapped:\n"
        "        kwargs[mapped] = value\n"
        "    elif key not in FIELD_MAP:\n"
        "        kwargs[key] = value\n"
        "\n"
        f"{algorithm}(**kwargs)\n"
    )


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


def _generate_container_script(job_type: str, config: dict[str, Any]) -> str:
    """Generate a self-contained Python script for container execution."""
    if job_type == JobType.sdg.value:
        return _sdg_script()
    if job_type == JobType.eval.value:
        return _eval_script()
    raise ValueError(f"No container script for job type: {job_type}")


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


def _eval_script() -> str:
    return """\
import json, os, re

config = json.load(open("/amortized/config.json"))["config"]

dataset_path = config["dataset"]
data = [json.loads(line) for line in open(dataset_path) if line.strip()]
max_samples = config.get("max_samples")
if max_samples:
    data = data[:max_samples]

checks = config.get("deterministic_checks", [])

judge_config = config.get("judge", {})
judge_model = judge_config.get("model")
judge_prompt = judge_config.get("prompt", "Evaluate this response: {response}")
judge_system = judge_config.get("system_instruction")
judge_temp = judge_config.get("temperature", 0.0)
judge_judgment_type = judge_config.get("judgment_type", "bool")
judge_response_format = judge_config.get("response_format", "json")
judge_include_explanation = judge_config.get("include_explanation", True)

model_endpoint = config.get("model_endpoint")
model_name = config.get("model_name")

results = []
if model_endpoint and model_name:
    from openai import OpenAI
    client = OpenAI(base_url=model_endpoint, api_key="dummy")
    temperature = config.get("temperature", 0.0)

    for i, row in enumerate(data):
        messages = row.get("messages", [])
        prompt_messages = [m for m in messages if m["role"] != "assistant"]

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=256,
            )
            actual_output = completion.choices[0].message.content
        except Exception as e:
            actual_output = f"ERROR: {e}"

        expected_output = ""
        for m in messages:
            if m["role"] == "assistant":
                expected_output = m["content"]
                break

        result = {
            "index": i,
            "input": prompt_messages[-1]["content"] if prompt_messages else "",
            "expected": expected_output,
            "actual": actual_output,
        }

        for check in checks:
            field = check["field"]
            check_type = check.get("type", "exact_match")

            if check_type == "exact_match":
                pattern = rf"{field}:\\s*(\\w+)"
                expected_match = re.search(pattern, expected_output, re.IGNORECASE)
                actual_match = re.search(pattern, actual_output, re.IGNORECASE)
                expected_val = expected_match.group(1).lower() if expected_match else ""
                actual_val = actual_match.group(1).lower() if actual_match else ""
                result[f"{field}_expected"] = expected_val
                result[f"{field}_actual"] = actual_val
                result[f"{field}_correct"] = expected_val == actual_val

            elif check_type == "contains":
                values = check.get("values", [])
                result[f"{field}_correct"] = all(v in actual_output for v in values)

            elif check_type == "enum_match":
                pattern = rf"{field}:\\s*(\\w+)"
                actual_match = re.search(pattern, actual_output, re.IGNORECASE)
                actual_val = actual_match.group(1).lower() if actual_match else ""
                allowed = [a.lower() for a in check.get("allowed", [])]
                result[f"{field}_actual"] = actual_val
                result[f"{field}_correct"] = actual_val in allowed if actual_val else False

        results.append(result)
else:
    for i, row in enumerate(data):
        results.append({"index": i, **row})

summary = {"total": len(results)}
check_fields = [c["field"] for c in checks]
for field in check_fields:
    correct = sum(1 for r in results if r.get(f"{field}_correct", False))
    summary[f"{field}_accuracy"] = round(correct / len(results), 4) if results else 0.0
    summary[f"{field}_correct"] = correct

if check_fields:
    all_correct = sum(
        1 for r in results
        if all(r.get(f"{f}_correct", False) for f in check_fields)
    )
    summary["overall_accuracy"] = round(
        all_correct / len(results), 4
    ) if results else 0.0

if judge_model:
    from asynth import create_judge, JudgeConfig as AsynthJudgeConfig
    from asynth.configs.params.judge_params import JudgeParams
    from asynth.inference.litellm_engine import LiteLLMInferenceConfig

    judge_params_kwargs = {
        "prompt_template": judge_prompt,
        "response_format": judge_response_format,
        "judgment_type": judge_judgment_type,
        "include_explanation": judge_include_explanation,
    }
    if judge_system:
        judge_params_kwargs["system_instruction"] = judge_system
    jc = AsynthJudgeConfig(
        judge_params=JudgeParams(**judge_params_kwargs),
        inference_config=LiteLLMInferenceConfig(
            model=judge_model,
            temperature=judge_temp,
        ),
    )
    j = create_judge(jc)
    judge_data = [{"request": r["input"], "response": r["actual"]} for r in results]
    judge_outputs = j.judge(judge_data)

    for i, output in enumerate(judge_outputs):
        results[i]["judge_passed"] = output.field_values.get("judgment", False)
        results[i]["judge_score"] = output.field_scores.get("judgment", 0.0)
        results[i]["judge_explanation"] = output.field_values.get("explanation", "")

    passed = sum(1 for r in results if r.get("judge_passed", False))
    scores = [r.get("judge_score", 0.0) for r in results if r.get("judge_score") is not None]
    summary["judge_pass_rate"] = round(passed / len(results), 4) if results else 0.0
    summary["judge_avg_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0

os.makedirs("/amortized/work", exist_ok=True)
output = {"results": results, "summary": summary}
with open("/amortized/work/eval_results.json", "w") as f:
    json.dump(output, f, indent=2)
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


async def _register_artifacts_for_job(
    job_id: str, output_dir: str, *, job_type: str | None = None
) -> None:
    """Scan output directory and register found artifacts via core layer."""
    db = await _get_db()
    try:
        repo = Repository(db)
        await register_artifacts_for_job(repo, job_id, output_dir, job_type=job_type)
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
        cmd = ["--config", "/amortized/work/config.yaml"]
    else:
        cmd = _build_runner_command({**job, "config": config, "output_dir": output_dir})
        spec_ports = {}

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

    if job["type"] == JobType.eval.value:
        config = _resolve_judge_template(config)

    spec_env["_config"] = json.dumps(config)

    if image and job["type"] == JobType.training.value:
        algorithm = config.get("algorithm", "sft")
        if algorithm in TRAINING_HUB_ALGOS:
            script = _training_hub_script(algorithm)
            spec_env["_run_script"] = script
            cmd = ["python3.11", "/amortized/work/run.py"]
        elif algorithm in TRL_ALGOS:
            subcommand = _get_trl_subcommand(algorithm)
            trl_yaml = _training_config_yaml(config)
            spec_env["_run_config"] = trl_yaml
            cmd = ["trl", subcommand, "--config", "/amortized/work/config.yaml"]
        else:
            raise ValueError(f"Unknown training algorithm: {algorithm}")
    elif image and job["type"] == JobType.serve.value:
        spec_env["_run_config"] = _serve_config_yaml(config)
        if config.get("gpu_ids"):
            spec_env["CUDA_VISIBLE_DEVICES"] = str(config["gpu_ids"])
    elif image:
        script = _generate_container_script(job["type"], config)
        spec_env["_run_script"] = script
        cmd = ["python3", "/amortized/work/run.py"]

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
            await _register_artifacts_for_job(job_id, output_dir, job_type=job["type"])
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
