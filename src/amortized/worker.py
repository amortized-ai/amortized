"""Background worker that picks up queued jobs and runs them via ComputeBackend."""

import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import amortized.config as config_mod
from amortized.backends import BackendHandle, Capability, JobSpec, Resources, S3Download
from amortized.core.compute import MissingCapabilityError, check_capabilities, get_backend
from amortized.core.jobs import deserialize_handle
from amortized.core.mlflow_client import MLflowClient
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType

logger = logging.getLogger("amortized.worker")


_JOB_TYPE_IMAGES: dict[str, str] = {
    "training": "ghcr.io/amortized-ai/training:latest",
    "sdg": "ghcr.io/amortized-ai/data-designer:latest",
}


_TRAINING_HUB_FIELD_MAP: dict[str, str] = {
    "model_name_or_path": "model_path",
    "num_train_epochs": "num_epochs",
    "per_device_train_batch_size": "micro_batch_size",
    "max_length": "max_seq_len",
    "output_dir": "ckpt_output_dir",
}

_TRAINING_HUB_SKIP_KEYS = {
    "algorithm",
    "engine",
    "use_peft",
    "qlora",
    "bnb_4bit_quant_type",
    "bnb_4bit_compute_dtype",
    "lora_target_modules",
    "model_id",
    "model",
    "num_samples",
    "compute",
    "task_description",
    "method",
    "dataset_job_id",
    "model_job_id",
}


def _training_hub_config_yaml(algorithm: str, config: dict[str, Any]) -> str:
    import yaml

    thub_config: dict[str, Any] = {}
    for key, value in config.items():
        if key in _TRAINING_HUB_SKIP_KEYS or value is None:
            continue
        if key == "output_dir" and algorithm == "gepa":
            thub_config["output_dir"] = value
            continue
        th_key = _TRAINING_HUB_FIELD_MAP.get(key, key)
        thub_config[th_key] = value

    output_dir = config.get("output_dir", "/amortized/work/output")
    if algorithm == "gepa":
        thub_config.setdefault("output_dir", output_dir)
    else:
        thub_config.setdefault("ckpt_output_dir", output_dir)
        thub_config.setdefault("data_output_dir", output_dir + "/processed_data")

    if algorithm in ("sft", "lora_sft"):
        batch = thub_config.pop("micro_batch_size", 2)
        thub_config.setdefault("effective_batch_size", batch * 4)
        thub_config.setdefault("max_seq_len", 2048)
        thub_config.setdefault("max_batch_len", 60000)
    elif algorithm == "osft":
        batch = thub_config.pop("micro_batch_size", 2)
        thub_config.setdefault("effective_batch_size", batch * 4)
        thub_config.setdefault("max_seq_len", 2048)
        thub_config.setdefault("max_tokens_per_gpu", 4096)
        thub_config.setdefault("learning_rate", 2e-5)

    result: str = yaml.dump(thub_config, default_flow_style=False, sort_keys=False)
    return result


async def _get_repo() -> Repository:
    from amortized.db.connection import _get_shared_db

    db = await _get_shared_db()
    return Repository(db)


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
    repo = await _get_repo()
    await repo.update_job(job_id, **kwargs)


async def _pick_pending_job() -> dict[str, Any] | None:
    repo = await _get_repo()
    return await repo.pick_pending_job()


async def _resolve_mlflow_artifact_uri(mlflow_run_id: str) -> str:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return ""

    try:
        client = MLflowClient(tracking_uri)
        run = await client.get_run(mlflow_run_id)
        uri: str = run["info"]["artifact_uri"]
        return uri
    except Exception:
        logger.warning(
            "Failed to resolve MLflow artifact URI for run %s", mlflow_run_id, exc_info=True
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


async def _set_mlflow_run_tag(mlflow_run_id: str, key: str, value: str) -> None:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return

    try:
        client = MLflowClient(tracking_uri)
        await client.set_tag(mlflow_run_id, key, value)
    except Exception:
        logger.warning(
            "Failed to set MLflow tag %s=%s on run %s",
            key,
            value,
            mlflow_run_id,
            exc_info=True,
        )


async def _register_training_model(job: dict[str, Any], mlflow_run_id: str) -> bool:
    """Register a trained model in the MLflow model registry. Returns True on success."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return False

    model_id = job.get("config", {}).get("model_id", "unknown")
    algorithm = job.get("config", {}).get("algorithm", "sft")
    job_id = job["id"]
    model_name = f"{model_id}-{algorithm}-{job_id[:8]}"

    try:
        client = MLflowClient(tracking_uri)
        description = f"Fine-tuned {model_id} via {algorithm} (job {job_id[:8]})"
        return await client.register_model(model_name, mlflow_run_id, description)
    except Exception:
        logger.warning("Failed to register model %s", model_name, exc_info=True)
        return False


async def _fetch_document_chunks(document_id: str) -> list[str]:
    """Fetch pre-chunked document content from MLflow artifact store.

    Raises on failure so callers can decide whether to fail the job.
    """
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not document_id:
        return []

    client = MLflowClient(tracking_uri)
    files = await client.list_artifacts(document_id, "chunks")
    chunk_paths = sorted(
        f["path"] for f in files
        if f.get("path", "").endswith(".md")
    )
    chunks: list[str] = []
    for path in chunk_paths:
        text = await client.get_artifact_text(document_id, path)
        if text:
            chunks.append(text)
    return chunks


async def _resolve_parent_artifacts(
    job: dict[str, Any],
    config: dict[str, Any],
    s3_downloads: list[S3Download],
) -> dict[str, Any]:
    """Resolve parent job artifacts and inject into config for chaining."""
    parent_job_id = job.get("parent_job_id", "") or config.get("parent_job_id", "")
    if not parent_job_id:
        return config

    repo = await _get_repo()
    parent = await repo.get_job(parent_job_id)

    if not parent:
        logger.warning("Parent job %s not found", parent_job_id)
        return config

    parent_run_id = parent.get("mlflow_run_id", "")
    if not parent_run_id:
        logger.warning("Parent job %s has no mlflow_run_id", parent_job_id)
        return config

    artifact_uri = await _resolve_mlflow_artifact_uri(parent_run_id)
    if not artifact_uri:
        logger.warning("Could not resolve artifact URI for run %s", parent_run_id)
        return config

    config = dict(config)
    if job["type"] == JobType.training.value and parent["type"] in ("sdg", "upload"):
        existing = config.get("data_path", "")
        if not existing or not existing.startswith("s3://"):
            s3_dir = f"{artifact_uri}/generated_data/"
            local_dir = "/amortized/work/data"
            s3_downloads.append(
                S3Download(
                    s3_uri=s3_dir,
                    local_path=local_dir,
                    is_directory=True,
                )
            )
            config["data_path"] = local_dir
            logger.info(
                "Injected SDG data from MLflow run %s: %s",
                parent_run_id,
                s3_dir,
            )

    return config


async def _run_job(job: dict[str, Any]) -> None:
    job_id = job["id"]
    now = datetime.now(UTC).isoformat()
    config = dict(job["config"])
    config.pop("_secrets", None)

    backend_name = config_mod.settings.resolved_default_backend

    output_dir_names = {
        JobType.training.value: "training_output",
        JobType.sdg.value: "sdg_output",
    }
    dir_name = output_dir_names.get(job["type"], f"{job['type']}_output")
    base_dir = str(config_mod.settings.data_dir / dir_name)
    output_dir = os.path.abspath(os.path.expanduser(os.path.join(base_dir, job_id)))

    if job["type"] == JobType.training.value or "output_dir" not in config:
        config["output_dir"] = output_dir

    if backend_name == "kubernetes" and job["type"] == JobType.training.value:
        config["output_dir"] = "/amortized/work/output"

    for key, value in list(config.items()):
        if isinstance(value, str) and value.startswith("~"):
            config[key] = os.path.expanduser(value)

    try:
        backend = get_backend(backend_name)
    except KeyError:
        await _update_job(
            job_id,
            status=JobStatus.failed.value,
            completed_at=datetime.now(UTC).isoformat(),
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
                completed_at=datetime.now(UTC).isoformat(),
                error=f"Job '{job_id}' cannot run on backend '{backend_name}': {exc}",
            )
            return

    spec_env: dict[str, str] = {}
    for env_name in config_mod.settings.forward_env:
        value = os.environ.get(env_name)
        if value:
            spec_env[env_name] = value

    if config_mod.settings.mlflow_tracking_uri:
        mlflow_experiment = f"amortized/{job['type']}/{job_id[:8]}"
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
        if job["type"] == JobType.training.value:
            spec_env["HF_MLFLOW_LOG_ARTIFACTS"] = "true"
        await _update_job(job_id, mlflow_experiment=mlflow_experiment)

    image = _JOB_TYPE_IMAGES.get(job["type"])

    config_files: dict[str, str] = {}
    s3_downloads: list[S3Download] = []
    cmd: list[str] = []

    config = await _resolve_parent_artifacts(job, config, s3_downloads)

    if image and job["type"] == JobType.training.value:
        algo_aliases = {"lora": "lora_sft", "qlora": "lora_sft", "qlora_sft": "lora_sft"}
        algorithm = config.get("algorithm", "sft")
        algorithm = algo_aliases.get(algorithm, algorithm)
        data_path = config.get("data_path", config.get("dataset", ""))
        if data_path.startswith("s3://"):
            is_dir = data_path.endswith("/")
            if is_dir:
                local_path = "/amortized/work/data"
            else:
                local_name = data_path.split("/")[-1]
                local_path = f"/amortized/work/{local_name}"
            s3_downloads.append(
                S3Download(
                    s3_uri=data_path,
                    local_path=local_path,
                    is_directory=is_dir,
                )
            )
            config = {**config, "data_path": local_path}
        if config_mod.settings.mlflow_tracking_uri:
            config.setdefault("report_to", "mlflow")
        config_files["config.yaml"] = _training_hub_config_yaml(algorithm, config)
        thub_subcommand = algorithm.replace("_", "-")
        cmd = ["thub", thub_subcommand, "--config", "/amortized/config.yaml"]
    elif image and job["type"] == JobType.sdg.value:
        import yaml

        spec_env["DD_API_KEY"] = "not-needed"

        for stale_key in (
            "model",
            "api_base",
            "api_key",
            "num_samples",
            "max_concurrency",
            "temperature",
            "max_tokens",
            "top_p",
            "seed",
            "num_retries",
            "input_data",
            "input_documents",
            "strategy_params",
            "task_description",
            "document_id",
            "output_dir",
        ):
            config.pop(stale_key, None)

        document_ids = config.pop("document_ids", []) or config.pop("document_id", [])
        if isinstance(document_ids, str):
            document_ids = [document_ids]
        doc_setup_cmds: list[str] = []
        chunk_count = 0
        if document_ids and config_mod.settings.mlflow_tracking_uri:
            seed_config = config.get("seed_config", {})
            source = seed_config.get("source", {})
            source.pop("chunk_size", None)
            source.pop("chunk_overlap", None)
            source.pop("tokenizer", None)
            source.pop("sentences_per_chunk", None)
            source.pop("min_text_length", None)

            for doc_id in document_ids:
                try:
                    chunks = await _fetch_document_chunks(doc_id)
                except Exception:
                    error_msg = f"Failed to fetch chunks for document {doc_id}"
                    logger.error("Job %s: %s", job_id, error_msg, exc_info=True)
                    await _update_job(
                        job_id,
                        status=JobStatus.failed.value,
                        completed_at=datetime.now(UTC).isoformat(),
                        error=error_msg,
                    )
                    return
                for chunk_text in chunks:
                    config_files[f"chunk_{chunk_count}.md"] = chunk_text
                    chunk_count += 1

            if chunk_count:
                doc_setup_cmds = [
                    "mkdir -p /tmp/chunks",
                    *(f"cp /amortized/chunk_{i}.md /tmp/chunks/" for i in range(chunk_count)),
                ]
                source["seed_type"] = "file_contents"
                source["path"] = "/tmp/chunks"
                source.setdefault("encoding", "utf-8")
                seed_config["source"] = source
                config["seed_config"] = seed_config
                logger.info(
                    "Job %s: fetched %d pre-chunked chunks from %d document(s)",
                    job_id, chunk_count, len(document_ids),
                )
            else:
                error_msg = (
                    f"Job requires {len(document_ids)} document(s) but no chunks"
                    " could be fetched from MLflow. Check that the document"
                    " IDs are valid and MLflow is reachable."
                )
                logger.error("Job %s: %s", job_id, error_msg)
                await _update_job(
                    job_id,
                    status=JobStatus.failed.value,
                    completed_at=datetime.now(UTC).isoformat(),
                    error=error_msg,
                )
                return

        for mc in config.get("model_configs", []):
            params = mc.setdefault("inference_parameters", {})
            params.setdefault("max_parallel_requests", 32)

        num_records = config.pop("num_records", 100)
        config.pop("topic", None)
        dd_config = {"data_designer": config}
        config_files["config.yaml"] = yaml.dump(
            dd_config, default_flow_style=False, sort_keys=False
        )
        dd_cmd = (
            "data-designer create /amortized/config.yaml"
            f" --num-records {num_records}"
            " --artifact-path /amortized/work"
            " --no-tui"
        )
        processor_names = [p.get("name", "") for p in config.get("processors", [])]
        proc_dir = f"processors-files/{processor_names[-1]}" if processor_names else ""
        upload_cmd = (
            f"python3 /usr/local/bin/upload_to_mlflow.py /amortized/work/dataset {proc_dir}"
        )
        all_cmds = [*doc_setup_cmds, dd_cmd, upload_cmd]
        cmd = ["sh", "-c", " && ".join(all_cmds)]

    job_type = job["type"]
    resolved_config = dict(config)
    if job_type == JobType.sdg.value:
        resolved_config["num_records"] = num_records
    await _update_job(job_id, config=json.dumps(resolved_config))

    needs_gpu = job_type == "training"
    spec = JobSpec(
        job_id=job_id,
        command=cmd,
        env=spec_env,
        work_dir=output_dir,
        image=image,
        config_files=config_files,
        s3_downloads=s3_downloads,
        job_type=job_type,
        user_id=job.get("user_id", ""),
        resources=Resources(gpus=1 if needs_gpu else 0),
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

        poll_interval = 2.0
        transitioned_to_running = False
        while True:
            status = await backend.status(handle)
            if not status.running:
                break
            if not transitioned_to_running:
                await _update_job(job_id, status=JobStatus.running.value)
                transitioned_to_running = True
            await asyncio.sleep(poll_interval)

        completed_at = datetime.now(UTC).isoformat()

        if handle.secret_names and hasattr(backend, "cleanup_secrets"):
            try:
                await backend.cleanup_secrets(handle)
            except Exception:
                logger.warning("Failed to clean up secrets for job %s", job_id, exc_info=True)

        if status.exit_code == 0:
            mlflow_run_id = await _extract_mlflow_run_id(backend, handle)
            if mlflow_run_id:
                await _set_mlflow_run_tag(mlflow_run_id, "job_type", job["type"])
                await _set_mlflow_run_tag(mlflow_run_id, "job_id", job_id)

                if job["type"] == JobType.sdg.value:
                    job_config = job.get("config", {})
                    if isinstance(job_config, str):
                        job_config = json.loads(job_config)
                    nr = job_config.get("num_records", "")
                    if nr:
                        await _set_mlflow_run_tag(mlflow_run_id, "num_samples", str(nr))
                    mc = job_config.get("model_configs", [])
                    if mc and isinstance(mc, list):
                        model_name = mc[0].get("model", "")
                        await _set_mlflow_run_tag(mlflow_run_id, "teacher_model", model_name)
                    topic = job_config.get("topic", "")
                    if topic:
                        await _set_mlflow_run_tag(mlflow_run_id, "dataset_topic", topic)

                if job["type"] == JobType.training.value:
                    model_registered = await _register_training_model(job, mlflow_run_id)
                    if not model_registered:
                        logger.warning("Job %s succeeded but model registration failed", job_id)

            await _update_job(
                job_id,
                status=JobStatus.succeeded.value,
                completed_at=completed_at,
                mlflow_run_id=mlflow_run_id,
            )
            logger.info("Job %s succeeded", job_id)
        elif status.exit_code is not None and status.exit_code < 0:
            await _update_job(
                job_id,
                status=JobStatus.cancelled.value,
                completed_at=completed_at,
                error="Job was cancelled",
            )
            logger.info("Job %s was cancelled", job_id)
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

    except Exception as exc:
        error_text = f"Job '{job_id}' failed during submission to backend '{backend_name}': {exc}"
        # Write error to stderr.log so logs endpoint can serve it even without a backend handle
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
            completed_at=datetime.now(UTC).isoformat(),
            error=error_text,
            backend_handle=fallback_handle,
        )
        logger.exception("Job %s failed with exception", job_id)


async def cleanup_orphaned_jobs() -> None:
    repo = await _get_repo()
    running_jobs = await repo.list_jobs(status=JobStatus.running)
    now = datetime.now(UTC).isoformat()

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
            await repo.update_job(
                job_id,
                status=JobStatus.failed.value,
                completed_at=now,
                error="Orphaned job — process no longer running",
            )
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
