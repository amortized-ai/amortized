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
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{tracking_uri}/api/2.0/mlflow/runs/get",
                params={"run_id": mlflow_run_id},
            )
            resp.raise_for_status()
            run = resp.json()["run"]
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
            if len(log_lines) > 200:
                log_lines = log_lines[-200:]
        log_text = "\n".join(log_lines)
        match = re.search(r"/runs/([a-f0-9]{32})", log_text)
        return match.group(1) if match else ""
    except Exception:
        return ""


async def _set_mlflow_run_tag(mlflow_run_id: str, key: str, value: str) -> None:
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{tracking_uri}/api/2.0/mlflow/runs/set-tag",
                json={
                    "run_id": mlflow_run_id,
                    "key": key,
                    "value": value,
                },
            )
    except Exception:
        logger.warning(
            "Failed to set MLflow tag %s=%s on run %s",
            key,
            value,
            mlflow_run_id,
            exc_info=True,
        )


async def _get_training_job_for_serve(training_job_id: str) -> dict[str, Any]:
    repo = await _get_repo()
    job = await repo.get_job(training_job_id)
    if not job:
        raise ValueError(f"Training job not found: {training_job_id}")
    if job["status"] != "succeeded":
        raise ValueError(f"Training job has not succeeded (status: {job['status']})")
    return job


async def _register_training_model(job: dict[str, Any], mlflow_run_id: str) -> bool:
    """Register a trained model in the MLflow model registry. Returns True on success."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not mlflow_run_id:
        return False
    import httpx

    model_id = job.get("config", {}).get("model_id", "unknown")
    algorithm = job.get("config", {}).get("algorithm", "sft")
    job_id = job["id"]
    model_name = f"{model_id}-{algorithm}-{job_id[:8]}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{tracking_uri}/api/2.0/mlflow/registered-models/create",
                json={"name": model_name},
            )
            if resp.status_code == 409:
                logger.info("Model %s already registered", model_name)
            elif resp.is_error:
                logger.warning("Failed to create registered model: %s", resp.text)
                return False

            source = f"runs:/{mlflow_run_id}/model"
            await client.post(
                f"{tracking_uri}/api/2.0/mlflow/model-versions/create",
                json={
                    "name": model_name,
                    "source": source,
                    "run_id": mlflow_run_id,
                    "description": f"Fine-tuned {model_id} via {algorithm} (job {job_id[:8]})",
                },
            )
            logger.info("Registered model version %s from run %s", model_name, mlflow_run_id)
        return True
    except Exception:
        logger.warning("Failed to register model %s", model_name, exc_info=True)
        return False


async def _fetch_document_content(document_id: str) -> str:
    """Fetch parsed document content from MLflow artifact store."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri or not document_id:
        return ""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{tracking_uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts"
                "/parsed_content.md",
                params={"run_id": document_id},
            )
            if resp.is_success:
                return resp.text
            logger.warning(
                "Failed to fetch document %s: %d", document_id, resp.status_code
            )
    except Exception:
        logger.warning("Failed to fetch document %s", document_id, exc_info=True)
    return ""


async def _upload_sdg_results_to_mlflow(
    backend: Any, handle: BackendHandle, job: dict[str, Any]
) -> str:
    """Read DD output from container logs and upload to MLflow as an artifact."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri:
        return ""
    import httpx

    job_id = job["id"]
    experiment_name = (
        job.get("mlflow_experiment") or f"amortized/sdg/{job_id[:8]}"
    )

    try:
        log_lines: list[str] = []
        async for line in backend.logs(handle):
            log_lines.append(line)

        jsonl_lines: list[str] = []
        in_jsonl = False
        for line in log_lines:
            if "=== AMORTIZED_JSONL_START ===" in line:
                in_jsonl = True
                continue
            if "=== AMORTIZED_JSONL_END ===" in line:
                in_jsonl = False
                continue
            if in_jsonl:
                stripped = line.strip()
                if stripped:
                    jsonl_lines.append(stripped)

        if not jsonl_lines:
            for line in log_lines:
                stripped = line.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        json.loads(stripped)
                        jsonl_lines.append(stripped)
                    except json.JSONDecodeError:
                        pass

        if not jsonl_lines:
            logger.warning("Job %s: no JSONL data found in container logs", job_id)
            return ""

        jsonl_content = "\n".join(jsonl_lines) + "\n"
        logger.info("Job %s: captured %d JSONL records from container", job_id, len(jsonl_lines))

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": experiment_name},
            )
            if resp.is_success:
                experiment_id = resp.json()["experiment"]["experiment_id"]
            else:
                resp = await client.post(
                    f"{tracking_uri}/api/2.0/mlflow/experiments/create",
                    json={"name": experiment_name},
                )
                resp.raise_for_status()
                experiment_id = resp.json()["experiment_id"]

            resp = await client.post(
                f"{tracking_uri}/api/2.0/mlflow/runs/create",
                json={
                    "experiment_id": experiment_id,
                    "run_name": f"sdg-{job_id[:8]}",
                    "tags": [
                        {"key": "job_type", "value": "sdg"},
                        {"key": "job_id", "value": job_id},
                        {"key": "record_count", "value": str(len(jsonl_lines))},
                    ],
                },
            )
            resp.raise_for_status()
            run_id = resp.json()["run"]["info"]["run_id"]

            run_info = resp.json()["run"]["info"]
            artifact_uri = run_info["artifact_uri"]
            s3_key = artifact_uri.split("/", 3)[3] if "://" in artifact_uri else artifact_uri
            s3_key = f"{s3_key}/generated_data/generated_data.jsonl"
            s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
            s3_bucket = artifact_uri.split("/")[2] if "://" in artifact_uri else "amortized"

            import boto3
            s3 = boto3.client(
                "s3",
                endpoint_url=s3_endpoint or None,
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
            s3.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=jsonl_content.encode(),
                ContentType="application/jsonl",
            )

            await client.post(
                f"{tracking_uri}/api/2.0/mlflow/runs/update",
                json={"run_id": run_id, "status": "FINISHED"},
            )

            logger.info(
                "Job %s: uploaded %d records to MLflow run %s",
                job_id, len(jsonl_lines), run_id,
            )
            return run_id

    except Exception:
        logger.warning("Job %s: failed to upload SDG results to MLflow", job_id, exc_info=True)
        return ""


async def _resolve_job_artifact_uri(job_id: str) -> str | None:
    """Look up a job's MLflow artifact URI by job ID."""
    if not job_id:
        return None
    repo = await _get_repo()
    job = await repo.get_job(job_id)
    if not job:
        return None
    run_id = job.get("mlflow_run_id", "")
    if not run_id:
        return None
    uri = await _resolve_mlflow_artifact_uri(run_id)
    return uri or None


async def _resolve_parent_artifacts(job: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
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
        logger.warning("Could not resolve artifacts for parent %s", parent_job_id)
        return config

    config = dict(config)
    if job["type"] == JobType.training.value and parent["type"] == "sdg":
        data_file = f"{artifact_uri}/generated_data/generated_data.jsonl"
        existing = config.get("data_path", "")
        if not existing or not existing.startswith("s3://"):
            config["data_path"] = data_file
            logger.info("Injected SDG data path from parent: %s", data_file)

    return config


async def _run_job(job: dict[str, Any]) -> None:
    job_id = job["id"]
    now = datetime.now(UTC).isoformat()
    config = dict(job["config"])

    secrets = config.pop("_secrets", {})

    backend_name = config_mod.settings.resolved_default_backend

    config = await _resolve_parent_artifacts(job, config)

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
                error=str(exc),
            )
            return

    spec_env: dict[str, str] = {}
    for env_name in config_mod.settings.forward_env:
        value = os.environ.get(env_name)
        if value:
            spec_env[env_name] = value

    llm_secret_keys = {"api_key", "api_secret", "token"}
    use_gateway = bool(config_mod.settings.gateway_url) and job["type"] == JobType.sdg.value
    secret_to_env = {"api_key": "OPENAI_API_KEY"}
    for secret_key, secret_val in secrets.items():
        if use_gateway and secret_key in llm_secret_keys:
            logger.info("Job %s: skipping %s injection (gateway configured)", job_id, secret_key)
            continue
        env_name = secret_to_env.get(secret_key, secret_key.upper())
        spec_env[env_name] = secret_val
    if use_gateway and "OPENAI_API_KEY" not in spec_env:
        spec_env["OPENAI_API_KEY"] = "gateway-managed"

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

    if image and job["type"] == JobType.training.value:
        algo_aliases = {"lora": "lora_sft", "qlora": "lora_sft", "qlora_sft": "lora_sft"}
        algorithm = config.get("algorithm", "sft")
        algorithm = algo_aliases.get(algorithm, algorithm)
        data_path = config.get("data_path", config.get("dataset", ""))
        if data_path.startswith("s3://"):
            local_name = data_path.split("/")[-1]
            local_path = f"/amortized/work/{local_name}"
            s3_downloads.append(S3Download(s3_uri=data_path, local_path=local_path))
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
            "model", "api_base", "api_key", "num_samples", "max_concurrency",
            "temperature", "max_tokens", "top_p", "seed", "num_retries",
            "input_data", "input_documents", "strategy_params",
            "task_description", "document_id", "output_dir",
        ):
            config.pop(stale_key, None)


        document_ids = config.pop("document_ids", []) or config.pop("document_id", [])
        if isinstance(document_ids, str):
            document_ids = [document_ids]
        doc_setup_cmds: list[str] = []
        if document_ids and config_mod.settings.mlflow_tracking_uri:
            doc_count = 0
            for doc_id in document_ids:
                content = await _fetch_document_content(doc_id)
                if content:
                    config_files[f"doc_{doc_count}.md"] = content
                    doc_count += 1
            if doc_count:
                doc_setup_cmds = [
                    "mkdir -p /tmp/docs",
                    *(
                        f"cp /amortized/doc_{i}.md /tmp/docs/"
                        for i in range(doc_count)
                    ),
                ]
                seed_config = config.get("seed_config", {})
                source = seed_config.get("source", {})
                source.setdefault("seed_type", "document-chunker")
                source.setdefault("path", "/tmp/docs")
                source.setdefault("file_extensions", [".md"])
                seed_config["source"] = source
                config["seed_config"] = seed_config
                logger.info(
                    "Job %s: loaded %d documents as seed data", job_id, doc_count
                )

        num_records = config.pop("num_records", 100)
        dd_config = {"data_designer": config}
        config_files["config.yaml"] = yaml.dump(
            dd_config, default_flow_style=False, sort_keys=False
        )
        dd_cmd = (
            "data-designer create /amortized/config.yaml"
            f" --num-records {num_records}"
            " --artifact-path /amortized/work"
            " --no-tui"
            " --output-format jsonl"
        )
        dump_cmd = (
            'echo "=== AMORTIZED_JSONL_START ==="'
            " && cat /amortized/work/dataset/dataset.jsonl"
            ' && echo "=== AMORTIZED_JSONL_END ==="'
        )
        all_cmds = [*doc_setup_cmds, dd_cmd, dump_cmd]
        cmd = ["sh", "-c", " && ".join(all_cmds)]

    job_type = job["type"]
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
            if job["type"] == JobType.sdg.value:
                mlflow_run_id = await _upload_sdg_results_to_mlflow(backend, handle, job)
            else:
                mlflow_run_id = await _extract_mlflow_run_id(backend, handle)
            if mlflow_run_id:
                await _set_mlflow_run_tag(mlflow_run_id, "job_type", job["type"])
                await _set_mlflow_run_tag(mlflow_run_id, "job_id", job_id)

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
            error_msg = status.error or f"Process exited with code {status.exit_code}"
            await _update_job(
                job_id,
                status=JobStatus.failed.value,
                completed_at=completed_at,
                error=error_msg,
            )
            logger.error("Job %s failed with code %s", job_id, status.exit_code)

    except Exception as exc:
        error_text = str(exc)
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
