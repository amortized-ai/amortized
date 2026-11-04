"""Prepare immutable input snapshots, credentials and model-serving leases."""

import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

import httpx

import amortized.config as config_mod
from amortized.backends import JobSpec, Resources
from amortized.backends.kubernetes import KubernetesBackend
from amortized.core.compute import get_backend
from amortized.core.jobs import deserialize_handle
from amortized.core.mlflow_client import MLflowClient
from amortized.db.evalhub import EvalHubRepository
from amortized.integrations.evalhub import client
from amortized.jobs.base import JobBuildError
from amortized.jobs.eval import _ASSETS_DIR, _build_embedded_serving, _endpoint_spec


async def _artifact_paths(mlflow: MLflowClient, run_id: str, root: str) -> list[str]:
    paths = []
    for entry in await mlflow.list_artifacts(run_id, root):
        path = str(entry["path"])
        if not path.startswith(root + "/") or ".." in PurePosixPath(path).parts:
            raise JobBuildError("Unsafe source artifact path")
        if entry.get("is_dir"):
            paths.extend(await _artifact_paths(mlflow, run_id, path))
        else:
            paths.append(path)
    return sorted(paths)


async def stage_inputs(job: dict[str, Any], repo: Any) -> tuple[dict[str, Any], str, str]:
    settings = config_mod.settings
    bucket = settings.evalhub_staging_bucket
    if not bucket or not settings.mlflow_tracking_uri:
        raise JobBuildError("Hub task evaluation requires staging bucket and MLflow configuration")
    config = job["config"]
    data_run = config.get("eval_data_run_id")
    if job.get("parent_job_id"):
        parent = await repo.get_job(job["parent_job_id"])
        if not parent or parent["status"] != "succeeded":
            raise JobBuildError("Evaluation data producer must still be succeeded")
        data_run = parent.get("mlflow_run_id") or data_run
    if not data_run:
        raise JobBuildError("No dataset artifact run is available")
    mlflow = MLflowClient(settings.mlflow_tracking_uri)
    sources = [
        (str(data_run), p, "data/" + str(PurePosixPath(p).relative_to("generated_data")))
        for p in await _artifact_paths(mlflow, str(data_run), "generated_data")
        if p.endswith((".jsonl", ".parquet"))
    ]
    if not sources:
        raise JobBuildError("Dataset has no JSONL/parquet artifacts")
    model_path = str(config.get("model_name_or_path") or "")
    if config.get("eval_mode") == "classification" and config.get("training_job_id"):
        training = await repo.get_job(config["training_job_id"])
        if not training or training["status"] != "succeeded" or not training.get("mlflow_run_id"):
            raise JobBuildError("Embedding training job must be succeeded with model artifacts")
        run_id = training["mlflow_run_id"]
        models = await _artifact_paths(mlflow, run_id, "model")
        if not models:
            raise JobBuildError("Embedding training run has no model artifacts")
        sources.extend((run_id, p, p) for p in models)
        model_path = "/test_data/model"

    import boto3

    s3 = boto3.client("s3", endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL") or None)
    manifest = []
    # Stream one artifact at a time to a temporary file. In particular, never
    # buffer an embedding checkpoint's weights in the API process's RAM.
    with TemporaryDirectory(prefix="amortized-eval-") as directory:
        for index, (run_id, source, destination) in enumerate(sources):
            local = Path(directory) / str(index)
            digest = hashlib.sha256()
            prefix = await mlflow._resolve_artifact_prefix(run_id)
            async with (
                mlflow._client() as transport,
                transport.stream(
                    "GET", mlflow._url(f"/api/2.0/mlflow-artifacts/artifacts/{prefix}/{source}")
                ) as response,
            ):
                response.raise_for_status()
                with local.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        digest.update(chunk)
                        output.write(chunk)
            manifest.append(
                {
                    "path": destination,
                    "sha256": digest.hexdigest(),
                    "run_id": run_id,
                    "source": source,
                }
            )
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        dataset_identity = hashlib.sha256(
            json.dumps(
                [
                    {"path": e["path"], "sha256": e["sha256"]}
                    for e in manifest
                    if e["path"].startswith("data/")
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        key = f"amortized/evaluations/{job['id']}/inputs/{identity}/"
        for index, entry in enumerate(manifest):
            await asyncio.to_thread(
                s3.upload_file, str(Path(directory) / str(index)), bucket, key + entry["path"]
            )
        await asyncio.to_thread(
            s3.put_object,
            Bucket=bucket,
            Key=key + "manifest.json",
            Body=json.dumps(manifest).encode(),
        )
    return (
        {"s3": {"bucket": bucket, "key": key, "secret_ref": settings.evalhub_s3_secret}},
        model_path,
        dataset_identity,
    )


async def _credential_secret(
    job: dict[str, Any],
    store: EvalHubRepository,
    endpoint: dict[str, Any],
    judge: dict[str, Any] | None,
) -> str:
    from kubernetes_asyncio.client import ApiException, CoreV1Api, V1ObjectMeta, V1Secret

    backend = get_backend("kubernetes")
    if not isinstance(backend, KubernetesBackend):
        raise JobBuildError("Expected Kubernetes compute backend")
    api = CoreV1Api(await backend._get_client())
    name = f"amortized-eval-{job['id']}-auth"
    # Record intent before creation so restart/failure cleanup can find the resource.
    await store.update(job["id"], secret_name=name)
    env: dict[str, str] = {}
    from amortized.core.model_catalog import inject_enabled_provider_keys

    inject_enabled_provider_keys(env)
    model = _endpoint_spec({"endpoint": endpoint}, "endpoint", "EVAL_MODEL_API_KEY", env)
    data = {
        "api-key": env.get(model["api_key_env"], "")
        or os.getenv(model["api_key_env"], "")
        or "unused-anonymous-key"
    }
    # Match the native job's explicit credential-forwarding policy. Hub projects
    # this supported key into the adapter for gated embedding model downloads.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if key in config_mod.settings.forward_env and os.getenv(key):
            data["hf-token"] = os.environ[key]
            break
    if judge:
        spec = _endpoint_spec({"judge": judge}, "judge", "EVAL_JUDGE_API_KEY", env)
        data.update(
            {
                "judge_url": spec["base_url"],
                "judge_api-key": env.get(spec["api_key_env"], "")
                or os.getenv(spec["api_key_env"], "")
                or "unused-anonymous-key",
            }
        )
    secret = V1Secret(
        metadata=V1ObjectMeta(name=name, labels={"amortized/job-id": job["id"]}), string_data=data
    )
    try:
        await api.create_namespaced_secret(config_mod.settings.compute_namespace, secret)
    except ApiException as exc:
        if exc.status != 409:
            raise
        await api.replace_namespaced_secret(name, config_mod.settings.compute_namespace, secret)
    return name


async def prepare_serving(
    job: dict[str, Any], store: EvalHubRepository, execution: dict[str, Any]
) -> dict[str, Any]:
    from amortized.worker import _inject_job_mlflow_auth, _serialize_handle, _wrap_command

    config = job["config"]
    backend = get_backend("kubernetes")
    if not isinstance(backend, KubernetesBackend):
        raise JobBuildError("Expected Kubernetes compute backend")
    serving_id = f"{job['id']}-serve"
    handle = deserialize_handle(execution.get("serving_handle"))
    if handle is None:
        handle = backend.reconstruct_handle(serving_id)
        await store.update(job["id"], serving_handle=_serialize_handle(handle))
    status = await backend.status(handle)
    port = int(config.get("port", 8000))
    if status.not_found:
        files: dict[str, str] = {}
        settings = config_mod.settings
        env = {key: os.environ[key] for key in settings.forward_env if os.getenv(key)}
        if settings.mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
            if endpoint := os.getenv("MLFLOW_S3_ENDPOINT_URL"):
                env["MLFLOW_S3_ENDPOINT_URL"] = endpoint
                env["FSSPEC_S3_ENDPOINT_URL"] = endpoint
            for key in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "AWS_S3_BUCKET",
                "AWS_REGION",
                "AWS_DEFAULT_REGION",
            ):
                if value := os.getenv(key):
                    env[key] = value
            if settings.mlflow_proxy_multipart_download:
                env["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "true"
        auth = _inject_job_mlflow_auth(env, files)
        commands, script, _extras = await _build_embedded_serving(config, env)
        # Reuse model preparation and startup checks, then keep serving. There
        # is no evaluator command here: Hub owns the evaluation workload.
        script = script.split('echo "=== EVAL-STAGE: evaluating ==="', 1)[0] + "wait $SERVE_PID\n"
        for asset in (
            "serve_vllm.py",
            "patch_model_config.py",
            "merge_lora.py",
            "check_gpu_memory.py",
            "wait_for_endpoint.py",
        ):
            files[asset] = (Path(_ASSETS_DIR) / asset).read_text()
        if auth:
            commands.insert(0, auth)
        handle = await backend.submit(
            JobSpec(
                job_id=serving_id,
                job_type="eval-model",
                user_id=job.get("user_id", ""),
                command=_wrap_command(["sh", "-c", "{\n" + script + "\n}"], commands, []),
                image=config_mod.job_image("eval"),
                config_files=files,
                env=env,
                resources=Resources(
                    gpus=int(config.get("nproc_per_node", 1)), cpus=4, memory_gb=16
                ),
                ports={port: port},
            )
        )
        await store.update(job["id"], serving_handle=_serialize_handle(handle))
    elif status.exit_code is not None or status.reclaim:
        raise JobBuildError("Owned model-serving resource stopped; see preparation logs")
    elif not status.running:
        raise ConnectionError("Model-serving status is temporarily unavailable")
    host = f"{handle.scheduler_id}.{config_mod.settings.compute_namespace}.svc"
    url = f"http://{host}:{port}/v1"
    async with httpx.AsyncClient(timeout=5) as transport:
        response = await transport.get(url + "/models")
        response.raise_for_status()
        models = response.json().get("data") or []
    if not models:
        raise httpx.ConnectError("Prepared endpoint has not exposed a model yet")
    return {"base_url": url, "model": models[0]["id"]}


def _approved_recipe(definition: dict[str, Any], provider: dict[str, Any]) -> dict[str, Any]:
    """The recipe stored at Confirm, if the registered adapter is still the approved one."""
    from amortized.integrations.evalhub.catalog import image_digest

    recipe = definition.get("recipe")
    if not isinstance(recipe, dict) or not recipe.get("adapter_digest"):
        raise JobBuildError(
            "This benchmark job has no approved recipe; validate and confirm it again"
        )
    runtime = (provider.get("runtime") or {}).get("k8s") or {}
    # Hub 1.0.3 serializes the image as "Image"; provider YAML uses "image".
    if image_digest(runtime.get("Image") or runtime.get("image") or "") != recipe["adapter_digest"]:
        raise JobBuildError(
            "The registered benchmark adapter image changed after approval;"
            " validate and confirm the benchmark again"
        )
    return recipe


async def prepare(
    job: dict[str, Any], store: EvalHubRepository, execution: dict[str, Any]
) -> dict[str, Any]:
    settings = config_mod.settings
    if settings.resolved_default_backend != "kubernetes":
        raise JobBuildError(
            "Hub integration requires Kubernetes for input/credential/model preparation"
        )
    config = job["config"]
    definition = config.get("evaluation") or {}
    benchmark = definition.get("kind") == "benchmark"
    inputs = execution.get("inputs") or {}
    if "catalog" not in inputs:
        provider_id = definition["provider_id"] if benchmark else settings.evalhub_task_provider
        async with client.client(execution["hub_url"], execution["tenant"]) as sdk:
            provider = await sdk.providers.get(provider_id)
        inputs["catalog"] = provider.model_dump(mode="json", exclude_none=True)
        await store.update(job["id"], inputs=inputs)
    if benchmark:
        recipe = _approved_recipe(definition, inputs["catalog"])
        data_ref, model_path = None, ""
    else:
        if "data_ref" not in inputs:
            data_ref, model_path, identity = await stage_inputs(job, store.repo)
            inputs.update(
                data_ref=data_ref,
                model_path=model_path,
                dataset_identity=identity,
                dataset_run_id=config.get("eval_data_run_id") or "",
            )
            if job.get("parent_job_id"):
                parent = await store.repo.get_job(job["parent_job_id"])
                inputs["dataset_run_id"] = (parent or {}).get("mlflow_run_id") or ""
            await store.update(job["id"], inputs=inputs)
        data_ref, model_path = inputs["data_ref"], inputs["model_path"]
    if config.get("eval_mode") == "classification":
        endpoint = {"base_url": "http://unused.invalid", "model": model_path}
    elif config.get("endpoint"):
        env: dict[str, str] = {}
        resolved = _endpoint_spec(config, "endpoint", "EVAL_MODEL_API_KEY", env)
        endpoint = {**config["endpoint"], "base_url": resolved["base_url"]}
    else:
        endpoint = await prepare_serving(job, store, execution)
    secret = await _credential_secret(job, store, endpoint, config.get("judge"))
    if benchmark:
        # Built from the recipe approved on the card, never from later edits.
        selected = {
            "id": recipe["benchmark_id"],
            "provider_id": recipe["provider_id"],
            "parameters": dict(recipe["parameters"]),
        }
        if definition.get("test_data_ref"):
            selected["test_data_ref"] = definition["test_data_ref"]
    else:
        # Only accepted semantic configuration crosses the engine boundary.
        keys = (
            "rubric",
            "max_samples",
            "judge_max_samples",
            "temperature",
            "eval_mode",
            "class_labels",
            "text_column",
            "label_column",
            "anchors_per_class",
            "top_k",
            "tau",
            "seed",
            "evaluation",
        )
        parameters = {key: config[key] for key in keys if key in config}
        parameters["dataset_identity"] = inputs["dataset_identity"]
        if config.get("judge"):
            parameters["judge_model"] = config["judge"]["model"]
        selected = {
            "id": "task",
            "provider_id": settings.evalhub_task_provider,
            "parameters": parameters,
            "test_data_ref": data_ref,
        }
    return {
        "name": execution["correlation"],
        "tags": [execution["correlation"]],
        "model": {
            "url": endpoint["base_url"],
            "name": endpoint["model"],
            "auth": {"secret_ref": secret},
        },
        "benchmarks": [selected],
        "experiment": {
            "name": settings.evalhub_experiment,
            "tags": [{"key": "amortized_job_id", "value": job["id"]}],
        },
    }


async def cleanup(execution: dict[str, Any]) -> None:
    """Release owned dependencies only after the evaluator is terminal/never submitted."""
    from kubernetes_asyncio.client import ApiException, CoreV1Api

    handle = deserialize_handle(execution.get("serving_handle"))
    if handle:
        backend = get_backend(handle.backend_name)
        await backend.cancel(handle)
        status = await backend.status(handle)
        if status.running or (status.exit_code is None and not status.not_found):
            raise ConnectionError("Waiting for owned model server to stop")
    if execution.get("secret_name"):
        backend = get_backend("kubernetes")
        if not isinstance(backend, KubernetesBackend):
            raise JobBuildError("Expected Kubernetes compute backend")
        api = CoreV1Api(await backend._get_client())
        try:
            await api.delete_namespaced_secret(
                execution["secret_name"], config_mod.settings.compute_namespace
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
