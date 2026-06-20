# PRD: Amortized on OpenShift

## Problem

Amortized runs on a developer's laptop and dispatches jobs to GPU nodes via SSH. This works for one person but can't be shared with a team, doesn't survive laptop reboots, and requires direct SSH access to GPU machines. Platform engineers can't install it. Data scientists can't use it without terminal access.

The target deployment is any OpenShift cluster with GPUs. No OpenShift AI required.

## Solution

Deploy amortized as a containerized application on OpenShift. Replace the SSH compute backend with Kubernetes Job dispatch. Use MLflow for artifact management and lineage tracking, backed by S3. Everything else — config generation, recipes, agent, evaluation — stays the same.

```
┌───────────────────────────────────────────────────────┐
│  Namespace: amortized                                 │
│                                                       │
│  ┌──────────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ amortized-server │  │ studio   │  │ MLflow      │ │
│  │ (Deployment)     │  │ (Deploy) │  │ (Deployment)│ │
│  │ FastAPI + worker  │  │ nginx    │  │ tracking +  │ │
│  │ PVC: /data       │  │ React    │  │ artifact    │ │
│  │  (SQLite + logs) │  │          │  │ server      │ │
│  └───────┬──────────┘  └────┬─────┘  └──────┬──────┘ │
│          │ :8000             │ :8080         │ :5000  │
│          │                   │ Route         │ Route  │
│          │                   │               │        │
│          │  ┌────────────────┴───────────────┘        │
│          │  │ Studio links to MLflow UI               │
│          │  │ Server logs to MLflow API               │
│          │                                            │
│  ┌───────▼────────────────────────────────────────┐   │
│  │ KubernetesBackend                              │   │
│  │ creates Jobs + Deployments in:                 │   │
│  └───────┬────────────────────────────────────────┘   │
│          │                                            │
├──────────▼────────────────────────────────────────────┤
│  Namespace: amortized-jobs                            │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ SDG Job  │  │Train Job │  │ Serve Deployment │    │
│  │ (CPU)    │  │ (L40S)   │  │ (L40S, vLLM)    │    │
│  │ asynth   │  │ TRL      │  │ long-running     │    │
│  └──────────┘  └──────────┘  └──────────────────┘    │
│       │              │                │               │
│       │   MLFLOW_TRACKING_URI env var │               │
│       │   report_to: mlflow           │               │
│       └──────────────┴────────────────┘               │
│                      │ log artifacts + metrics        │
└──────────────────────┼────────────────────────────────┘
                       ▼
               ┌──────────────┐
               │  S3 Bucket   │
               │  (AWS / MinIO)│
               │              │
               │  MLflow       │
               │  artifact     │
               │  store:       │
               │  /<exp>/<run>/│
               │    model/     │
               │    data/      │
               │    metrics/   │
               └──────────────┘
```

## Target Environment

| Requirement | Value |
|---|---|
| Platform | OpenShift 4.14+ (any — self-managed, ROSA, ARO, OCP) |
| GPUs | NVIDIA GPU Operator installed, `nvidia.com/gpu` resource available |
| Storage | MinIO on-cluster (default) or any S3-compatible endpoint (AWS S3, Ceph RGW) |
| PVC | Default StorageClass for server PVC (SQLite + logs only) |
| Registry | Access to ghcr.io (or mirror images to internal registry) |
| Auth | OpenShift OAuth (built-in, not RHOAI) |
| NOT required | OpenShift AI, KServe, Model Registry, Training Operator, Kueue, MLflow |

Validated against: ROSA on AWS, OpenShift 4.20.21, 4x NVIDIA L40S (48GB), gp3-csi StorageClass.

## What Changes

### Replaced by OpenShift primitives

| Component | Lines Eliminated | Replaced By |
|---|---|---|
| SSH backend (`backends/ssh.py`) | ~400 | K8s `batch/v1 Job` + `apps/v1 Deployment` |
| podman secret create/rm | ~60 | K8s Secrets mounted as env vars |
| SFTP artifact download | ~80 | Shared PVC or S3 |
| Container runtime detection | ~30 | K8s pod spec (image + command) |
| Serve job monitor (`_monitor_serve_job`) | ~50 | K8s Deployment with health checks |
| Port mapping / network detection | ~30 | K8s Service + Route |
| Heartbeat probe logic | ~40 | K8s Job status API |
| Events URL resolution | ~20 | K8s Service DNS (deterministic) |
| Bearer token auth middleware | ~30 | OpenShift OAuth proxy sidecar |
| **Total** | **~740** | **K8s API calls** |

### Stays the same

| Component | Why |
|---|---|
| Config translation | Product logic — translates amortized job configs into tool-native YAML (TRL, asynth, vLLM) |
| Agent/chat | Unique value — guided workflow via function-calling |
| Recipes/templates | Product logic — composable YAML configs with `extends:` |
| MCP server | Already HTTP-based, no change |
| Studio UI | Containerized with nginx, served via Route |
| Job state machine | Core orchestration — simplified without SSH but same states |
| CLI/SDK | REST clients — point at Route URL instead of localhost |

## Modules

### Module 1: KubernetesBackend

New file: `src/amortized/backends/kubernetes.py`

Implements the existing `ComputeBackend` protocol using `kubernetes_asyncio`.

**Architecture: config only, no code generation.**

The control plane translates amortized job configs into tool-native YAML configs and creates K8s Jobs that run the tool's own CLI. No Python scripts are generated.

| Job Type | Container Image | Command | Config |
|---|---|---|---|
| Training (sft, dpo, kto, grpo) | `training:latest` | `trl {algorithm} --config /amortized/config.yaml` | TRL-native YAML |
| Training (gkd, gold) | `training:latest` | `amortized-train --config /amortized/config.yaml` | Amortized training config (thin entrypoint for experimental algos) |
| SDG | `asynth:latest` | `asynth synthesize --config /amortized/config.yaml` | asynth-native YAML |
| Eval | `asynth:latest` | `asynth judge --config /amortized/config.yaml` | asynth judge config |
| Serve | `vllm-openai` | `--config /amortized/config.yaml` | vLLM-native YAML |

**Training / SDG / Eval jobs → K8s `batch/v1 Job`:**

```python
async def submit(self, spec: JobSpec) -> BackendHandle:
    # 1. Create ConfigMap with the tool-native config
    config_map = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(
            name=f"amortized-{spec.job_id[:8]}-config",
            namespace=self.namespace,
            labels={"app": "amortized", "amortized/job-id": spec.job_id},
        ),
        data={"config.yaml": spec.config_yaml},
    )
    await self._core.create_namespaced_config_map(
        namespace=self.namespace, body=config_map,
    )

    # 2. Create Job that mounts the config and runs the tool CLI
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=f"amortized-{spec.job_id[:8]}",
            namespace=self.namespace,
            labels={"app": "amortized", "amortized/job-id": spec.job_id},
        ),
        spec=client.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=3600,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[client.V1Container(
                        name="worker",
                        image=spec.image,
                        command=spec.command,  # e.g. ["trl", "sft", "--config", "/amortized/config.yaml"]
                        env=self._build_env(spec),
                        resources=self._build_resources(spec),
                        volume_mounts=[
                            client.V1VolumeMount(name="config", mount_path="/amortized", read_only=True),
                            client.V1VolumeMount(name="work", mount_path="/amortized/work"),
                            client.V1VolumeMount(name="shm", mount_path="/dev/shm"),
                        ],
                    )],
                    volumes=[
                        client.V1Volume(
                            name="config",
                            config_map=client.V1ConfigMapVolumeSource(
                                name=config_map.metadata.name,
                            ),
                        ),
                        client.V1Volume(name="work", empty_dir=client.V1EmptyDirVolumeSource()),
                        client.V1Volume(name="shm", empty_dir=client.V1EmptyDirVolumeSource(
                            medium="Memory", size_limit="12Gi",
                        )),
                    ],
                    node_selector={"nvidia.com/gpu.present": "true"} if spec.resources.gpus else None,
                ),
            ),
        ),
    )
    # Set ownerReference so ConfigMap is auto-deleted when Job is cleaned up
    config_map.metadata.owner_references = [client.V1OwnerReference(
        api_version="batch/v1", kind="Job",
        name=job.metadata.name, uid=job.metadata.uid,
    )]
    await self._batch.create_namespaced_job(namespace=self.namespace, body=job)
```

**Serve jobs → K8s `apps/v1 Deployment` + `v1 Service`:**

Serve jobs are fundamentally different from training/SDG/eval — they run continuously until explicitly stopped. K8s Deployments handle this natively: auto-restart on crash, rolling updates, stable DNS via Service.

```python
async def _submit_deployment(self, spec: JobSpec) -> BackendHandle:
    labels = {"app": "amortized", "amortized/job-id": spec.job_id}

    # Deployment (long-running, K8s auto-restarts crashed pods)
    deployment = client.V1Deployment(
        metadata=client.V1ObjectMeta(
            name=f"amortized-serve-{spec.job_id[:8]}",
            namespace=self.namespace,
            labels=labels,
        ),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels=labels),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=client.V1PodSpec(
                    containers=[client.V1Container(
                        name="vllm",
                        image=spec.image,
                        command=spec.command,
                        env=self._build_env(spec),
                        env_from=self._build_env_from(spec),
                        resources=self._build_resources(spec),
                        volume_mounts=[
                            client.V1VolumeMount(name="config", mount_path="/amortized"),
                            client.V1VolumeMount(name="shm", mount_path="/dev/shm"),
                        ],
                        ports=[client.V1ContainerPort(container_port=8000)],
                    )],
                    volumes=[
                        client.V1Volume(name="config", config_map=...),
                        client.V1Volume(name="shm", empty_dir=client.V1EmptyDirVolumeSource(
                            medium="Memory", size_limit="12Gi",
                        )),
                    ],
                    node_selector={"nvidia.com/gpu.present": "true"},
                ),
            ),
        ),
    )
    await self._apps.create_namespaced_deployment(namespace=self.namespace, body=deployment)

    # Service (stable endpoint for other pods to reach vLLM)
    service = client.V1Service(
        metadata=client.V1ObjectMeta(
            name=f"amortized-serve-{spec.job_id[:8]}",
            namespace=self.namespace,
        ),
        spec=client.V1ServiceSpec(
            selector=labels,
            ports=[client.V1ServicePort(port=8000, target_port=8000)],
        ),
    )
    await self._core.create_namespaced_service(namespace=self.namespace, body=service)

    return BackendHandle(
        handle_id=deployment.metadata.name,
        kind="deployment",
        endpoint=f"http://{service.metadata.name}.{self.namespace}.svc.cluster.local:8000",
    )
```

**Job lifecycle by type:**

| | Training / SDG / Eval | Serve |
|---|---|---|
| **K8s resource** | `batch/v1 Job` | `apps/v1 Deployment` + `v1 Service` |
| **Runs until** | Container exits | Explicitly cancelled |
| **On crash** | Job fails (backoffLimit=0) | K8s auto-restarts the pod |
| **Terminal states** | Succeeded, Failed | None — always running or cancelled |
| **Cleanup** | Automatic (`ttl_seconds_after_finished=3600`) | Manual (`cancel()` deletes Deployment + Service) |
| **Monitoring** | Worker polls Job status every 2s | Not needed — K8s handles restarts. Worker only checks on user cancel. |
| **Endpoint** | None | `http://amortized-serve-{id}.amortized-jobs.svc.cluster.local:8000` |

**Status (both types):**

```python
async def status(self, handle: BackendHandle) -> BackendStatus:
    if handle.kind == "deployment":
        deploy = await self._apps.read_namespaced_deployment(
            name=handle.handle_id, namespace=self.namespace)
        available = next(
            (c for c in (deploy.status.conditions or []) if c.type == "Available"), None)
        if available and available.status == "True":
            return BackendStatus(state="running")
        # Check if pods are crash-looping
        pods = await self._core.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"amortized/job-id={handle.job_id}")
        for pod in pods.items:
            for cs in (pod.status.container_statuses or []):
                if cs.state.waiting and cs.state.waiting.reason == "CrashLoopBackOff":
                    return BackendStatus(state="failed", error="CrashLoopBackOff")
        return BackendStatus(state="provisioning")  # waiting for GPU

    # Job status
    job = await self._batch.read_namespaced_job(
        name=handle.handle_id, namespace=self.namespace)
    if job.status.succeeded:
        return BackendStatus(state="succeeded", exit_code=0)
    if job.status.failed:
        return BackendStatus(state="failed", exit_code=1)
    if job.status.active:
        return BackendStatus(state="running")
    return BackendStatus(state="queued")
```

**Cancel (both types):**

```python
async def cancel(self, handle: BackendHandle) -> None:
    if handle.kind == "deployment":
        await self._apps.delete_namespaced_deployment(
            name=handle.handle_id, namespace=self.namespace)
        await self._core.delete_namespaced_service(
            name=handle.handle_id, namespace=self.namespace)
        return
    await self._batch.delete_namespaced_job(
        name=handle.handle_id, namespace=self.namespace,
        body=client.V1DeleteOptions(propagation_policy="Foreground"))
```

**Logs (both types):** Same K8s API — `read_namespaced_pod_log` with label selector `amortized/job-id={id}`.

**What this eliminates:** `_monitor_serve_job()` background task (~50 lines). K8s Deployment handles pod restarts automatically.

**Events URL:** `http://amortized-server.amortized.svc.cluster.local:8000/api/v1/events/ingest` — deterministic via K8s Service DNS. Injected as env var in all job/deployment pods.

**In-cluster auth:** `config.load_incluster_config()` — uses the ServiceAccount token mounted into the server pod.

**Dependencies:** `kubernetes_asyncio`

### Module 2: Server Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -e '.[dev]'

USER 1001
ENV AMORTIZED_DB_PATH=/data/amortized.db \
    AMORTIZED_DATA_DIR=/data

EXPOSE 8000
CMD ["uvicorn", "amortized.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- Non-root user (UID 1001) for OpenShift restricted SCC
- PVC mounted at `/data` for SQLite + artifacts
- Recipes, examples, templates bundled in the image

### Module 3: Studio Dockerfile

```dockerfile
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 8080
```

nginx config:

```nginx
server {
    listen 8080;

    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://${BACKEND_HOST}:${BACKEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;  # SSE streaming
    }
}
```

- `nginx-unprivileged` for OpenShift restricted SCC (port 8080)
- SPA fallback (`try_files`)
- Reverse proxy to server with WebSocket upgrade and SSE timeout

### Module 4: MLflow Deployment

MLflow is deployed as part of amortized — not expected to pre-exist on the cluster. It provides:

- **Experiment tracking** — loss curves, hyperparameters, run comparison
- **Artifact store** — all artifacts (models, datasets, eval results) stored in S3 via MLflow
- **Model Registry** — model versioning, staging labels, deployment promotion
- **Lineage** — `mlflow.log_input()` links datasets to training runs to models
- **UI** — browsable via OpenShift Route, linked from Studio

**Architecture:**

```yaml
# MLflow server deployment
image: ghcr.io/mlflow/mlflow:latest
command: ["mlflow", "server",
  "--backend-store-uri", "sqlite:///mlflow/mlflow.db",
  "--default-artifact-root", "s3://amortized/mlflow/",
  "--host", "0.0.0.0", "--port", "5000"]
```

- **Backend store**: SQLite on PVC (sufficient for single-team use). PostgreSQL upgrade path exists for scale.
- **Artifact store**: S3 — same bucket as amortized, under `mlflow/` prefix. MLflow manages the `/<experiment_id>/<run_id>/artifacts/` structure.
- **PVC**: 10Gi for MLflow's SQLite DB (separate from server PVC).

**How each job type integrates:**

| Job Type | MLflow Integration | How |
|---|---|---|
| **Training** | Automatic via TRL MLflowCallback | Set `report_to: mlflow` + inject `MLFLOW_TRACKING_URI` env var. TRL auto-logs loss, learning rate, epoch, checkpoints. `HF_MLFLOW_LOG_ARTIFACTS=true` copies model weights to S3 via MLflow. |
| **SDG** | Explicit logging in runner script | Runner calls `mlflow.start_run()`, `mlflow.log_params()`, `mlflow.log_artifact()`, `mlflow.log_input()` for lineage. |
| **Eval** | Explicit logging in eval script | Runner logs eval metrics + results as artifacts. Links to training run via `mlflow.log_input()`. |
| **Serve** | Model Registry reference | Serve config references `models:/<name>/<version>` URI from MLflow Model Registry. |

**Lineage flow:**

```
SDG Run                    Training Run                Eval Run
mlflow.start_run()         mlflow.start_run()          mlflow.start_run()
  log_params(model,          log_params(lr, epochs,      log_params(judge,
    num_samples, temp)         lora_r, algorithm)           dataset_size)
  log_artifact(data.jsonl)   log_input(dataset,          log_input(model,
  log_input(→ source)          context="training")          context="eval")
                             log_artifact(model/)        log_metrics(accuracy,
                             register_model(→ registry)    f1, judge_pass)
```

This gives full traceability: which data was used → to train which model → with what hyperparams → producing what accuracy. Browsable in MLflow UI.

**Env vars injected into job containers:**

```python
env = {
    "MLFLOW_TRACKING_URI": f"http://mlflow.amortized.svc.cluster.local:5000",
    "MLFLOW_EXPERIMENT_NAME": f"amortized/{job_type}/{job_id[:8]}",
    "HF_MLFLOW_LOG_ARTIFACTS": "true",  # training only
}
```

**What this replaces in amortized:**

| Current | Replaced By |
|---|---|
| SQLite `artifacts` table | MLflow artifact store (S3-backed) |
| `artifact:<uuid>` reference system | MLflow `runs:/<run_id>/<path>` URIs |
| Custom `register_artifacts_for_job()` | `mlflow.log_artifact()` in container runners |
| `training_metrics.jsonl` file tailing | MLflow tracking API (metrics already there) |
| No run comparison | MLflow UI run comparison |
| No lineage | `mlflow.log_input()` dataset→run links |

**What amortized keeps:**

- SQLite `jobs` table — job queue, status, scheduling (MLflow doesn't do job orchestration)
- SQLite `conversations` + `messages` — agent chat history (nothing to do with MLflow)
- SQLite `evaluators` — evaluator definitions (product logic)
- `events` table — real-time job events for SSE streaming (MLflow is not real-time)

**Studio integration:**

- "Open in MLflow" button on job detail pages → links to `{MLFLOW_URI}/#/experiments/{exp}/runs/{run}`
- Don't rebuild MLflow's comparison UI — link to it
- Training metrics in Studio can optionally query MLflow API instead of tailing `training_metrics.jsonl`

### Module 5: K8s Manifests

All in `k8s/` directory. Applied with `oc apply -f k8s/`.

| File | Resource | Purpose |
|---|---|---|
| `namespace.yaml` | 2 Namespaces | `amortized` (control plane), `amortized-jobs` (compute) |
| `server-deployment.yaml` | Deployment | amortized-server, 1 replica, PVC mount |
| `server-service.yaml` | Service | ClusterIP on port 8000 |
| `studio-deployment.yaml` | Deployment | amortized-studio (nginx), 1 replica |
| `studio-service.yaml` | Service | ClusterIP on port 8080 |
| `studio-route.yaml` | Route | TLS edge termination, external access |
| `mlflow-deployment.yaml` | Deployment | MLflow tracking server, 1 replica, PVC mount |
| `mlflow-service.yaml` | Service | ClusterIP on port 5000 |
| `mlflow-route.yaml` | Route | TLS edge termination, MLflow UI access |
| `minio-deployment.yaml` | Deployment | MinIO S3-compatible object store, 1 replica, PVC mount |
| `minio-service.yaml` | Service | ClusterIP on port 9000 (API) + 9001 (console) |
| `minio-pvc.yaml` | PVC | 200Gi gp3-csi for MinIO data (model weights, datasets, artifacts). Expandable online. |
| `mlflow-pvc.yaml` | PVC | 10Gi gp3-csi for MLflow SQLite |
| `server-pvc.yaml` | PVC | 10Gi gp3-csi for server SQLite + logs |
| `serviceaccount.yaml` | ServiceAccount | `amortized-server` in `amortized` namespace |
| `rbac.yaml` | Role + RoleBinding | Create/watch/delete Jobs, Deployments, Services, Secrets, ConfigMaps in `amortized-jobs` |
| `configmap.yaml` | ConfigMap | `AMORTIZED_COMPUTE_BACKEND=kubernetes`, MLflow URI, namespace, registry |
| `s3-secret.yaml` | Secret | MinIO credentials — applied to both `amortized` and `amortized-jobs` namespaces |

### Module 6: Secret Management

Two categories of secrets, managed differently.

**Infrastructure secrets (platform engineer, install-time):**

S3/MinIO credentials needed by MLflow, server, and job pods. Defined in manifests, applied to both namespaces:

```yaml
# k8s/s3-secret.yaml — applied to both namespaces
apiVersion: v1
kind: Secret
metadata:
  name: amortized-s3
  namespace: amortized        # also create in amortized-jobs
stringData:
  AWS_ACCESS_KEY_ID: "minioadmin"
  AWS_SECRET_ACCESS_KEY: "minioadmin"
  AWS_S3_BUCKET: "amortized"
  AWS_S3_ENDPOINT: "http://minio.amortized.svc.cluster.local:9000"
  AWS_S3_ENDPOINT_URL: "http://minio.amortized.svc.cluster.local:9000"
  AWS_DEFAULT_REGION: "us-east-2"
```

Server, MLflow, and job pods all reference via `envFrom.secretRef`.

**User LLM keys (data scientist, via Studio):**

API keys for LLM providers (OpenAI, Anthropic, HuggingFace). Managed by data scientists through Studio's settings UI → stored encrypted in amortized's `api_keys` table. At job dispatch time, the KubernetesBackend creates a per-job K8s Secret:

```python
# In KubernetesBackend — at job dispatch time:

# 1. Read user's LLM keys from amortized DB (already encrypted at rest)
api_keys = await self._db.get_api_keys()
# → {"OPENAI_API_KEY": "sk-...", "HF_TOKEN": "hf_..."}

# 2. Create per-job Secret in jobs namespace
job_secret = client.V1Secret(
    metadata=client.V1ObjectMeta(
        name=f"amortized-{spec.job_id[:8]}-keys",
        namespace=self.namespace,
    ),
    string_data=api_keys,
)
await self._core.create_namespaced_secret(
    namespace=self.namespace, body=job_secret,
)

# 3. Job pod references both infrastructure + per-job secrets
env_from=[
    client.V1EnvFromSource(
        secret_ref=client.V1SecretEnvSource(name="amortized-s3"),
    ),
    client.V1EnvFromSource(
        secret_ref=client.V1SecretEnvSource(name=job_secret.metadata.name),
    ),
],
```

Per-job Secrets are cleaned up automatically — `ownerReferences` on the Secret point to the Job, so when the Job is deleted (via `ttl_seconds_after_finished`), the Secret goes with it.

**RBAC:** The ServiceAccount needs permission to create and delete Secrets in `amortized-jobs`:

```yaml
# In rbac.yaml
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: [""]
    resources: ["services"]
    verbs: ["create", "get", "list", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["secrets", "configmaps"]
    verbs: ["create", "get", "delete"]
```

**What this replaces:**

| Current (SSH) | New (K8s Secrets) |
|---|---|
| `forward_env` reads from host `os.environ` | Infrastructure secrets from K8s Secret, user keys from DB |
| `podman secret create` per job | `create_namespaced_secret` per job |
| `podman secret rm` on cleanup | ownerReferences auto-delete |
| API keys only in `api_keys` table | DB for storage, K8s Secrets for injection |

### Module 7: Backend Selection via Environment

The server already uses pydantic-settings with `AMORTIZED_` prefix. Add:

```python
# config.py additions
compute_backend: str = "local"           # "local" | "ssh" | "kubernetes"
compute_namespace: str = "amortized-jobs"
image_registry: str = "ghcr.io/amortized-ai"
```

Startup logic in `main.py._load_backends()`:

```python
if settings.compute_backend == "kubernetes":
    from amortized.backends.kubernetes import KubernetesBackend
    backend = KubernetesBackend(
        name="kubernetes",
        namespace=settings.compute_namespace,
        image_registry=settings.image_registry,
    )
    register_backend(backend)
elif settings.compute_backend == "ssh":
    # existing config.yaml loading
    ...
```

The `config.yaml` path still works for local/SSH development. On OpenShift, env vars from ConfigMap take precedence.

### Module 8: Worker Adaptations

The worker (`worker.py`) shifts from **code generation** to **config translation**. The control plane generates tool-native YAML configs, not Python scripts.

**What changes:**

| Current (SSH) | New (K8s) |
|---|---|
| `_trl_trainer_script()` — generates 150-line Python script | `_trl_config_yaml()` — generates TRL-native YAML config |
| `_eval_script()` — generates 80-line Python script | `_eval_config_yaml()` — generates asynth judge YAML config |
| `_build_synth_config()` — generates asynth config dict | Same — already config-only, just output as YAML |
| `_serve_config_yaml()` — generates vLLM YAML | Same — already config-only |
| Writes `run.py` + `config.json` to remote via SSH | Creates ConfigMap with `config.yaml`, mounted into Job pod |
| `_fetch_remote_outputs()` via SFTP | Reads from MLflow API after job completion |
| `cleanup_secrets()` deletes podman secrets | No-op — K8s Secrets persist |
| `_get_events_url()` resolves SSH connection IP | Static: `http://amortized-server.amortized.svc.cluster.local:8000` |
| Serve jobs: `podman run -p` | Creates Deployment + Service |

**Config translation example (training):**

```python
def _trl_config_yaml(config: dict[str, Any]) -> str:
    """Translate amortized training config → TRL-native YAML."""
    trl_config = {
        "model_name_or_path": config.get("model_name_or_path", config.get("model_path")),
        "dataset_name": config.get("data_path"),
        "output_dir": config.get("output_dir", "/amortized/work/output"),
        "report_to": "mlflow",
    }

    # Map amortized field names → TRL field names
    for key, value in config.items():
        if key in _SKIP_KEYS or value is None:
            continue
        trl_key = _TRL_FIELD_MAP.get(key, key)
        trl_config[trl_key] = value

    # LoRA → peft_config section
    if config.get("use_peft") or config.get("lora_r"):
        trl_config["peft_config"] = {
            "r": config.get("lora_r", 16),
            "lora_alpha": config.get("lora_alpha", 32),
            "lora_dropout": config.get("lora_dropout", 0.05),
            "target_modules": config.get("lora_target_modules", "all-linear"),
        }

    return yaml.dump(trl_config, default_flow_style=False, sort_keys=False)
```

**What this eliminates:**

- `_trl_trainer_script()` — 125 lines of generated Python → ~30 lines of config translation
- `_training_hub_script()` — 75 lines of generated Python → same pattern
- `_eval_script()` — 80 lines of generated Python → config for `asynth judge`
- No more string concatenation of Python code
- No more `import` statements embedded in strings
- No more runtime `exec()` or file-write-then-execute patterns

**What stays:**

- `_build_synth_config()` — already config-only
- `_serve_config_yaml()` — already config-only
- `_resolve_artifact_refs()` — still resolves `artifact:<id>` to paths (now MLflow URIs)
- `_resolve_judge_template()` — still merges judge templates into eval config

### Module 9: Artifact Storage (S3 via MLflow)

All artifacts — training data, model weights, SDG output, eval results, checkpoints — go to S3 via MLflow's artifact store. The server PVC only stores SQLite and logs.

**Why S3:**
- Model weights are 1-15GB. PVCs don't scale for shared access (EBS is RWO).
- On AWS (ROSA), S3 is native. On-prem, MinIO or Ceph RGW provide S3-compatible APIs.

**How it works:**

```
Job config delivery:
  Server → AMORTIZED_CONFIG_JSON env var → Job pod

Artifact flow (via MLflow):
  Job pod → mlflow.log_artifact() → S3 (MLflow manages structure)
  Server → mlflow.get_artifact_uri() → S3 pre-signed URL
  Studio → "Open in MLflow" → MLflow UI shows artifacts
  Studio → download link → S3 pre-signed URL via MLflow API
```

**S3 bucket structure (managed by MLflow):**

```
s3://amortized/
  mlflow/
    <experiment_id>/
      <run_id>/
        artifacts/
          model/              # trained model weights
          adapter/            # LoRA adapter weights
          generated_data/     # SDG output JSONL
          eval_results/       # eval output
          config.yaml         # job config (logged as artifact)
```

MLflow organizes by experiment → run → artifacts. Amortized maps: experiment = `amortized/{job_type}`, run = one job execution.

**Credentials:**

S3 credentials stored in a K8s Secret, injected into both server and job pods:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: amortized-s3
  namespace: amortized
stringData:
  AWS_ACCESS_KEY_ID: "minioadmin"
  AWS_SECRET_ACCESS_KEY: "minioadmin"
  AWS_S3_BUCKET: "amortized"
  AWS_S3_ENDPOINT: "http://minio.amortized.svc.cluster.local:9000"
  AWS_S3_ENDPOINT_URL: "http://minio.amortized.svc.cluster.local:9000"  # MLflow uses this form
  AWS_DEFAULT_REGION: "us-east-2"
```

MinIO runs on-cluster — no AWS access needed. To switch to real S3 later, just update the endpoint and credentials in this Secret.

The same Secret is referenced in job pod specs via `envFrom.secretRef`.

**Config settings:**

```python
# config.py
storage_backend: str = "local"     # "local" | "s3"
storage_bucket: str = ""           # S3 bucket name
storage_prefix: str = "jobs/"      # prefix in bucket
storage_region: str = "us-east-2"
storage_endpoint: str = ""         # for MinIO
```

When `storage_backend=s3`: artifacts uploaded via `boto3`, downloads via pre-signed URLs.
When `storage_backend=local`: current behavior (files on local disk). For local dev.

**Worker changes:**

| Current (SSH) | Change (K8s + MLflow) |
|---|---|
| Writes run.py to remote via SSH | Injects config as env var |
| `_fetch_remote_outputs()` via SFTP | Reads from MLflow API after job completion |
| Artifact paths are local filesystem | Artifact URIs are `runs:/<run_id>/model` |
| `register_artifacts_for_job()` scans local files | MLflow already has the artifacts — just record the run_id |
| Job gets work dir from SSH remote path | Job writes to local EmptyDir, logs to MLflow |

**Container-side artifact logging:**

1. **Training containers** — TRL's MLflowCallback handles everything when `report_to: mlflow` and `HF_MLFLOW_LOG_ARTIFACTS=true` are set. Model weights, metrics, and checkpoints all go to MLflow automatically.

2. **SDG/eval containers** — the generated runner script includes MLflow logging:
   ```python
   import mlflow
   with mlflow.start_run(run_name=f"sdg-{job_id[:8]}"):
       mlflow.log_params({"model": config.model, "num_samples": config.num_samples})
       mlflow.log_artifact(output_path, "generated_data")
       dataset = mlflow.data.from_pandas(df, source=output_path)
       mlflow.log_input(dataset, context="training_data")  # lineage
   ```
   The `MLFLOW_TRACKING_URI` and S3 credentials are in the pod's environment via Secrets.

**Dataset loading from S3 (training input):**

Training containers need the SDG output as input data. The HuggingFace `datasets` library reads S3 paths natively via `fsspec`:

```yaml
# TRL config.yaml — dataset loaded directly from S3/MinIO, no download step
dataset_name: s3://amortized/mlflow/<exp_id>/<run_id>/artifacts/generated_data/generated_data.jsonl
```

The `datasets` library picks up S3 credentials from env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT_URL`) which are already injected via the `amortized-s3` Secret. No init container or pre-download needed.

**Required pip packages in training container:** `s3fs`, `fsspec` (for datasets S3 access) + `mlflow`, `boto3` (for artifact logging).

### Module 10: Container Images CI

GitHub Actions workflow triggered on release or manual dispatch:

```yaml
images:
  - name: amortized
    context: .
    dockerfile: Dockerfile
  - name: studio
    context: studio/
    dockerfile: studio/Dockerfile
```

Build and push to `ghcr.io/amortized-ai/`. Training and asynth images already exist.

### Implementation Notes

**Server restart resilience:**

If the amortized-server pod restarts while jobs are running, the current `cleanup_orphaned_jobs()` marks all in-flight jobs as failed. On K8s this is wrong — the Jobs/Deployments continue running independently. On startup, the worker must reconcile with K8s state:

```python
async def reconcile_k8s_jobs(db, backend):
    """On startup, sync amortized DB with actual K8s Job/Deployment state."""
    running_jobs = await db.get_jobs_by_status("running")
    for job in running_jobs:
        handle = deserialize_handle(job.backend_handle)
        actual_status = await backend.status(handle)
        if actual_status.state == "succeeded":
            await db.update_job_status(job.id, "succeeded")
        elif actual_status.state == "failed":
            await db.update_job_status(job.id, "failed")
        # else: still running — leave it, resume polling
```

**Shared memory (`/dev/shm`):**

Training and serve containers both mount a Memory-backed EmptyDir at `/dev/shm` (12Gi). Without this, PyTorch DataLoader workers crash with "bus error" when `num_workers > 0`. Already included in the Job and Deployment pod specs above.

**MinIO capacity:**

The MinIO PVC is 100Gi. Model weights are 5-15GB each, SDG datasets are ~10-100MB. At ~15 training runs the PVC fills up. Mitigations:

- Start with 200Gi PVC (gp3-csi supports online expansion via `oc edit pvc`)
- Add a cleanup job or CLI command: `amortized cleanup --older-than 30d`
- Monitor via `oc exec deploy/minio -- mc du local/amortized`
- gp3-csi supports volume expansion — `oc patch pvc minio-data -p '{"spec":{"resources":{"requests":{"storage":"500Gi"}}}}'`

## Deployment Steps

```bash
# 1. Build and push images (CI does this, or manually)
docker build -t ghcr.io/amortized-ai/amortized:latest .
docker build -t ghcr.io/amortized-ai/studio:latest studio/
docker push ghcr.io/amortized-ai/amortized:latest
docker push ghcr.io/amortized-ai/studio:latest

# 2. Create namespaces
oc new-project amortized
oc new-project amortized-jobs

# 3. Deploy (includes MinIO, MLflow, server, studio, S3 secrets)
oc apply -f k8s/

# 4. Copy S3 secret to jobs namespace (job pods need MinIO access too)
oc get secret amortized-s3 -n amortized -o json \
  | jq '.metadata.namespace = "amortized-jobs" | del(.metadata.resourceVersion,.metadata.uid,.metadata.creationTimestamp)' \
  | oc apply -f -

# 5. Create the MinIO bucket (one-time, after MinIO pod is running)
oc wait --for=condition=available deploy/minio -n amortized --timeout=60s
oc exec deploy/minio -n amortized -- \
  mc alias set local http://localhost:9000 minioadmin minioadmin && \
  mc mb local/amortized

# 6. Add LLM API keys via Studio UI (or CLI)
#    Data scientists manage their own keys through Studio → Settings → API Keys
#    Keys are stored encrypted in amortized DB, injected as per-job K8s Secrets at dispatch

# 6. Get Studio URL
oc get route amortized-studio -n amortized -o jsonpath='{.spec.host}'
```

## Testing

### Test 1: Server + Studio

```bash
oc apply -f k8s/
oc get pods -n amortized           # both Running
curl https://<studio-route>/       # Studio loads
curl https://<studio-route>/api/v1/health  # {"status": "ok"}
```

### Test 2: SDG job (CPU, no GPU)

```bash
# Via CLI pointed at the cluster
amortized --url https://<route> submit examples/ticket-classifier/synth.yaml -x
# Verify: K8s Job created in amortized-jobs
# Verify: Pod runs on any node (no GPU needed)
# Verify: Output artifact registered
```

### Test 3: Training job (GPU)

```bash
amortized --url https://<route> submit examples/ticket-classifier/train.yaml \
  --set config.data_path=<artifact-ref> -x
# Verify: K8s Job with nvidia.com/gpu=1 request
# Verify: Pod scheduled on L40S node
# Verify: Training metrics streamed via events
# Verify: Model artifact registered
```

### Test 4: Serve job (GPU, long-running)

```bash
amortized --url https://<route> submit recipes/serve/vllm.yaml \
  --set config.model=Qwen/Qwen3-0.6B -x
# Verify: Deployment + Service created in amortized-jobs
# Verify: vLLM pod running, model loaded
# Verify: Model accessible via Service endpoint
```

### Test 5: Full pipeline via Studio

```
1. Open Studio → chat
2. "Build me a ticket classifier"
3. Agent proposes SDG → Confirm → K8s Job runs → data generated
4. Agent proposes training → Confirm → K8s Job on GPU → model trained
5. Agent proposes deploy → Confirm → Deployment + Service created
```

## Out of Scope

| Item | When |
|---|---|
| OpenShift AI integration (Kubeflow Model Registry, KServe, Training Operator, Kueue) | After this PRD ships and works |
| Helm chart | After raw manifests prove out |
| OpenShift OAuth proxy sidecar | After Phase 1 (bearer token auth works for now) |
| Multi-replica server | After PostgreSQL migration (if ever needed) |
| AWS S3 migration | When outgrowing MinIO or needing cross-region access |
| Multi-node distributed training | After single-GPU LoRA works |
| Custom container images / registry auth | After ghcr.io works |

## Code Changes Summary

| File | Change | Effort |
|---|---|---|
| `backends/kubernetes.py` (new) | KubernetesBackend — ConfigMap + Job + Deployment + Service creation | Medium |
| `worker.py` | Replace `_trl_trainer_script()` and `_eval_script()` with config translation functions (`_trl_config_yaml()`, `_eval_config_yaml()`). Remove SFTP, SSH heredoc, podman secret code paths. Add K8s backend dispatch. | Medium |
| `main.py` | Add `kubernetes` backend type in `_load_backends()` | Small |
| `config.py` | Add `compute_backend`, `compute_namespace`, `image_registry`, `mlflow_tracking_uri` settings | Small |
| `core/artifacts.py` | Replace local file scanning with MLflow artifact queries. Keep `artifacts` table as cache but source of truth is MLflow. | Medium |
| `api/artifacts.py` | Download via MLflow `get_artifact_uri()` → S3 pre-signed URL | Small |
| `Dockerfile` (new) | Server production image | Small |
| `studio/Dockerfile` (new) | Studio production image | Small |
| `studio/nginx.conf.template` (new) | SPA + reverse proxy | Small |
| `k8s/` (new directory) | All K8s manifests (including MLflow deployment) | Small |
| `.github/workflows/images.yml` (new) | Container image CI | Small |

**New dependencies:** `kubernetes_asyncio`, `mlflow`

**Total effort:** ~2-3 weeks. KubernetesBackend + MLflow integration are the bulk. Everything else is configuration.

## Decision Log

| Decision | Choice | Rationale |
|---|---|---|
| Job type for training/SDG/eval | `batch/v1 Job` | Simplest K8s primitive. No dependency on Training Operator. |
| Job type for serve | `apps/v1 Deployment` + `v1 Service` | Long-running, needs restart policy and stable endpoint. |
| Artifact management | MLflow (deployed with amortized) | Provides artifact storage, experiment tracking, model registry, and lineage in one system. S3-backed. |
| Artifact storage | MinIO (S3-compatible, on-cluster) via MLflow artifact store | Model weights are 1-15GB. S3 API is shared, scalable. MinIO runs on-cluster — zero AWS dependency. Swap to real S3 by changing one Secret. |
| Config delivery to jobs | `AMORTIZED_CONFIG_JSON` env var | Already supported by container runners. |
| MLflow backend store | SQLite on PVC | Sufficient for single-team use. PostgreSQL upgrade path exists. |
| Auth | Bearer token (existing) | OpenShift OAuth is Phase 2. Existing auth works for initial deployment. |
| Database (amortized) | SQLite on PVC | Job queue + chat conversations only. Model/artifact metadata lives in MLflow. |
| Server replicas | 1 | SQLite is single-writer. Scale later if needed. |
| Image registry | ghcr.io (public) | Avoids ImagePullSecret complexity. Mirror to internal registry later. |
| Storage class | Default (gp3-csi on ROSA) | EBS is RWO, sufficient for single-replica PVCs. |
| GPU scheduling | `nodeSelector` + resource requests | Native K8s. Kueue is Phase 2 (RHOAI integration). |
| RHOAI Model Registry | Not used (Phase 2) | On plain OpenShift, MLflow Model Registry fills this role. On RHOAI, can bridge to Kubeflow Hub later. |
