# Amortized on OpenShift AI — Integration Plan

> **Updated 2026-06-19** — Revised after deep research (105 agents, adversarially verified). Key correction: RHOAI Model Registry is Kubeflow Hub (v0.3.9), not MLflow Model Registry. MLflow is used only for experiment tracking. Model versioning and registration use the `model_registry` Python client against the Kubeflow Hub REST API (`v1alpha3`).

Amortized is a thin orchestration layer on top of OpenShift AI. It adds what OpenShift AI lacks (SDG, agent chat, task-model recipes, LLM-as-judge eval) and delegates infrastructure to what OpenShift AI already provides (Model Registry, MLflow, KServe, Kueue, S3, OAuth, Prometheus).

## Architecture

```
┌─────────────────────────────────────────────┐
│  Amortized (thin layer)                     │
│  ┌──────────┐ ┌──────┐ ┌───────┐ ┌───────┐ │
│  │  Studio   │ │Agent │ │asynth │ │Judges │ │
│  │  (React)  │ │(chat)│ │(SDG)  │ │(eval) │ │
│  └────┬──────┘ └──┬───┘ └───┬───┘ └───┬───┘ │
│       │           │         │         │     │
│  ┌────▼───────────▼─────────▼─────────▼───┐ │
│  │  Amortized API (FastAPI)               │ │
│  │  - Job orchestration (thin)            │ │
│  │  - Recipe resolution                   │ │
│  │  - Agent tools                         │ │
│  └────────────────┬───────────────────────┘ │
└───────────────────┼─────────────────────────┘
                    │ delegates to
┌───────────────────▼─────────────────────────┐
│  OpenShift AI (infrastructure)              │
│  ┌──────────┐ ┌──────────┐ ┌─────┐ ┌─────┐ │
│  │  Model   │ │  MLflow   │ │Kueue│ │ S3  │ │
│  │ Registry │ │ (expt.   │ │ GPU │ │stor.│ │
│  │(Kubeflow)│ │ tracking)│ │sched│ │     │ │
│  └──────────┘ └──────────┘ └─────┘ └─────┘ │
│  ┌──────────┐ ┌───────┐ ┌───────────────┐  │
│  │OpenShift │ │KServe │ │  Prometheus   │  │
│  │  OAuth   │ │serving│ │  + Grafana    │  │
│  └──────────┘ └───────┘ └───────────────┘  │
└─────────────────────────────────────────────┘
```

## What Amortized Owns vs Delegates

### Amortized owns (unique value — OpenShift AI can't do this)

| Component | What it does |
|---|---|
| **asynth** | Synthetic data generation — attribute-based, multi-turn, tool-use |
| **Agent chat** | Guided workflow via AI agent with option cards, action confirmation |
| **Studio UI** | Task-model-focused dashboard (not a generic ML dashboard) |
| **Recipes** | Pre-built end-to-end configs (ticket-classifier, entity-extractor, etc.) |
| **LLM-as-judge** | Evaluate task model quality with custom criteria |
| **Task-model workflow** | Generate → train → eval → deploy as one orchestrated flow |

### OpenShift AI provides (don't rebuild)

| Component | OpenShift AI service | What amortized does |
|---|---|---|
| **Experiment tracking** | MLflow (optional) | Set `report_to: mlflow`, log params/metrics/artifacts, loss curves |
| **Artifact storage** | MLflow artifact store → S3 | Call `mlflow.log_artifact()`, never manage S3 directly |
| **Model registry** | Model Registry (Kubeflow Hub) | Register models via `model_registry` Python client against `v1alpha3` REST API |
| **Model serving** | KServe (RawDeployment) | Create `InferenceService` CR, KServe handles vLLM + scaling. RHOAI 3.4 adds `LLMInferenceService` CRDs. |
| **GPU scheduling** | Kueue | Submit training as Kueue `Workload`, Kueue handles queuing + quotas |
| **S3 storage** | Data Science Project secrets | Auto-discover bucket config from DSP |
| **Auth** | OpenShift OAuth | OAuth proxy sidecar on Studio |
| **Monitoring** | Prometheus + Grafana | Expose `/metrics` endpoint, cluster Prometheus scrapes |
| **Database** | Model Registry has its own MySQL (managed by RHOAI operator) | Amortized keeps lightweight SQLite for job queue + conversations only |
| **GPU quotas** | Kueue ClusterQueue + ResourceFlavor | Configure via Kueue CRs, not amortized code |

## What This Eliminates from Our Backlog

| Originally planned | Status | Reason |
|---|---|---|
| PostgreSQL migration | **Eliminated** | Model Registry uses its own MySQL (managed by RHOAI operator). Amortized only needs SQLite for job queue and chat conversations. |
| Custom artifact storage system | **Eliminated** | Artifacts go to MLflow. `mlflow.log_artifact()` stores them in the DSP's S3. |
| S3/GCS storage backend code | **Simplified** | S3Storage class exists but is only needed for non-OpenShift deployments. On OpenShift AI, MLflow handles storage. |
| Helm chart PostgreSQL subchart | **Eliminated** | No database to manage — Model Registry uses its own MySQL via RHOAI operator. |
| K8s Job dispatch code | **Simplified** | Submit as Kueue Workload or Kubeflow Pipeline run. Don't manage K8s Jobs directly. |
| Model serving containers | **Eliminated** | Create KServe InferenceService CR. KServe handles vLLM, scaling, routing, GPU allocation. |
| Custom model registry | **Eliminated** | Model Registry (Kubeflow Hub) with versioning, metadata, lineage labels. |
| Auth system / RBAC | **Eliminated** | OpenShift OAuth handles identity. OpenShift RBAC handles permissions per namespace. |
| Monitoring / Prometheus metrics | **Simplified** | Expose `/metrics`, cluster Prometheus scrapes automatically. No custom monitoring infrastructure. |
| GPU quota enforcement | **Eliminated** | Kueue ClusterQueue with ResourceFlavor handles per-team GPU quotas. |

## How Amortized Interacts with Each Service

### MLflow — Experiment Tracking + Artifacts (NOT Model Registration)

MLflow provides: loss curves, hyperparameter logging, run comparison, artifact store. It does **not** provide model versioning on RHOAI — that's Model Registry (Kubeflow Hub).

```python
# Training job — inject env vars into container
env = {
    "MLFLOW_TRACKING_URI": discover_mlflow_uri(),  # from OpenShift AI DSP
    "MLFLOW_EXPERIMENT_NAME": f"amortized/{project}/{job_type}",
    "HF_MLFLOW_LOG_ARTIFACTS": "true",
}
# TRL's built-in MLflowCallback handles the rest:
# - Logs loss, learning_rate, epoch per step
# - Copies checkpoints to MLflow artifact store (S3)

# SDG job — log explicitly from sdg_runner.py
import mlflow
with mlflow.start_run(run_name=f"sdg-{job_id[:8]}"):
    mlflow.log_params({"model": config.model, "num_samples": config.num_samples})
    mlflow.log_artifact(output_path, "generated_data")
    dataset = mlflow.data.from_pandas(df, source=output_path)
    mlflow.log_input(dataset, context="training_data")  # lineage link
```

### Model Registry (Kubeflow Hub) — Model Versioning + Registration

The RHOAI Model Registry (`default-modelregistry`) is Kubeflow-based (renamed Kubeflow Hub in v0.3.9). It exposes a REST API at `v1alpha3` under `/api/model_registry/v1alpha3/`. Uses MySQL backend (managed by RHOAI operator). Stores model metadata only, not artifacts.

```python
# After training — register model via Model Registry (NOT mlflow.register_model)
from model_registry import ModelRegistry

registry = ModelRegistry(
    server_address="https://default-modelregistry.redhat-ods-applications.svc:8443",
    author="amortized",
)
registered_model = registry.register_model(
    name=f"{project}/{model_name}",
    uri=f"s3://{bucket}/{model_path}",
    model_format_name="vllm",
)
```

### KServe — Model Serving (RawDeployment)

Target RawDeployment mode — Serverless mode (Knative+Istio) is deprecated in RHOAI 3.3. RHOAI 3.4 introduces `LLMInferenceService` CRDs for LLM workloads.

```python
# Deploy a trained model via KServe InferenceService
from kubernetes import client

isvc = {
    "apiVersion": "serving.kserve.io/v1beta1",
    "kind": "InferenceService",
    "metadata": {
        "name": model_name,
        "namespace": compute_namespace,
        "annotations": {
            "serving.knative.dev/progress-deadline": "30m",
        },
        "labels": {
            "modelregistry/registered-model-id": model_id,
            "modelregistry/model-version-id": version_id,
        },
    },
    "spec": {
        "predictor": {
            "model": {
                "modelFormat": {"name": "vllm"},
                "storageUri": f"s3://{bucket}/{model_path}",
                "resources": {
                    "requests": {"nvidia.com/gpu": "1"},
                    "limits": {"nvidia.com/gpu": "1"},
                },
            }
        }
    }
}
custom_api.create_namespaced_custom_object(
    group="serving.kserve.io", version="v1beta1",
    namespace=compute_namespace, plural="inferenceservices",
    body=isvc,
)
```

### Kueue — GPU Scheduling

```python
# Submit training as a Kueue-managed K8s Job
job = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {
        "name": f"amortized-train-{job_id[:8]}",
        "namespace": compute_namespace,
        "labels": {"kueue.x-k8s.io/queue-name": "amortized-queue"},
    },
    "spec": {
        "template": {
            "spec": {
                "containers": [{
                    "name": "training",
                    "image": training_image,
                    "env": env_vars,  # includes MLFLOW_TRACKING_URI
                    "resources": {
                        "requests": {"nvidia.com/gpu": str(gpu_count)},
                        "limits": {"nvidia.com/gpu": str(gpu_count)},
                    },
                }],
                "restartPolicy": "Never",
                "nodeSelector": {"nvidia.com/gpu": "true"},
            }
        }
    }
}
batch_api.create_namespaced_job(namespace=compute_namespace, body=job)
# Kueue intercepts and queues based on ClusterQueue quota
```

### S3 Auto-Discovery from Data Science Project

```python
# Read S3 config from OpenShift AI DSP secrets
def discover_s3_config(namespace: str) -> dict:
    """Read S3 bucket config from the Data Science Project's connection secrets."""
    v1 = client.CoreV1Api()
    secrets = v1.list_namespaced_secret(namespace, label_selector="opendatahub.io/managed=true")
    for secret in secrets.items:
        data = {k: base64.b64decode(v).decode() for k, v in secret.data.items()}
        if "AWS_S3_BUCKET" in data:
            return {
                "bucket": data["AWS_S3_BUCKET"],
                "endpoint": data.get("AWS_S3_ENDPOINT", ""),
                "region": data.get("AWS_DEFAULT_REGION", "us-east-1"),
                "access_key": data.get("AWS_ACCESS_KEY_ID", ""),
                "secret_key": data.get("AWS_SECRET_ACCESS_KEY", ""),
            }
    return {}
```

### MLflow Auto-Discovery

```python
def discover_mlflow_uri(namespace: str) -> str:
    """Find MLflow tracking server URI from OpenShift AI."""
    v1 = client.CoreV1Api()
    # MLflow is typically deployed as a Route in the RHOAI namespace
    services = v1.list_namespaced_service(
        "redhat-ods-applications",
        label_selector="app=mlflow"
    )
    if services.items:
        svc = services.items[0]
        return f"http://{svc.metadata.name}.{svc.metadata.namespace}.svc:5000"
    # Fallback: check env var
    return os.environ.get("MLFLOW_TRACKING_URI", "")
```

## Helm Chart (Simplified)

Much simpler without PostgreSQL, custom storage, or auth. Model Registry has its own MySQL managed by the RHOAI operator — no database subchart needed.

```yaml
# values.yaml
server:
  image:
    registry: ghcr.io/amortized-ai
    repository: amortized
    tag: latest
  replicas: 1

studio:
  image:
    registry: ghcr.io/amortized-ai
    repository: studio
    tag: latest

# OpenShift AI integration (auto-detected when possible)
openshiftAI:
  autoDetect: true          # Look for RHOAI operator, auto-configure below
  modelRegistry:
    serverAddress: ""       # Auto-discovered from default-modelregistry service
  mlflow:
    trackingUri: ""          # Auto-discovered if empty (optional, experiment tracking only)
  storage:
    autoDiscover: true       # Read S3 config from DSP secrets
  compute:
    namespace: amortized-jobs
    createNamespace: true
    queueName: amortized-queue  # Kueue local queue name

# Container images for jobs
jobImages:
  training: ghcr.io/amortized-ai/training:latest
  sdg: ghcr.io/amortized-ai/asynth:latest
  eval: ghcr.io/amortized-ai/asynth:latest

# Auth
auth:
  openshiftOAuth: true      # Add oauth-proxy sidecar

# Agent
agent:
  model: gpt-5.4
  apiKeySecret: amortized-agent-key  # K8s secret with OPENAI_API_KEY
```

```
charts/amortized/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment-server.yaml
│   ├── deployment-studio.yaml
│   ├── service-server.yaml
│   ├── service-studio.yaml
│   ├── route-studio.yaml          # OpenShift Route
│   ├── route-api.yaml
│   ├── configmap.yaml             # Model Registry, MLflow, S3 config, compute namespace
│   ├── secret.yaml                # Agent API key
│   ├── serviceaccount.yaml
│   ├── role.yaml                  # Create Jobs + InferenceServices in compute namespace
│   ├── rolebinding.yaml
│   ├── kueue-localqueue.yaml      # Kueue LocalQueue for amortized jobs
│   └── NOTES.txt
```

No PostgreSQL subchart. No database to manage. No storage provisioning. No auth backend.

## Execution Plan (6 items to production)

### Phase 1: Connect to OpenShift AI

| # | Work | Effort | What it enables |
|---|---|---|---|
| 1a | **MLflow integration** — `report_to: mlflow`, SDG logging, lineage via `log_input()` | Medium | Experiment tracking + artifact storage via existing MLflow |
| 1b | **Model Registry integration** — register models via `model_registry` Python client, lineage labels on InferenceService | Medium | Model versioning + registration via Kubeflow Hub (not MLflow) |
| 2 | **Kueue job submission** — submit training/SDG/eval as K8s Jobs with Kueue queue label | Medium | GPU scheduling with quotas and fair-sharing via existing Kueue |
| 3 | **KServe model deployment** — create InferenceService CR instead of raw vLLM pods | Small | Model serving with auto-scaling via existing KServe |
| 4 | **S3 + MLflow auto-discovery** — read config from DSP secrets and RHOAI services | Small | Zero-config storage and tracking |
| 5 | **Production Studio build** — `vite build` → nginx, Dockerfile, configurable API URL | Small | Deployable Studio container |
| 6 | **Helm chart** — deploy server + studio + configure integrations | Small | One-command install on OpenShift |

### Phase 2: Polish

| # | Work | Effort |
|---|---|---|
| 7 | OpenShift OAuth proxy sidecar for Studio | Medium |
| 8 | "Open in MLflow" links in Studio | Small |
| 9 | Failure classification (OOM, NCCL, config errors) | Small |
| 10 | Recipes and examples upgrade | Small |
| 11 | Agent `present_options` for structured guided workflows | Done (#176) |

### Phase 3: Scale

| # | Work | When |
|---|---|---|
| 12 | Project scoping (maps to Model Registry namespaces + MLflow experiments + K8s namespaces) | When multi-team |
| 13 | Retry with checkpoint resume | When on spot instances |
| 14 | Pipeline definitions (synth→train→eval as declared flow) | When workflow complexity grows |

## Model Storage Options

Three verified storage backends for model artifacts on RHOAI:

| Backend | URI scheme | Notes |
|---|---|---|
| **S3** | `s3://bucket/path` | Default. Uses DSP-configured S3 (MinIO or cloud). |
| **PVC** | `pvc://claim-name/path` | Persistent Volume Claim. Good for air-gapped clusters. |
| **OCI ModelCar** | `oci://registry/repo:tag` | Packages models as OCI images. Introduced in RHOAI 2.14. Enables versioned model distribution via container registries. |

## Local Development (Without OpenShift AI)

Amortized still works without OpenShift AI for local development:

```bash
# Local dev — no OpenShift, no MLflow, no K8s
amortized up                          # SQLite + local filesystem
amortized config                      # Configure SSH backend to GPU node
amortized submit examples/ticket-classifier/synth.yaml --confirm
```

When `MLFLOW_TRACKING_URI` is empty → `report_to: none` (no MLflow).
When no Kueue → fall back to direct K8s Job or SSH backend.
When no KServe → fall back to raw vLLM container dispatch.

The OpenShift AI integration is additive, not required.

## Platform Engineer Install (Target Experience)

```bash
# Sarah logs into her ROSA cluster
oc login --token=... --server=https://api.cluster.openshiftapps.com:443

# Install amortized via Helm
helm repo add amortized https://charts.amortized.ai
helm install amortized amortized/amortized \
  --namespace amortized --create-namespace \
  --set openshiftAI.autoDetect=true \
  --set agent.apiKeySecret=openai-key

# That's it. MLflow, S3, Kueue, KServe, OAuth are auto-discovered.
# Studio: https://amortized-studio-amortized.apps.cluster.openshiftapps.com
```

## Data Scientist Experience

```
1. Open Studio → SSO login (OpenShift OAuth)
2. Chat: "I want to build a ticket classifier"
3. Agent guides through options → generates data (asynth)
   → SDG runs as Kueue Job on GPU node
   → Params + artifacts logged to MLflow
4. Agent: "Data ready. Train?" → training runs
   → Loss curves in MLflow
   → Checkpoint saved to S3 via MLflow artifact store
5. Agent: "94% accuracy. Deploy?" → creates KServe InferenceService
   → Model registered in Model Registry (Kubeflow Hub)
   → Served via KServe (RawDeployment) with lineage labels
6. Platform engineer sees:
   → MLflow UI: experiments, runs, loss curves
   → Model Registry UI: registered models, versions, lineage
   → Grafana: GPU utilization, job queue depth
   → OpenShift console: pods, services, routes
```
