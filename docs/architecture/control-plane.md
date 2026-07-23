# Control Plane for Task Model Creation

A control plane that orchestrates the end-to-end workflow for building task models — small, fine-tuned LLMs that replace expensive frontier model API calls for specific tasks (classification, extraction, routing, summarization).

Deployed on OpenShift AI. Uses RHOAI-native services where they exist, builds only what's missing.

---

## Components

### 1. Artifacts — MLflow

**Status**: Agreed

**What it covers**: artifact storage, dataset tracking, model registry, experiment tracking, lineage

**Decision**: MLflow is the single source of truth for all artifacts and metadata. No custom artifact tables in amortized.

#### Why MLflow (not Kubeflow Model Registry, not custom)

We evaluated three approaches:

| Approach | Verdict | Reason |
|---|---|---|
| **Kubeflow Model Registry** | Not sufficient | Tracks models only (RegisteredModel → ModelVersion → ModelArtifact). No dataset tracking. Stores metadata pointers only — doesn't manage actual artifact storage. Newer, alpha-stage. |
| **granite.build's ArtifactRegistration** | Good design, but custom | Unified registry for models + datasets with `origin_uris` lineage. URI-based asset stores. But it's a custom implementation — we'd be building our own. |
| **MLflow** | **Use this** | Covers all artifact types — experiments, datasets, models, eval results. Stores actual artifacts (not just pointers). Built-in model registry. Dataset lineage via `mlflow.data`. SQLite default (3.7+), PostgreSQL for production. Already deployed on RHOAI. |

MLflow provides everything granite.build built custom (artifact registry, lineage, URI-based storage) plus everything Kubeflow Model Registry provides (model versioning, aliases) — in a single service that's already part of RHOAI.

#### MLflow Capabilities We Use

| Capability | MLflow Feature | How We Use It |
|---|---|---|
| **Experiment tracking** | Experiments + Runs | Every SDG/training/eval job creates an MLflow run with params, metrics, tags |
| **Dataset tracking** | `mlflow.data` + `mlflow.log_input()` | SDG outputs logged as datasets with source lineage, digest, schema |
| **Model artifacts** | `mlflow.log_artifact()` | Training outputs (LoRA adapters, checkpoints) stored in MLflow's artifact store (S3) |
| **Model registry** | `RegisteredModel` → `ModelVersion` | Trained models registered with name, version, alias (`@champion`), linked to training run |
| **Lineage** | Run → inputs (datasets) → outputs (models) | Full chain: SDG dataset → training run → registered model version |
| **Storage backend** | SQLite (standalone) / PostgreSQL (production) | MLflow 3.7+ defaults to SQLite (`sqlite:///mlflow.db`). Production on OpenShift uses PostgreSQL. |
| **Artifact store** | S3 (MinIO on-cluster or external) | Physical files stored in S3, MLflow tracks metadata + URI pointers |
| **Evaluation results** | Runs + metrics + `EvaluationDataset` | Eval runs log accuracy, latency, quality scores alongside the eval dataset |
| **Tracing** | OpenTelemetry-based traces (MLflow 3.0+) | Optional: trace LLM call chains during SDG for debugging |

#### MLflow Data Model

```
Experiment (e.g. "amortized/sdg/ticket-classifier")
  └── Run (one per job execution)
        ├── params: {model: "gpt-4o-mini", num_samples: 200, temperature: 0.7}
        ├── metrics: {num_samples_generated: 200, duration_seconds: 52}
        ├── tags: {job_id: "abc123", job_type: "sdg", recipe: "ticket-classifier/synth"}
        ├── artifacts:
        │     └── generated_data.jsonl (stored in S3)
        └── inputs:
              └── Dataset(name="sdg-abc123", source="s3://...", digest="sha256:...")

Experiment (e.g. "amortized/training/ticket-classifier")
  └── Run (one per training job)
        ├── params: {model_name_or_path: "Qwen/Qwen3-0.6B", lora_r: 16, epochs: 3}
        ├── metrics: {loss: 0.42, eval_loss: 0.51, learning_rate: 0.0002}
        ├── tags: {job_id: "def456", job_type: "training"}
        ├── artifacts:
        │     ├── adapter_model.safetensors (stored in S3)
        │     └── adapter_config.json
        └── inputs:
              └── Dataset(name="sdg-abc123", source="s3://...")  ← lineage to SDG output

RegisteredModel (e.g. "ticket-classifier")
  └── ModelVersion (v1)
        ├── source: "runs:/def456/artifacts/model"
        ├── aliases: ["@champion"]
        ├── tags: {task: "classification", base_model: "Qwen/Qwen3-0.6B"}
        └── linked to: Run def456 → Experiment → Dataset inputs
```

#### How Each Job Type Logs to MLflow

**SDG Jobs** (asynth / SDG Hub):
```python
# Already implemented in asynth's synthesis_pipeline.py
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "amortized/sdg"))

with mlflow.start_run(run_name=f"sdg-{job_id[:8]}"):
    mlflow.log_params({"model": "gpt-4o-mini", "num_samples": 200})
    mlflow.log_metrics({"num_samples_generated": 200})
    mlflow.log_artifact(output_path, "generated_data")       # actual file → S3
    dataset = mlflow.data.from_pandas(df, source=s3_path, name=f"sdg-{job_id[:8]}")
    mlflow.log_input(dataset, context="training_data")        # dataset lineage
```

**Training Jobs** (Training Hub / thub CLI):
```yaml
# Training Hub auto-logs to MLflow via instructlab-training MLflow callback
# Env vars set by amortized:
#   MLFLOW_TRACKING_URI=http://mlflow:5000
#   MLFLOW_EXPERIMENT_NAME=amortized/training/{job_id[:8]}
#   HF_MLFLOW_LOG_ARTIFACTS=true  ← uploads model weights to MLflow artifact store
```
Training Hub's instructlab-training backend automatically logs: loss per step, learning_rate, epoch, and eval metrics to MLflow. Model artifacts are uploaded by the thub CLI after training completes.

**Eval Jobs**:
```python
with mlflow.start_run(run_name=f"eval-{job_id[:8]}"):
    mlflow.log_params({"model": "ticket-classifier-v1", "benchmark": "accuracy"})
    mlflow.log_metrics({"accuracy": 0.94, "f1": 0.91, "latency_p99_ms": 12})
    mlflow.log_artifact(results_path, "eval_results")
    mlflow.log_input(eval_dataset, context="evaluation")
```

**Model Registration** (after training succeeds):
```python
# Register the trained model in MLflow Model Registry
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/artifacts/model",
    name="ticket-classifier"
)
# Set alias for deployment
client = mlflow.MlflowClient()
client.set_registered_model_alias("ticket-classifier", "champion", result.version)
```

#### Artifact Flow: End-to-End Lineage

```
User: "I want a ticket classifier"
  │
  ▼
SDG Job
  ├── MLflow Experiment: amortized/sdg/ticket-classifier
  ├── MLflow Run: sdg-abc123
  ├── Params: {model: gpt-4o-mini, num_samples: 200}
  ├── Artifact: s3://bucket/mlflow/exp1/run-abc/artifacts/generated_data.jsonl
  └── Dataset logged: name=sdg-abc123, source=s3://..., digest=sha256:...
         │
         ▼ (dataset consumed as input)
Training Job
  ├── MLflow Experiment: amortized/training/ticket-classifier
  ├── MLflow Run: train-def456
  ├── Input: Dataset(sdg-abc123)  ← lineage link
  ├── Params: {model: Qwen/Qwen3-0.6B, lora_r: 16, epochs: 3}
  ├── Metrics: {final_loss: 0.42}
  └── Artifact: s3://bucket/mlflow/exp2/run-def/artifacts/model/
         │
         ▼ (model registered)
Model Registry
  ├── RegisteredModel: "ticket-classifier"
  ├── Version: 1
  ├── Source: runs:/def456/artifacts/model
  ├── Alias: @champion
  └── Linked to: Run def456 → Dataset sdg-abc123 → SDG Run abc123
         │
         ▼ (deployed to inference)
KServe InferenceService
  └── storageUri: from MLflow Model Registry
```

#### Environment Variables for Job Containers

Every job container dispatched by amortized receives:

| Env Var | Value | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow.{namespace}.svc:5000` | MLflow server address |
| `MLFLOW_EXPERIMENT_NAME` | `amortized/{job_type}/{job_id[:8]}` | Experiment grouping |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio.{namespace}.svc:9000` | S3 endpoint for MinIO (MLflow reads this, not `AWS_S3_ENDPOINT_URL`) |
| `HF_MLFLOW_LOG_ARTIFACTS` | `true` (training only) | Tell training backend to upload model weights to MLflow |
| `AWS_ACCESS_KEY_ID` | from K8s Secret | S3 credentials |
| `AWS_SECRET_ACCESS_KEY` | from K8s Secret | S3 credentials |

#### MLflow Deployment on OpenShift

```yaml
# Standalone (dev): SQLite + local S3
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://mlflow-artifacts/ \
  --host 0.0.0.0 --port 5000

# Production: PostgreSQL + S3
mlflow server \
  --backend-store-uri postgresql://user:pass@postgres:5432/mlflow \
  --default-artifact-root s3://mlflow-artifacts/ \
  --host 0.0.0.0 --port 5000
```

MLflow runs as a K8s Deployment in the same namespace as amortized. No operator needed — it's a stateless server backed by a database and S3.

#### Studio Integration

Studio UI queries MLflow for:
- **Datasets tab**: `GET /api/2.0/mlflow/runs/search` filtered by `tags.job_type = "sdg"` → list SDG outputs with names, sizes, creation dates
- **Job detail → Artifacts**: `GET /api/2.0/mlflow/runs/get?run_id={mlflow_run_id}` → show params, metrics, artifact links
- **Models tab**: `GET /api/2.0/mlflow/registered-models/search` → list registered models with versions and aliases
- **Lineage view**: follow `run.inputs.dataset_inputs` → source run → upstream datasets
- **Open in MLflow**: direct link to `{MLFLOW_URI}/#/experiments/{exp_id}/runs/{run_id}`

#### What Amortized Does NOT Build

- No `artifacts` table in amortized's SQLite database
- No custom artifact registry or metadata store
- No artifact upload/download endpoints (use MLflow's REST API)
- No separate lineage tracking system (OpenLineage, W&B, etc.)
- No Kubeflow Model Registry integration (MLflow's registry is sufficient)

#### What Amortized Does Build

- Ensures every job container gets the correct MLflow env vars
- Extracts `mlflow_run_id` from job logs after completion and stores it on the job record
- Registers models in MLflow Model Registry after successful training (if auto-register is enabled)
- Provides MLflow tracking URI via the health/config API so Studio knows where MLflow is
- Resolves model artifacts from MLflow when creating serve jobs (`training_job_id` → `mlflow_run_id` → artifact URI)

#### Gotchas (from testing)

- `MLFLOW_S3_ENDPOINT_URL` (not `AWS_S3_ENDPOINT_URL`) is what MLflow reads for MinIO
- `FSSPEC_S3_ENDPOINT_URL` is what s3fs/asynth reads for MinIO (separate env var)
- MLflow security middleware blocks internal cluster requests by default — use `--allowed-hosts *` or `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true`
- Training Hub installed from `amortized-ai/training_hub` fork (includes thub CLI + MLflow artifact upload)
- `mlflow.log_artifact()` with S3 paths requires `boto3` in the container
- MLflow logging requires `mlflow` package in the training container and `MLFLOW_TRACKING_URI` env var

---

### 2. Jobs — Thin Job Table + Kueue + K8s Watch

**Status**: Agreed

**What it covers**: job creation, queuing, dispatch, lifecycle management, status tracking, history

**Decision**: Thin job table for durable operations state. Kueue for GPU scheduling. K8s Watch API for status updates. MLflow for experiment data. No pipeline orchestration — jobs submitted sequentially by the agent or user.

#### Why a Job Table (not pure K8s + MLflow)

We evaluated two approaches:

| Approach | Verdict | Reason |
|---|---|---|
| **Option A: Kueue + MLflow only** | Not sufficient | K8s Jobs are ephemeral (garbage-collected via `ttlSecondsAfterFinished`). MLflow Runs don't exist until the container starts — can't represent "queued" state. If server crashes between API call and Job creation, request is lost. Query performance for "list my jobs" is poor via MLflow search API at scale. |
| **Option B: Thin job table + Kueue + MLflow** | **Use this** | Durable record at submission time. Fast queries for UI. Clean separation: table = operations, MLflow = science, K8s = runtime. This is what every production ML platform does (eval-hub, granite.build, Kubeflow Pipelines, Argo Workflows). |

#### Three-Layer Job Architecture

Each job lives across three systems, each responsible for a different concern:

```
┌─────────────────────────────────────────────────────────┐
│                    Amortized Job Table                    │
│  Operations layer: "what was submitted, what's queued"   │
│  Durable, fast queries, submission metadata              │
│  SQLite (dev) / PostgreSQL (prod)                        │
└──────────────┬──────────────────────┬────────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│   K8s Job + Kueue    │  │         MLflow Run            │
│  Runtime layer:      │  │  Science layer:               │
│  "what's running"    │  │  "what happened in the run"   │
│  Pod status, logs,   │  │  Params, metrics, artifacts,  │
│  GPU scheduling,     │  │  datasets, model registry,    │
│  admission control   │  │  lineage                      │
│  Ephemeral (GC'd)    │  │  Persistent (never deleted)   │
└──────────────────────┘  └──────────────────────────────┘
```

#### Job Types

Three job types, submitted sequentially. No pipeline DAG. No serving (Red Hat MaaS handles deployment).

| Type | Container Image | What It Runs | Output |
|---|---|---|---|
| **SDG** | `ghcr.io/amortized-ai/asynth:latest` | `asynth synthesize --config config.yaml` | generated_data.jsonl → S3 + MLflow |
| **Training** | `ghcr.io/amortized-ai/training:latest` | `thub <algo> --config config.yaml` | LoRA adapter → S3 + MLflow Model Registry |
| **Eval** | `ghcr.io/amortized-ai/asynth:latest` | `asynth judge --config config.yaml` | eval_results.jsonl → S3 + MLflow |

#### Job Table Schema

The table is a thin bridge between K8s (runtime) and MLflow (science). Stores only what neither K8s nor MLflow provides.

```sql
CREATE TABLE jobs (
    id              TEXT PRIMARY KEY,    -- UUID, amortized job ID
    type            TEXT NOT NULL,       -- 'sdg', 'training', 'eval'
    status          TEXT NOT NULL,       -- 'queued', 'provisioning', 'running', 'succeeded', 'failed', 'cancelled'
    config          TEXT NOT NULL,       -- JSON snapshot of submitted config
    recipe          TEXT DEFAULT '',     -- recipe name if used (e.g. 'examples/ticket-classifier/synth')
    user_id         TEXT DEFAULT '',     -- OpenShift user who submitted
    k8s_job_name    TEXT DEFAULT '',     -- K8s Job resource name (for lifecycle ops)
    k8s_namespace   TEXT DEFAULT '',     -- namespace where the Job runs
    mlflow_run_id   TEXT DEFAULT '',     -- set after job starts logging to MLflow
    mlflow_experiment TEXT DEFAULT '',   -- experiment name
    parent_job_id   TEXT DEFAULT '',     -- links training→SDG, eval→training (for lineage)
    error           TEXT DEFAULT '',     -- error message on failure
    created_at      TEXT NOT NULL,       -- submission time
    started_at      TEXT DEFAULT '',     -- when K8s scheduled the pod
    completed_at    TEXT DEFAULT ''      -- when the job finished
);
```

**What's NOT in the job table** (belongs elsewhere):
- Params, metrics, artifacts → MLflow
- Pod status, container logs → K8s API
- GPU scheduling, admission → Kueue

#### Job Lifecycle

```
User/Morty                    Amortized Server              K8s + Kueue                 Job Container
    │                              │                            │                           │
    │  POST /api/v1/jobs           │                            │                           │
    │─────────────────────────────►│                            │                           │
    │                              │ 1. Insert job record       │                           │
    │                              │    (status: queued)        │                           │
    │                              │                            │                           │
    │                              │ 2. Create K8s Job          │                           │
    │                              │    + ConfigMap + Secret    │                           │
    │                              │───────────────────────────►│                           │
    │                              │                            │ 3. Kueue suspends Job     │
    │                              │                            │    (waiting for GPU quota) │
    │                              │                            │                           │
    │                              │ ◄── K8s Watch event ──────│ 4. Kueue admits Job       │
    │                              │ Update: provisioning       │    (GPU available)         │
    │                              │                            │──────────────────────────►│
    │                              │                            │                           │ 5. Init container
    │                              │                            │                           │    downloads data from S3
    │                              │                            │                           │
    │                              │ ◄── K8s Watch event ──────│                           │ 6. Main container starts
    │                              │ Update: running            │                           │    Logs to MLflow
    │                              │                            │                           │
    │                              │                            │                           │ 7. Job completes
    │                              │ ◄── K8s Watch event ──────│ 8. K8s Job status:        │
    │                              │ 9. Extract mlflow_run_id   │    succeeded/failed       │
    │                              │    Update: succeeded       │                           │
    │                              │                            │                           │
    │  GET /api/v1/jobs/{id}       │                            │                           │
    │─────────────────────────────►│                            │                           │
    │  ◄── {status, mlflow_run_id, │                            │                           │
    │       config, timing, ...}   │                            │                           │
```

#### Job Status States

```
queued ──► provisioning ──► running ──► succeeded
                │               │
                │               └──► failed
                │
                └──► failed (admission rejected)

Any state ──► cancelled (user-initiated)
```

| Status | Meaning | Set By |
|---|---|---|
| `queued` | Job record created, K8s Job submitted, waiting for Kueue admission | Amortized on submission |
| `provisioning` | Kueue admitted the Job, pod is being scheduled / init containers running | K8s Watch (pod scheduled) |
| `running` | Main container is executing | K8s Watch (container started) |
| `succeeded` | Job completed successfully | K8s Watch (Job status.succeeded > 0) |
| `failed` | Job failed | K8s Watch (Job status.failed > 0) |
| `cancelled` | User cancelled the job | Amortized (deletes K8s Job) |

#### Kueue Integration

Kueue is GA on OpenShift (Red Hat build of Kueue 1.3). Amortized creates normal K8s Jobs with a Kueue label — Kueue handles admission and GPU scheduling.

```yaml
# Amortized adds this label to every K8s Job
metadata:
  labels:
    kueue.x-k8s.io/queue-name: "amortized-gpu-queue"
    amortized/job-id: "abc123"
    amortized/job-type: "training"
```

Platform engineer configures Kueue resources once:

```yaml
# ClusterQueue — defines GPU quota
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: amortized-cluster-queue
spec:
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: gpu-l40s
          resources:
            - name: "nvidia.com/gpu"
              nominalQuota: 4

# LocalQueue — namespace-scoped entry point
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: amortized-gpu-queue
  namespace: amortized-jobs
spec:
  clusterQueue: amortized-cluster-queue
```

Amortized doesn't implement job queuing, GPU scheduling, fair sharing, or admission control — Kueue handles all of it natively.

#### K8s Watch for Status Updates

Instead of polling K8s Jobs every 5 seconds (current approach), use the K8s Watch API for instant status updates:

```python
from kubernetes_asyncio import watch, client

async def watch_jobs(namespace: str):
    batch_v1 = client.BatchV1Api()
    w = watch.Watch()
    async for event in w.stream(
        batch_v1.list_namespaced_job,
        namespace,
        label_selector="app=amortized",
    ):
        job = event['object']
        job_id = job.metadata.labels.get('amortized/job-id')
        event_type = event['type']  # ADDED, MODIFIED, DELETED

        if job.status.succeeded and job.status.succeeded > 0:
            mlflow_run_id = await extract_mlflow_run_id(job)
            await update_job(job_id, status='succeeded', mlflow_run_id=mlflow_run_id)
        elif job.status.failed and job.status.failed > 0:
            await update_job(job_id, status='failed', error=get_failure_reason(job))
        elif job.status.active and job.status.active > 0:
            await update_job(job_id, status='running')
```

This is the OpenShift-native pattern — what operators use internally. Instant updates, zero wasted API calls.

#### Config Delivery

Same pattern as current implementation — it's already correct:

| Mechanism | What It Delivers | Mount Path |
|---|---|---|
| **ConfigMap** | Job config YAML/JSON | `/amortized/config.yaml` |
| **K8s Secret** (per-job) | LLM provider API keys | Env vars via `secretKeyRef` |
| **K8s Secret** (shared) | S3 credentials | `envFrom: secretRef` |
| **Init container** | Data download from S3 | `/amortized/work/` |
| **OwnerReference** | ConfigMap + Secret owned by Job | Auto-cleanup on Job deletion |

#### Job Chaining

No automatic pipeline. The agent (Morty) or user chains jobs manually:

1. Submit SDG → get `sdg_job_id`
2. SDG completes → MLflow run has dataset at `s3://...`
3. Submit Training with `parent_job_id: sdg_job_id` and `data_path: s3://...`
4. Training completes → MLflow run has model, registered in Model Registry
5. Submit Eval with `parent_job_id: training_job_id`

The `parent_job_id` field is for lineage and display only. The control plane provides a helper endpoint: given a `job_id`, return its MLflow run artifacts (datasets, models, S3 URIs). The agent uses this to find the right artifact to pass to the next job.

#### Reference Architecture Comparison

| Aspect | **eval-hub** | **granite.build** | **amortized** |
|---|---|---|---|
| Job storage | PostgreSQL `evaluation_jobs` | SQLite/PostgreSQL `StoredBuild` + 2 child tables | SQLite/PostgreSQL `jobs` (single table) |
| Dispatch | Fire-and-forget goroutine → K8s Job | Poll loop → Environment.launch() | K8s Job + Kueue label |
| Status updates | Sidecar pushes via HTTP callback | Event bus (asyncio.Queue) | K8s Watch API (informers) |
| Config delivery | ConfigMap at `/meta/job.json` | ConfigMap for config files | ConfigMap at `/amortized/config.yaml` |
| Queuing | None (immediate dispatch) | None (sequential within build) | Kueue (GPU quota + fair sharing) |
| Multi-container | Init + Adapter + Sidecar | Single container | Init + Main (no sidecar for v1) |
| Job types | Single (evaluation) | Multi-target DAG | Three sequential (SDG, training, eval) |

#### What Amortized Does NOT Build

- No job queuing logic (Kueue handles it)
- No GPU scheduling (Kueue + K8s scheduler)
- No pipeline DAG execution (granite.build pattern — not needed)
- No sidecar containers for status reporting (eval-hub pattern — K8s Watch is sufficient for v1)
- No serve/deployment jobs (Red Hat MaaS handles serving)

#### What Amortized Builds

- Job table (thin, durable, fast queries)
- K8s Job creation with ConfigMap, Secret, Kueue label, and MLflow env vars
- K8s Watch loop for status updates
- MLflow run ID extraction from job logs on completion
- Job API: create, list, get, cancel
- Helper: resolve MLflow artifacts for a completed job (for chaining)

---

### 3. SDG (Synthetic Data Generation)

**Status**: To discuss

Questions:
- Use SDG Hub (Red Hat) or asynth (ours)? Or both?
- How do we run SDG on the cluster? K8s Job dispatched by amortized with Kueue?
- How do recipes/flows map to SDG Hub's blocks and flows?
- How does the teacher model API key get injected?

---

### 4. Training

**Status**: To discuss

Questions:
- Training Hub is the algorithm routing layer — do we call it directly?
- Kubeflow Trainer (`TrainJob` CRD) is lagging behind Training Hub — so we dispatch our own K8s Jobs?
- How do we get the SDG output (from MLflow/S3) into the training container?
- How do training metrics flow back to MLflow?

---

### 5. Evaluation

**Status**: To discuss

Questions:
- Use Eval Hub (Red Hat) or build our own?
- What evaluations matter for task models? (accuracy, latency, cost, quality)
- How do eval results feed back into the workflow? (gate before serving)

---

### 6. Job Orchestration

**Status**: Resolved — no separate orchestration needed

Decided: No pipeline DAG. Jobs submitted sequentially by the agent (Morty) or user. Kueue handles queuing and GPU scheduling. K8s Watch handles lifecycle. See Jobs component above.

---

### 3. Data Connectors — RHOAI Native + MLflow AI Gateway

**Status**: Agreed

**What it covers**: S3 storage, HuggingFace models/datasets, document ingestion, LLM provider API key management

**Decision**: Amortized builds zero custom data connectors. Everything is handled by RHOAI native services or MLflow.

#### Why No Custom Data Connectors

We evaluated what each data connection needs and found that RHOAI and MLflow already handle every case:

| Data Connector | Handled By | What Amortized Builds |
|---|---|---|
| **S3 credentials** | RHOAI data connections (K8s Secret with `opendatahub.io/connection-type-ref: s3`) | Nothing — reference the namespace's data connection Secret in job pods |
| **S3 read/write** | MLflow artifact store (stores/retrieves via S3) + init containers (`aws s3 cp/sync`) | Nothing — MLflow handles artifact I/O, init containers handle bulk data download |
| **HuggingFace models** | Auto-download via `model_path` in Training Hub containers | Nothing — pass-through config parameter |
| **HuggingFace datasets** | `datasets` library in containers, `input_data` in SDG config | Nothing — pass-through config parameter |
| **Document ingestion** | Docling (Python library included in SDG container image) | Nothing — user uploads docs, SDG flow uses Docling to extract text |
| **LLM provider API keys** | MLflow AI Gateway (see below) | Nothing — no custom key store |

#### RHOAI Data Connections

On OpenShift AI, a "data connection" is a K8s Secret with specific labels that the RHOAI dashboard manages:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-s3-connection
  namespace: amortized
  labels:
    opendatahub.io/dashboard: 'true'
    opendatahub.io/managed: 'true'
  annotations:
    opendatahub.io/connection-type-ref: s3
data:
  AWS_ACCESS_KEY_ID: <base64>
  AWS_SECRET_ACCESS_KEY: <base64>
  AWS_S3_ENDPOINT: <base64>
  AWS_S3_BUCKET: <base64>
  AWS_DEFAULT_REGION: <base64>
```

Created via the RHOAI dashboard or `oc create secret`. Job pods reference it via `envFrom: secretRef`. Platform engineer sets this up once per namespace.

Amortized job pods use this Secret for:
- Init container S3 downloads (training data)
- MLflow artifact store writes (SDG output, model weights)
- Any direct S3 operations in job containers

#### MLflow AI Gateway — LLM Provider API Key Management

**Decision**: Use MLflow AI Gateway instead of building our own encrypted API key store.

The MLflow AI Gateway (built into MLflow ≥ 3.0) is a database-backed LLM proxy that provides:

| Capability | How It Works |
|---|---|
| **Centralized API key storage** | Provider keys stored encrypted on the MLflow server. Never exposed to containers or users. |
| **Unified OpenAI-compatible API** | All providers (OpenAI, Anthropic, Google, OpenRouter, Ollama, 30+ more) accessible via one endpoint |
| **No key injection needed** | SDG/eval containers call `http://mlflow:5000/gateway/v1/chat/completions` instead of provider APIs directly. No `OPENAI_API_KEY` in pods. |
| **Provider switching** | Change the gateway endpoint config → all jobs use the new provider. No job config changes. |
| **Failover** | Automatic fallback to backup models on provider failure |
| **Budget tracking** | Per-endpoint and per-user token budgets |
| **Traffic splitting** | Route percentages of requests to different models for A/B testing |

**Before (custom key management)**:
```
User adds API key in Studio → stored encrypted in amortized's SQLite
  → amortized creates per-job K8s Secret with the key
  → SDG container reads OPENAI_API_KEY from Secret
  → SDG container calls openai.com directly
```

**After (MLflow AI Gateway)**:
```
Platform engineer configures provider endpoint in MLflow Gateway UI
  → MLflow stores key encrypted server-side
  → SDG container calls http://mlflow:5000/gateway/v1/chat/completions
  → MLflow Gateway routes to the right provider with server-side credentials
  → No API keys in containers, no per-job Secrets, no custom key store
```

**What this eliminates from amortized**:
- `api_keys` table in SQLite
- `add_api_key` / `list_api_keys` / `delete_api_key` API endpoints
- Encrypted key storage code
- Per-job Secret creation for API keys
- Key injection logic in the worker
- Provider key management in Studio settings

**How SDG/eval containers use it**:

Instead of calling the LLM provider directly, asynth/SDG Hub is configured to use the MLflow Gateway as the LLM endpoint:

```yaml
# SDG config — model points to MLflow Gateway
model: "gateway/gpt-4o-mini"  # or whatever endpoint name is configured
api_base: "http://mlflow.amortized.svc:5000/gateway/v1"
```

The gateway routes to the right provider using its server-side credentials. The container never sees an API key.

#### Model Storage Options on RHOAI

For reference — RHOAI supports three model storage mechanisms:

| Storage | URI Format | Use Case |
|---|---|---|
| **S3** | `s3://bucket/path/to/model` | Default for MLflow artifacts. LoRA adapters stored here. |
| **OCI containers** | `oci://registry.redhat.io/...` | "Modelcars" — models packaged as container images. Faster cold start. |
| **HuggingFace** | `hf://huggingface.co/org/model` | Base models downloaded on-demand by training/serving containers |

Task models (LoRA adapters) produced by amortized training jobs go to S3 via MLflow. If they need to be served via MaaS, they can be registered in Model Registry with the S3 URI, or packaged as OCI images for production deployment.

---

### 4. Experience Architecture — Studio as Single Frontend

**Status**: Agreed

**What it covers**: how users interact with the platform, which UI they see, how Studio integrates with MLflow

**Decision**: Studio is the single user-facing frontend. MLflow is a backend service. Users never need to open MLflow's UI.

#### Why a Single Frontend

Two UIs (Studio + MLflow) creates a fragmented experience:
- "Go to Studio to submit jobs, but go to MLflow to see your experiments"
- "Configure API keys in MLflow UI, but see your datasets in Studio"
- Users need to learn two interfaces, two mental models, two auth flows

One UI (Studio) with MLflow as a backend API:
- Studio is the only interface data scientists interact with
- Studio calls MLflow's REST APIs for datasets, models, metrics, API key management
- Consistent experience, one URL, one auth flow
- MLflow UI remains available for platform engineers and power users who want raw experiment analysis

#### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Studio (React SPA)                        │
│                                                               │
│  ┌─────────┐ ┌──────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ │
│  │  Chat   │ │ Jobs │ │ Datasets │ │ Models │ │ Settings │ │
│  │ (Morty) │ │      │ │          │ │        │ │          │ │
│  └────┬────┘ └──┬───┘ └────┬─────┘ └───┬────┘ └────┬─────┘ │
│       │         │          │            │           │        │
└───────┼─────────┼──────────┼────────────┼───────────┼────────┘
        │         │          │            │           │
        ▼         ▼          ▼            ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌──────────────────────────────┐
   │OpenCode │ │Amortized│ │          MLflow               │
   │ (Morty) │ │  API    │ │                               │
   │ :4096   │ │ :8000   │ │  REST API    AI Gateway       │
   │         │ │         │ │  :5000/api   :5000/gateway    │
   └─────────┘ └─────────┘ └──────────────────────────────┘
                    │                     │
                    ▼                     ▼
              ┌──────────┐         ┌───────────┐
              │ Job Table│         │ MLflow DB │
              │ (SQLite/ │         │ (SQLite/  │
              │  PG)     │         │  PG)      │
              └──────────┘         └───────────┘
```

#### Studio Page → Backend API Mapping

| Studio Page | Primary Backend | MLflow API Used | What Users See |
|---|---|---|---|
| **Chat** | OpenCode → MCP → Amortized API | — | Morty conversation for guided task model building |
| **Jobs** | Amortized `/api/v1/jobs` | `GET /api/2.0/mlflow/runs/get` (for metrics/params on job detail) | Job list with status, timing, type. Drill into job for MLflow metrics. |
| **Job Detail → Metrics** | — | `GET /api/2.0/mlflow/runs/get?run_id={mlflow_run_id}` | Loss curves, learning rate, eval scores, training params |
| **Job Detail → Artifacts** | — | `GET /api/2.0/mlflow/artifacts/list?run_id={mlflow_run_id}` | Browse/preview/download output files (datasets, models, logs) |
| **Job Detail → Logs** | Amortized API → K8s pod logs | — | Real-time container logs while job is running |
| **Datasets** | — | `POST /api/2.0/mlflow/runs/search` with `filter_string="tags.job_type='sdg'"` | List of generated datasets with name, size, creation date, source recipe |
| **Dataset Detail** | — | `GET /api/2.0/mlflow/runs/get` + `GET /api/2.0/mlflow/artifacts/list` | Preview rows, schema, quality scores, lineage to training jobs that consumed it |
| **Models** | — | `GET /api/2.0/mlflow/registered-models/search` | Registered models with versions, aliases (@champion), training lineage |
| **Model Detail** | — | `GET /api/2.0/mlflow/model-versions/search` + `GET /api/2.0/mlflow/runs/get` | Version history, metrics, params, linked training run and SDG dataset |
| **Recipes** | Amortized `/api/v1/recipes` | — | Pre-built task model templates (ticket-classifier, entity-extractor, etc.) |
| **Settings → LLM Providers** | — | MLflow AI Gateway API: `GET/POST /api/2.0/mlflow/gateway/routes` | Configure provider endpoints, add/remove API keys, view usage |
| **Settings → Data Connections** | — | RHOAI dashboard API or direct K8s Secret management | S3 connection config (platform engineer concern, not data scientist) |
| **Settings → Compute** | Amortized `/api/v1/compute` | — | View available backends, Kueue queue status |

#### MLflow APIs Studio Needs

Studio's frontend calls these MLflow REST API endpoints (proxied through nginx or amortized):

**Experiment Tracking**:
- `POST /api/2.0/mlflow/runs/search` — list runs (jobs) by experiment, filter by tags
- `GET /api/2.0/mlflow/runs/get?run_id=X` — get run detail (params, metrics, tags)
- `GET /api/2.0/mlflow/metrics/get-history?run_id=X&metric_key=loss` — metric time series for charts

**Artifacts**:
- `GET /api/2.0/mlflow/artifacts/list?run_id=X` — list artifacts in a run
- `GET /api/2.0/mlflow/artifacts/get?run_id=X&path=Y` — download artifact content (for preview)

**Model Registry**:
- `GET /api/2.0/mlflow/registered-models/search` — list registered models
- `GET /api/2.0/mlflow/model-versions/search?filter=name='ticket-classifier'` — list versions
- `POST /api/2.0/mlflow/registered-models/alias` — set alias (@champion)

**AI Gateway** (for API key management):
- `GET /api/2.0/mlflow/gateway/routes` — list configured provider endpoints
- `POST /api/2.0/mlflow/gateway/routes` — create new provider endpoint with API key
- `DELETE /api/2.0/mlflow/gateway/routes/{name}` — remove provider endpoint

**Dataset Tracking**:
- Datasets are logged as run inputs — query via `runs/get` and inspect `run.inputs.dataset_inputs`

#### Nginx Routing

Studio's nginx config routes requests to the right backend:

```nginx
# Studio SPA
location / {
    root /usr/share/nginx/html;
    try_files $uri $uri/ /index.html;
}

# Amortized API (jobs, recipes, compute)
location /api/ {
    proxy_pass http://amortized-server:8000;
}

# Agent (Morty via OpenCode)
location /agent/ {
    proxy_pass http://opencode:4096/;
}

# MLflow API (experiments, models, artifacts, gateway)
location /mlflow/ {
    proxy_pass http://mlflow:5000/;
}
```

Studio's API client uses:
- `/api/v1/...` for amortized-specific endpoints (jobs, recipes)
- `/mlflow/api/2.0/mlflow/...` for MLflow endpoints (runs, models, artifacts, gateway)
- `/agent/...` for OpenCode/Morty

#### MLflow UI Access

MLflow's built-in UI is still deployed and accessible at its own Route/URL (e.g., `mlflow.apps.cluster.example.com`). It's available for:
- Platform engineers debugging experiment data
- Power users who want raw experiment comparison charts
- Advanced analysis (metric scatter plots, parallel coordinates, etc.)

But **data scientists building task models use Studio only**. They never need to know MLflow exists.

#### What Amortized Does NOT Build (Experience)

- No custom experiment comparison UI (MLflow UI for power users)
- No custom metric charting library (use MLflow's metric history API + a simple chart component in Studio)
- No custom model registry UI (Studio queries MLflow's Model Registry API)
- No custom API key management backend (MLflow AI Gateway handles storage and routing)
- No custom artifact storage browser (Studio queries MLflow's artifact API)

#### What Amortized Builds (Experience)

- **Studio frontend**: React SPA that is the single user interface
- **Studio ↔ MLflow proxy**: nginx routes `/mlflow/` to MLflow server
- **Job-centric views**: Studio shows jobs (from amortized's table) enriched with MLflow data (metrics, artifacts, lineage)
- **Morty chat**: Agent interface for guided workflow
- **Recipe browser**: Template selection and configuration
- **Unified search**: "Show me all ticket-classifier experiments" → queries both job table and MLflow

---

### 5. API — Amortized REST + Two MCP Servers

**Status**: Agreed

**What it covers**: amortized's REST API surface, MCP for agent (Morty), how Studio talks to all backends

**Decision**: Amortized's API is thin — jobs + recipes only. MLflow handles everything else. Morty connects to two MCP servers: amortized (jobs/recipes) and MLflow (experiments/models/datasets/deployments). Studio proxies both via nginx.

#### Why Two MCP Servers (Not One Facade)

| Approach | Verdict | Reason |
|---|---|---|
| **Single facade**: Amortized wraps MLflow, exposes everything as one MCP | More code, more maintenance | Every MLflow feature update requires amortized wrapper updates. We become a bottleneck. |
| **Two MCP servers**: Amortized for jobs/recipes, MLflow for everything else | **Use this** | Each service exposes its native capabilities directly. MLflow MCP already has 45 tools covering experiments, models, deployments, traces. No wrapper code needed. |

#### Amortized REST API — What We Build

Amortized's API only exposes what K8s and MLflow don't provide: job management and recipes.

**Jobs** (our job table):

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/jobs` | Create a job (SDG, training, or eval) |
| `GET` | `/api/v1/jobs` | List jobs (filter by type, status, user) |
| `GET` | `/api/v1/jobs/{id}` | Get job detail (status, config, timing, mlflow_run_id) |
| `DELETE` | `/api/v1/jobs/{id}` | Cancel a running job (deletes K8s Job) |
| `GET` | `/api/v1/jobs/{id}/logs` | Stream pod logs (proxies K8s pod log API) |

**Recipes** (pre-built task model templates):

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/recipes` | List all available recipes |
| `GET` | `/api/v1/recipes/{name}` | Get recipe detail (config, description) |
| `POST` | `/api/v1/jobs/recipe` | Submit a job from a recipe with overrides |

**System**:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/config` | Platform config (MLflow URI, Kueue queue, namespace) |

**That's ~10 endpoints.** Down from 57 in the current amortized API.

#### What We Removed (Handled by MLflow or RHOAI)

| Removed Endpoint | Now Handled By |
|---|---|
| `POST /api/v1/jobs/training` | Merged into `POST /api/v1/jobs` with `type: training` |
| `POST /api/v1/jobs/sdg` | Merged into `POST /api/v1/jobs` with `type: sdg` |
| `GET /api/v1/jobs/{id}/metrics` | MLflow: `describe_run` tool / `runs/get` API |
| `GET /api/v1/jobs/{id}/artifacts` | MLflow: artifacts API |
| `POST/GET/DELETE /api/v1/artifacts` | MLflow: artifact tracking |
| `POST /api/v1/datasets` | MLflow: `mlflow.data` |
| `GET /api/v1/datasets/{path}/preview` | MLflow: artifacts API |
| `POST /api/v1/datasets/convert` | Handled inside SDG/training containers |
| `POST/GET/DELETE /api/v1/settings/api-keys` | MLflow AI Gateway |
| `GET/POST /api/v1/evaluators` | Eval Hub (if used) |
| `POST /api/v1/evaluations` | Eval Hub (if used) |
| `POST /api/v1/estimate` | Training Hub |
| `GET /api/v1/flows` | Recipes replace flows |
| `POST /api/v1/judge` | Part of eval jobs |
| `GET /api/v1/compute` | Kueue (K8s native) |
| `GET/POST /api/v1/settings/backends` | RHOAI data connections |

#### MCP Architecture — Two Servers for Morty

Morty (via OpenCode) connects to two MCP servers simultaneously:

```
┌──────────────────────────────────────────────────────────┐
│                    Morty (OpenCode)                        │
│                                                           │
│         MCP Client ──────────► MCP Client                 │
│              │                      │                     │
└──────────────┼──────────────────────┼─────────────────────┘
               │                      │
               ▼                      ▼
    ┌──────────────────┐   ┌──────────────────────────────┐
    │  Amortized MCP   │   │       MLflow MCP              │
    │  (fastapi-mcp)   │   │  (mlflow mcp run)             │
    │                  │   │                               │
    │  ~10 tools:      │   │  45 tools:                    │
    │  • create_job    │   │                               │
    │  • list_jobs     │   │  Experiments:                 │
    │  • get_job       │   │  • create_experiment          │
    │  • cancel_job    │   │  • search_experiments         │
    │  • stream_logs   │   │  • get_experiment             │
    │  • list_recipes  │   │                               │
    │  • get_recipe    │   │  Runs:                        │
    │  • submit_recipe │   │  • list_runs                  │
    │  • health        │   │  • describe_run               │
    │  • config        │   │  • create_run                 │
    │                  │   │                               │
    └──────────────────┘   │  Models:                      │
                           │  • serve_model                │
                           │  • predict_with_model         │
                           │                               │
                           │  Deployments (AI Gateway):    │
                           │  • create_deployment_endpoint │
                           │  • list_deployment_endpoints  │
                           │  • get_deployment_endpoint    │
                           │  • predict_with_deployment    │
                           │                               │
                           │  Traces:                      │
                           │  • search_traces              │
                           │  • get_trace                  │
                           │  • evaluate_traces            │
                           │                               │
                           │  Scorers:                     │
                           │  • list_scorers               │
                           │  • register_llm_judge_scorer  │
                           └──────────────────────────────┘
```

**MLflow MCP tool categories** (45 tools total, from MLflow 3.13.0):

| Category | Tools | What Morty Uses Them For |
|---|---|---|
| **Experiments** (7) | `create_experiment`, `search_experiments`, `get_experiment`, `update_experiment`, `delete_experiment`, `restore_experiment`, `rename_experiment` | Browse and manage experiment namespaces |
| **Runs** (5) | `list_runs`, `describe_run`, `create_run`, `delete_run`, `restore_run` | Check job results, metrics, params after completion |
| **Traces** (10) | `search_traces`, `get_trace`, `delete_traces`, `set_trace_tag`, `delete_trace_tag`, `log_trace_feedback`, `log_trace_expectation`, `get_trace_assessment`, `update_trace_assessment`, `delete_trace_assessment` | Debug LLM call chains in SDG jobs |
| **Deployments/Gateway** (12) | `create_deployment_endpoint`, `list_deployment_endpoints`, `get_deployment_endpoint`, `update_deployment_endpoint`, `delete_deployment_endpoint`, `create_deployment`, `list_deployments`, `get_deployment`, `update_deployment`, `delete_deployment`, `predict_with_deployment`, `explain_deployment` | Manage AI Gateway provider endpoints (API keys), test models |
| **Models** (6) | `serve_model`, `predict_with_model`, `prepare_model_env`, `generate_model_dockerfile`, `build_model_docker`, `update_model_pip_requirements` | Inspect and test trained models |
| **Scorers** (2) | `list_scorers`, `register_llm_judge_scorer` | Set up evaluation scorers for data quality |
| **Evaluation** (1) | `evaluate_traces` | Run evaluations on trace data |
| **Other** (2) | `link_traces_to_run`, `run_deployment_locally` | Linkage and local testing |

**OpenCode MCP config** (`.opencode.json`):

```json
{
  "mcp": {
    "amortized": {
      "type": "remote",
      "url": "http://amortized-server:8000/mcp",
      "enabled": true
    },
    "mlflow": {
      "type": "local",
      "command": ["mlflow", "mcp", "run"],
      "environment": {
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "MLFLOW_MCP_TOOLS": "all"
      },
      "enabled": true
    }
  }
}
```

`MLFLOW_MCP_TOOLS=all` enables all 45 tools. Morty can discover them automatically.

#### How Studio Calls Each Backend

Studio's nginx routes to three backends:

```nginx
# Amortized API (jobs, recipes)
location /api/ {
    proxy_pass http://amortized-server:8000;
}

# MLflow API (experiments, models, artifacts, gateway)
location /mlflow/ {
    proxy_pass http://mlflow:5000/;
}

# Agent (Morty via OpenCode)
location /agent/ {
    proxy_pass http://opencode:4096/;
}
```

Studio's TypeScript API client:
- `/api/v1/jobs` → amortized (job list, create, cancel)
- `/api/v1/recipes` → amortized (recipe browser)
- `/mlflow/api/2.0/mlflow/runs/search` → MLflow (datasets, experiment history)
- `/mlflow/api/2.0/mlflow/registered-models/search` → MLflow (model registry)
- `/mlflow/api/2.0/mlflow/gateway/routes` → MLflow (API key management)
- `/mlflow/ajax-api/2.0/mlflow/artifacts/list` → MLflow (artifact browser)
- `/agent/session/{id}/message` → OpenCode (Morty chat)

#### Amortized MCP via fastapi-mcp

Amortized continues to use `fastapi-mcp` to auto-generate MCP tools from its OpenAPI spec. With the reduced API surface (~10 endpoints), this produces ~10 clean MCP tools for Morty:

```python
# src/amortized/mcp/server.py (unchanged)
from fastapi_mcp import FastApiMCP

def create_mcp_server(app: FastAPI) -> FastApiMCP:
    mcp = FastApiMCP(
        app,
        name="amortized",
        description="Task model control plane — submit and manage SDG, training, and eval jobs.",
    )
    mcp.mount_http(app)
    return mcp
```

#### What Amortized Does NOT Build (API)

- No artifact CRUD endpoints (MLflow)
- No dataset management endpoints (MLflow)
- No model registry endpoints (MLflow)
- No API key management endpoints (MLflow AI Gateway)
- No evaluator CRUD endpoints (Eval Hub)
- No compute backend management endpoints (Kueue)
- No VRAM estimation endpoint (Training Hub)
- No MLflow wrapper/facade (MLflow has its own MCP and REST API)

---

### 6. Agent (Morty) — OpenCode + Skills + Structured UI

**Status**: Agreed (details to be refined)

**What it covers**: the AI agent that guides users through task model creation, its runtime, workflow enforcement, and how it integrates with Studio

**Decision**: OpenCode is the agent runtime. Morty connects to two MCP servers (amortized + MLflow). Workflow, gates, and skills are defined separately and iterated on. LLM backend must be easily switchable at deployment time.

#### Agent Runtime: OpenCode

| Decision | Choice | Reason |
|---|---|---|
| **Runtime** | OpenCode (`opencode serve`) | Already working. Built-in MCP support, session management, HTTP API for Studio integration. |
| **Identity** | Custom agent definition (`morty.md`) | `.opencode/agents/morty.md` with system prompt (identity + capabilities + workflow). Skill guides loaded on demand from `skills/` directory. Built-in agents (build, plan) disabled. |
| **LLM Backend** | Configurable at deploy time | Platform engineer sets the provider and model in `.opencode.json`. No code changes to switch between Vertex AI (Claude), OpenAI, Anthropic direct, or on-cluster vLLM. |
| **MCP Servers** | Two: amortized (jobs/recipes) + MLflow (experiments/models/datasets/gateway) | See API component for details. |
| **Permission scoping** | Deny all filesystem tools, allow only MCP | Morty can't edit files, run bash, or access the filesystem. Only MCP tools. |

**LLM switching** — change one config value:

```json
// .opencode.json — platform engineer configures at deployment
{
  "model": "google-vertex-anthropic/claude-opus-4-6@default"
}
// or
{
  "model": "anthropic/claude-sonnet-4-20250514"
}
// or
{
  "model": "openai/gpt-4o"
}
// or on-cluster vLLM:
{
  "model": "hosted-vllm/my-model",
  "providers": {
    "hosted-vllm": {
      "baseURL": "http://vllm.amortized.svc:8000/v1"
    }
  }
}
```

No image rebuild, no code change. OpenCode supports 75+ providers natively.

#### Workflow (To Be Defined)

The task model creation workflow will follow a structured plan, similar to Oumi's approach:

```
1. UNDERSTAND    — What task? What domain? What data?
2. PLAN          — Define evaluators, choose recipe, set parameters
3. GENERATE      — Create test data (small sample for validation)
4. EVALUATE      — Evaluate baseline model on test data
5. SYNTHESIZE    — Generate full training dataset (SDG)
6. REVIEW        — Preview generated data, check quality
7. TRAIN         — Fine-tune model (LoRA SFT)
8. EVALUATE      — Evaluate fine-tuned model vs baseline
9. REGISTER      — Register model in MLflow Model Registry
10. ITERATE      — Review results, adjust and repeat if needed
```

The specific steps, entry/exit criteria, and enforcement mechanism are to be designed separately. This workflow will be encoded as a **skill** that Morty follows.

#### Confirmation Gates (To Be Defined)

Before every job submission, Morty must:
1. Show the full job configuration in a structured format
2. Present explicit **Approve / Adjust** options
3. Wait for user confirmation before calling the submit tool
4. Never auto-submit jobs

Enforcement mechanism options (to be decided):
- **System prompt instruction** — "NEVER call create_job without showing config and getting explicit approval"
- **OpenCode skill** — a skill that wraps the job submission tool with a confirmation step
- **OpenCode permission hook** — a stop hook that intercepts job creation tool calls and requires user approval
- **Studio-side gate** — Studio intercepts the tool call and renders a confirmation card before allowing it through

Reference: Oumi's "Waiting for approval" → "Approved" pattern with links to Jobs and Recipe pages.

#### Skills Architecture (To Be Defined)

MCP tool organization and workflow steps will be implemented as OpenCode skills:

- **Plan skill** — Generates the workflow plan checklist, tracks progress across steps
- **SDG skill** — Guides through data generation (asks questions, picks recipe, configures, confirms, submits)
- **Training skill** — Guides through model training (picks base model, configures LoRA, confirms, submits)
- **Eval skill** — Guides through evaluation (picks benchmarks, runs eval, compares results)
- **Debug skill** — Checks job status, streams logs, diagnoses failures

Skills provide structure that the system prompt alone cannot enforce reliably.

#### Job Handling: Blocking with Background Option

Following Oumi's pattern, job execution is **blocking by default** with an option to run in background:

**Blocking (default)**:
```
User: "Generate training data for ticket classification"
Morty: [shows config, gets approval]
Morty: "SDG job submitted (Job #21). I'll monitor it for you."
       [Job #21 card — "View on Jobs page"]
       ...polling status...
Morty: "SDG job completed! 200 samples generated. Let me preview the data..."
       [shows data preview]
```

Morty stays engaged — checks status, reports progress, handles completion. The user can see the job card and navigate to the Jobs page, but the conversation stays focused on this job.

**Background (user-initiated)**:
```
Morty: "SDG job submitted (Job #21). I'll monitor it for you."
User: "Run it in the background, I have another question"
Morty: "Got it — Job #21 is running in the background. I'll let you know when it completes. What's your question?"
```

The conversation continues with other topics. When the job completes, Morty picks it up in the next interaction.

#### Structured UI Components

The agent outputs structured data that Studio renders as interactive widgets. This requires Studio to recognize specific patterns in Morty's responses:

**Plan checklist** (Image 9 pattern):
```markdown
<!-- Studio renders this as a collapsible checklist -->
**Plan** Next: Generate test data
- [x] Define evaluators
- [ ] Generate test data
- [ ] Evaluate baseline model
- [ ] Create training data
- [ ] Fine-tune model
- [ ] Evaluate fine-tuned model
- [ ] Review & iterate
```

**Confirmation card** (Image 10 pattern):
```markdown
<!-- Studio renders this as an approval card -->
**Ready to generate test data?**

| Setting | Value |
|---|---|
| Recipe | ticket-classifier/synth |
| Model | openai/gpt-4o-mini |
| Samples | 50 |

**[Looks good, run it]** — Generate the 50-sample test dataset
**[I'd like to adjust]** — Change the number of samples or other settings
```

**Job card** (Image 10 pattern):
```markdown
<!-- Studio renders this as a job status card -->
**Job #21** — SDG (ticket-classifier)
Status: Running
[View on Jobs page](/jobs/21) | [View Recipe](/recipes/ticket-classifier/synth)
```

The exact rendering format (markdown conventions vs structured JSON in tool results) is a Studio implementation detail to be designed.

#### What Amortized Does NOT Build (Agent)

- No custom agent runtime (using OpenCode)
- No custom LLM integration (OpenCode handles 75+ providers)
- No custom MCP client (OpenCode's built-in MCP client)
- No custom session management (OpenCode's session API)
- No custom conversation persistence for the agent (OpenCode handles sessions; Studio has its own local conversation store for the UI)

#### What Amortized Builds (Agent)

- **morty.md** — Custom agent definition (identity + capabilities + workflow), built from `agent/prompts/`
- **Skills** — On-demand expertise in `agent/skills/` (sdg, training, eval skill trees with guidance routers, sub-skill guides, and config templates). Delivered via ConfigMap + initContainer
- **Studio components** — Plan checklist, confirmation cards, job cards, approval buttons
- **OpenCode deployment** — K8s Deployment + ConfigMaps (morty-config, morty-skills) + Secret for LLM credentials

---

### 7. Auth & Multi-tenancy — OpenShift Native

**Status**: Agreed

**What it covers**: user authentication, authorization, tenant isolation

**Decision**: Use OpenShift OAuth for auth, K8s RBAC for authorization. Design for namespace-per-tenant isolation, but start with single namespace + user tags for v1.

#### Authentication — OpenShift OAuth

No custom login system. Users authenticate via OpenShift OAuth/OIDC, which every OpenShift cluster provides.

**Flow**:
```
User opens Studio → redirected to OpenShift login → gets OAuth token
  → Studio passes token in Authorization header
  → amortized validates via K8s TokenReview API
  → extracts user identity (username, groups)
```

**How amortized validates tokens**:
```python
# Validate incoming bearer token via K8s TokenReview
from kubernetes_asyncio.client import AuthenticationV1Api

async def authenticate(token: str) -> str:
    auth_api = AuthenticationV1Api()
    review = await auth_api.create_token_review({
        "apiVersion": "authentication.k8s.io/v1",
        "kind": "TokenReview",
        "spec": {"token": token}
    })
    if review.status.authenticated:
        return review.status.user.username
    raise HTTPException(401, "Invalid token")
```

This is the standard OpenShift pattern — no custom auth database, no password storage, no session management. OpenShift handles all of it.

**Studio auth flow**:
- Studio is served behind the OpenShift Route (HTTPS with edge TLS)
- OpenShift OAuth proxy can be deployed as a sidecar (standard RHOAI pattern)
- Or Studio redirects to OpenShift login page and stores the token client-side
- All API calls include `Authorization: Bearer <token>`

#### Authorization — K8s RBAC

Use OpenShift's built-in RBAC to control who can do what:

| Action | Required Permission | K8s Resource |
|---|---|---|
| Submit a job | `create` on `jobs` in the namespace | Custom RBAC Role |
| View jobs | `get`, `list` on `jobs` | Custom RBAC Role |
| Cancel a job | `delete` on `jobs` | Custom RBAC Role |
| View recipes | `get`, `list` on `configmaps` (recipes) | Standard view role |
| Configure providers | Access to MLflow AI Gateway | MLflow-level auth |

**Two roles**:
- **Task Model Builder** (data scientist): submit jobs, view results, use Morty
- **Platform Admin**: configure Kueue queues, MLflow, data connections, provider endpoints

Mapped to OpenShift groups via RHOAI dashboard settings ("Data Science user groups" / "Data Science administrator groups").

For v1, amortized can check authorization via `SubjectAccessReview`:
```python
# Check if user can create jobs in this namespace
from kubernetes_asyncio.client import AuthorizationV1Api

async def authorize(username: str, action: str, resource: str, namespace: str) -> bool:
    auth_api = AuthorizationV1Api()
    review = await auth_api.create_subject_access_review({
        "apiVersion": "authorization.k8s.io/v1",
        "kind": "SubjectAccessReview",
        "spec": {
            "user": username,
            "resourceAttributes": {
                "namespace": namespace,
                "verb": action,
                "resource": resource,
                "group": "amortized.ai"
            }
        }
    })
    return review.status.allowed
```

#### Multi-tenancy — Two Phases

**v1: Single Namespace + User Tags**

All users share one namespace. Jobs tagged with the submitting user:

```yaml
# K8s Job labels
metadata:
  labels:
    amortized/job-id: "abc123"
    amortized/user: "shiv"
    amortized/job-type: "training"
```

```sql
-- Job table has user_id column
SELECT * FROM jobs WHERE user_id = 'shiv' ORDER BY created_at DESC;
```

MLflow experiments scoped by user:
```
amortized/shiv/sdg/ticket-classifier
amortized/shiv/training/ticket-classifier
```

Kueue: single LocalQueue shared by all users. Fair sharing via Kueue's built-in mechanisms.

**Good enough for**: single-team deployments, demos, development clusters.

**v2: Namespace Per Tenant**

Full isolation following eval-hub's pattern:

```
Namespace: team-alpha
  ├── amortized-server (or shared central server with tenant routing)
  ├── MLflow (workspace: team-alpha)
  ├── Kueue LocalQueue → ClusterQueue (team-alpha quota)
  ├── Data Connection (team-alpha S3 bucket)
  ├── RBAC (team-alpha ServiceAccount + RoleBindings)
  └── Jobs (only team-alpha's jobs)

Namespace: team-beta
  ├── (same structure, isolated)
```

**Tenant routing**: `X-Tenant` header (eval-hub pattern) or derive from OpenShift namespace. All SQL queries scoped by tenant. All K8s API calls scoped by namespace.

**MLflow isolation**: MLflow Workspaces (MLflow 3.x feature) — each tenant gets a workspace with its own experiments, models, and permissions.

**Kueue isolation**: Per-tenant LocalQueue with separate GPU quotas:
```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: amortized-gpu-queue
  namespace: team-alpha    # tenant-scoped
spec:
  clusterQueue: team-alpha-cluster-queue   # tenant-scoped quota
```

**Auto-provisioning** (eval-hub pattern): Label a namespace, controller provisions ServiceAccount, RoleBindings, LocalQueue, MLflow workspace automatically.

#### What Each Service Needs for Multi-tenancy

| Service | v1 (user tags) | v2 (namespace isolation) |
|---|---|---|
| **Amortized API** | `user_id` column in job table, filter by user | Namespace-scoped API, `X-Tenant` header |
| **MLflow** | Experiments named `amortized/{user}/...` | MLflow Workspaces per tenant |
| **Kueue** | Shared LocalQueue, fair sharing | Per-tenant LocalQueue + ClusterQueue |
| **K8s Jobs** | Labels: `amortized/user=shiv` | Jobs created in tenant namespace |
| **Studio** | User identity from OAuth token | Tenant selector or auto-detect from namespace |
| **OpenCode (Morty)** | Single instance, user context from session | Per-tenant or shared with tenant routing |

#### What Amortized Does NOT Build (Auth)

- No custom login page (OpenShift OAuth)
- No custom user database (OpenShift identity provider)
- No custom session management (OAuth tokens)
- No custom password hashing/storage (OpenShift handles it)
- No custom RBAC engine (K8s RBAC)
- No custom namespace provisioning for v1 (single namespace)

#### What Amortized Builds (Auth)

- **Token validation middleware**: `TokenReview` on incoming requests to extract user identity
- **User scoping**: filter job queries by `user_id` (v1)
- **Job ownership**: tag K8s Jobs and MLflow runs with submitting user
- **Studio OAuth integration**: redirect to OpenShift login, store token, pass in API calls

#### Reference: How Eval-Hub Does It

| Aspect | eval-hub | amortized (v1) | amortized (v2) |
|---|---|---|---|
| Auth | OpenShift ServiceAccount tokens | OpenShift OAuth user tokens | Same |
| Tenant ID | `X-Tenant` HTTP header | `user_id` from TokenReview | `X-Tenant` or namespace |
| SQL scoping | `WithTenant()` builder on all queries | `WHERE user_id = ?` | `WHERE tenant_id = ?` |
| Namespace | Label-driven auto-discovery | Single shared namespace | Per-tenant with auto-provisioning |
| RBAC | Per-tenant ServiceAccount + RoleBindings | OpenShift user roles | Per-tenant roles |
| MLflow | Workspace per tenant | User-scoped experiment names | MLflow Workspaces |

Sources:
- [OpenShift OIDC configuration](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.2/html/managing_openshift_ai/configuring-external-oidc-provider_managing-rhoai)
- [TokenReview and SubjectAccessReview](https://oneuptime.com/blog/post/2026-02-09-tokenreview-subjectaccessreview-apis/view)
- [RHOAI Dashboard configuration](https://ai-on-openshift.io/odh-rhoai/configuration/)
- [OpenShift multi-tenant isolation](https://ones.com/blog/mastering-openshift-multi-tenant-isolation/)
- [Kueue on OpenShift](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html/ai_workloads/red-hat-build-of-kueue)
