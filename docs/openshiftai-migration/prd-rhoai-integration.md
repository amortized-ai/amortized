# PRD: Amortized as a Thin Layer on OpenShift AI

## Problem Statement

Amortized currently rebuilds infrastructure that OpenShift AI already provides: model versioning, model serving, GPU scheduling, artifact storage, authentication. This creates two problems:

1. **Wasted engineering effort** — building a model registry, vLLM orchestration, S3 storage layer, and auth system that RHOAI already packages.
2. **Integration friction** — enterprises running RHOAI expect new ML tools to plug into their existing MLflow, KServe, Model Registry, and monitoring infrastructure, not bring their own.

The ROSA cluster already has KServe (GA), Model Registry (GA), Training Operator (GA), and KubeRay running. MLflow and Kueue can be added. Amortized should use all of them.

## Solution

Reposition amortized as a **thin orchestration layer** that owns what RHOAI lacks — synthetic data generation (asynth), agent-guided workflows (Studio chat), task-model recipes, and LLM-as-judge evaluation — and delegates infrastructure to RHOAI services.

```
 Amortized (unique value — RHOAI can't do this)
 ┌──────────┐ ┌──────┐ ┌───────┐ ┌───────┐
 │  Studio   │ │Agent │ │asynth │ │Judges │
 │  (React)  │ │(chat)│ │(SDG)  │ │(eval) │
 └────┬──────┘ └──┬───┘ └───┬───┘ └───┬───┘
 ┌────▼───────────▼─────────▼─────────▼───┐
 │  Amortized API (FastAPI)               │
 │  - Job orchestration                   │
 │  - Recipe resolution                   │
 │  - Agent tools                         │
 └────────────────┬───────────────────────┘
                  │ generates CRs + API calls
 ┌────────────────▼───────────────────────┐
 │  OpenShift AI 3.3.1 (infrastructure)   │
 │  ┌────────────┐ ┌───────────────────┐  │
 │  │ KServe     │ │ Model Registry    │  │
 │  │ (serving)  │ │ (Kubeflow Hub)    │  │
 │  └────────────┘ └───────────────────┘  │
 │  ┌────────────┐ ┌───────┐ ┌────────┐  │
 │  │ Training   │ │ Kueue │ │  S3    │  │
 │  │ Operator   │ │ (GPU) │ │(AWS)   │  │
 │  └────────────┘ └───────┘ └────────┘  │
 │  ┌────────────┐ ┌───────────────────┐  │
 │  │ OpenShift  │ │  MLflow           │  │
 │  │ OAuth      │ │  (experiment)     │  │
 │  └────────────┘ └───────────────────┘  │
 └────────────────────────────────────────┘
```

## What Amortized Owns vs Delegates

### Amortized owns (unique value)

| Component | Why RHOAI Can't Do This |
|---|---|
| **asynth (SDG)** | No synthetic data generation in RHOAI |
| **Agent chat** | No guided workflow with option cards, action confirmation |
| **Studio UI** | Task-model-focused dashboard, not generic ML dashboard |
| **Recipes** | Pre-built end-to-end configs for common task types |
| **LLM-as-judge eval** | No evaluation with custom criteria |
| **Task-model workflow** | Generate → train → eval → deploy as one guided flow |

### RHOAI provides (don't rebuild)

| What | RHOAI Service | How Amortized Uses It |
|---|---|---|
| Model versioning | Model Registry (Kubeflow Hub) | `model_registry` Python client to register models |
| Model serving | KServe + vLLM ServingRuntime | Create InferenceService CR with lineage labels |
| Training dispatch | Training Operator (PyTorchJob) | Create PyTorchJob CR with GPU resource requests |
| GPU scheduling | Kueue | Add `kueue.x-k8s.io/queue-name` label to jobs |
| Artifact storage | AWS S3 | S3 paths in Model Registry, training configs |
| Experiment tracking | MLflow (when deployed) | `report_to: mlflow` in TRL TrainingArguments |
| Authentication | OpenShift OAuth | oauth-proxy sidecar on Studio |
| Monitoring | Prometheus + Grafana | Expose `/metrics` endpoint |

### What This Eliminates From the Backlog

| Originally Planned | Status | Why |
|---|---|---|
| PostgreSQL migration | **Eliminated** | Model Registry has its own MySQL. Amortized only needs SQLite for job queue + conversations. |
| Custom artifact storage system | **Eliminated** | Artifacts stored in S3, referenced by Model Registry metadata. |
| Custom model registry | **Eliminated** | Kubeflow Model Registry already running as `default-modelregistry`. |
| MLflow Model Registry integration | **Replaced** | Use Kubeflow Model Registry (different system). MLflow only for experiment tracking. |
| vLLM container orchestration | **Eliminated** | KServe InferenceService handles vLLM lifecycle, scaling, routing. |
| Auth system / RBAC | **Eliminated** | OpenShift OAuth for identity. OpenShift RBAC for permissions. |
| GPU quota enforcement | **Eliminated** | Kueue handles per-team GPU quotas. |
| Custom S3 storage backend code | **Simplified** | S3 is just a bucket path. Model Registry stores the reference. |
| Helm chart PostgreSQL subchart | **Eliminated** | No database to manage beyond amortized's lightweight SQLite. |

## User Stories

### Platform Engineer (installs amortized)

1. Deploy amortized with `oc apply -f k8s/` or `helm install` — auto-discovers RHOAI services (Model Registry, KServe, Training Operator).
2. Jobs run in `amortized-jobs` namespace with RBAC isolation.
3. Trained models appear in RHOAI Model Registry, browsable via RHOAI dashboard.
4. Deployed models appear as KServe InferenceServices, visible in OpenShift console.
5. GPU quotas managed by Kueue, not amortized config.

### Data Scientist (uses Studio)

1. Open Studio → SSO login (OpenShift OAuth).
2. Chat: "Build me a ticket classifier" → agent guides through options.
3. SDG runs as K8s Job → data stored in S3.
4. Training runs as PyTorchJob on L40S GPU → model registered in Model Registry.
5. Deploy → KServe InferenceService created → model served via vLLM.
6. "Open in Model Registry" link in Studio → RHOAI dashboard shows version history.

## Implementation Phases

### Phase 1: Deploy on RHOAI (MVP)

**Goal**: Amortized running on ROSA, submitting jobs to L40S GPUs, registering models in Model Registry.

| # | Work | Effort | Details |
|---|---|---|---|
| 1.1 | **KubernetesBackend** | Medium | Replace stub with working `batch/v1 Job` dispatch. `kubernetes_asyncio`, in-cluster config, GPU resource requests, log streaming, cancellation. |
| 1.2 | **Server Dockerfile** | Small | `python:3.12-slim`, non-root (UID 1001), PVC at `/data` for SQLite + artifacts. |
| 1.3 | **Studio Dockerfile** | Small | Multi-stage build, `nginx-unprivileged:alpine`, SPA fallback, `/api/` reverse proxy, SSE support (`proxy_read_timeout 300s`). |
| 1.4 | **K8s manifests** | Small | Namespaces, Deployments, Services, Route, PVC (50Gi gp3-csi), ServiceAccount, RBAC (create/watch/delete Jobs in amortized-jobs). |
| 1.5 | **Config via env vars** | Small | `AMORTIZED_COMPUTE_BACKEND=kubernetes`, `AMORTIZED_COMPUTE_NAMESPACE`, `AMORTIZED_IMAGE_REGISTRY`. ConfigMap/Secret injection. |
| 1.6 | **S3 artifact storage** | Small | Create AWS S3 bucket. Set `AMORTIZED_S3_BUCKET` env var. Workers write training outputs to S3 instead of local filesystem. |
| 1.7 | **Model Registry integration** | Medium | After training succeeds, call `model_registry.register_model()` with S3 URI. Store `registered_model_id` on amortized job record. Add "Open in Model Registry" link in Studio. |
| 1.8 | **Container images CI** | Small | GitHub Actions to build and push `amortized`, `studio`, `asynth` images to ghcr.io. |

**Acceptance test**: Deploy on ROSA → Studio loads → submit SDG job → submit training job → model appears in Model Registry → deploy via Studio → InferenceService created and serving.

### Phase 2: Integrate RHOAI Services

**Goal**: Use RHOAI services for serving, scheduling, and experiment tracking.

| # | Work | Effort | Depends On |
|---|---|---|---|
| 2.1 | **KServe serving** | Medium | Phase 1. Replace vLLM container dispatch with InferenceService CR creation. Target RawDeployment mode. Support S3 and OCI ModelCar storage URIs. Add `serving.knative.dev/progress-deadline: 30m` annotation. |
| 2.2 | **Kueue GPU scheduling** | Small | Phase 1. Install Kueue on cluster (platform engineer). Add `kueue.x-k8s.io/queue-name` label to training/eval K8s Jobs. |
| 2.3 | **MLflow experiment tracking** | Medium | Phase 1 + MLflow deployed on cluster. Set `report_to: mlflow` in TRL configs. Inject `MLFLOW_TRACKING_URI` env var into training containers. Log SDG params/metrics from asynth runner. |
| 2.4 | **OpenShift OAuth** | Medium | Phase 1. Add oauth-proxy sidecar to Studio deployment. Configure ServiceAccount with redirect URI annotation. |
| 2.5 | **Upgrade to PyTorchJob** | Medium | Phase 1. Replace `batch/v1 Job` with PyTorchJob CR for training. Enables multi-node distributed training. |
| 2.6 | **"Open in MLflow" links** | Small | 2.3. Job detail page → link to MLflow run. Don't rebuild MLflow's comparison UI. |

### Phase 3: Production Polish

| # | Work | Effort | Depends On |
|---|---|---|---|
| 3.1 | **Failure classification** | Small | Phase 1. Parse stderr on failure. Classify OOM, NCCL, config errors. Surface actionable messages in Studio. |
| 3.2 | **Retry framework** | Medium | 3.1. Configurable retry for transient failures (SIGKILL, pod eviction). Don't retry OOM or config errors. Checkpoint resume via `resume_from_checkpoint`. |
| 3.3 | **Helm chart** | Medium | Phase 2. Parameterized Helm chart for `helm install` deployment. Auto-discovery of RHOAI services. |
| 3.4 | **Prometheus metrics** | Small | Phase 1. Expose `/metrics` with job throughput, queue depth, error rates. Cluster Prometheus scrapes automatically. |
| 3.5 | **Structured JSON logs** | Small | Phase 1. Feed into OpenShift's logging stack (Loki/EFK). |

### Phase 4: Scale

| # | Work | When |
|---|---|---|
| 4.1 | **Project scoping** | When multi-team. Maps to K8s namespaces + Model Registry experiments. |
| 4.2 | **Trainer v2 migration** | When Trainer v2 reaches GA (RHOAI 3.4-3.5). Replace PyTorchJob with TrainJob CR. |
| 4.3 | **Pipeline definitions** | When workflow complexity grows. Synth → train → eval as declared flow. |
| 4.4 | **OCI ModelCar for task models** | When S3 dependency is undesirable. Build OCI image post-training, deploy via `oci://` URI. |

## Architectural Decisions

### AD-1: Model Registry, NOT MLflow Model Registry

The RHOAI Model Registry is Kubeflow Hub (v1alpha3 REST API), not MLflow's model registry. Use it for model versioning and deployment lineage. MLflow is only for experiment tracking (loss curves, hyperparameters).

**Rationale**: Model Registry is already deployed (`default-modelregistry`). It's the RHOAI-native way to manage models.

### AD-2: batch/v1 Job first, PyTorchJob later

Start with plain Kubernetes Jobs for training. PyTorchJob adds value only for multi-node distributed training, which amortized doesn't need for single-GPU LoRA SFT.

**Rationale**: Simplest possible integration. PyTorchJob overhead is unjustified for single-node workloads.

### AD-3: RawDeployment, NOT Serverless for KServe

Target RawDeployment (standard K8s Deployments) for model serving. Serverless mode (Knative + Istio) is deprecated in RHOAI 3.3.

**Rationale**: Verified by research — Serverless deprecation confirmed. RHOAI 3.4 introduces new LLMInferenceService CRDs that further move away from Knative.

### AD-4: SQLite on PVC, NOT PostgreSQL

Keep amortized's SQLite database on a PVC. Don't migrate to PostgreSQL.

**Rationale**: Model metadata now lives in Model Registry (MySQL backend). Experiment tracking goes to MLflow (PostgreSQL backend). Amortized's SQLite only stores job queue state and chat conversations — lightweight data that doesn't justify a PostgreSQL dependency.

### AD-5: S3 for all artifacts, PVC only for amortized control plane

Training data, checkpoints, model weights, SDG output, eval results → S3. Only amortized's own SQLite DB and runtime state → PVC.

**Rationale**: S3 survives pod restarts, is shared across namespaces, and is what Model Registry and KServe expect to read from.

### AD-6: Kueue for GPU scheduling, NOT custom queue logic

Don't build GPU queue management. Add one label to jobs and let Kueue handle quotas, fair-sharing, and preemption.

**Rationale**: Kueue integration is a single label per job. The one-time cluster setup (ClusterQueue, ResourceFlavor, LocalQueue) is a platform engineer task, not amortized's concern.

## Open Questions (Require Cluster Verification)

Before implementation begins, run these on the ROSA cluster:

```bash
# 1. What ClusterTrainingRuntimes exist?
oc get clustertrainingruntimes -o name

# 2. Is MLflow deployed anywhere?
oc get csv -A | grep mlflow
oc get pods -A | grep mlflow

# 3. What ServingRuntimes exist?
oc get servingruntimes -A -o custom-columns=NS:.metadata.namespace,NAME:.metadata.name,FORMAT:.spec.supportedModelFormats[*].name

# 4. KServe deployment mode?
oc get configmap inferenceservice-config -n redhat-ods-applications -o yaml | grep deploy

# 5. Does ModelCar work out of the box?
# Try a test InferenceService with oci:// URI

# 6. Model Registry API endpoint?
oc get routes -n rhoai-model-registries
oc get svc -n rhoai-model-registries
```

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Trainer v2 is Tech Preview, may change API | Training job CRs break on upgrade | Start with batch/v1 Job. Migrate to PyTorchJob v1 (GA) before Trainer v2. |
| ClusterTrainingRuntimes may not include TRL stack | Amortized's training images may not work with provided runtimes | Create custom TrainingRuntime with TRL/LoRA dependencies. |
| ModelCar may require ConfigMap patch (unverified) | Serving via OCI ModelCar fails silently | Test on cluster. Fall back to S3 storageUri. |
| MLflow status in RHOAI 3.3 is ambiguous | May need to deploy MLflow manually | Use `oc new-app` with upstream MLflow image as fallback. |
| RHOAI 3.4 introduces LLMInferenceService CRDs | Current InferenceService approach may become legacy | InferenceService will continue working. LLMInferenceService is additive. |

## Success Metrics

| Metric | Phase 1 | Phase 2 |
|---|---|---|
| Install time (platform engineer) | < 15 minutes with `oc apply` | < 5 minutes with `helm install` |
| Time to first model (data scientist) | < 2 hours (SDG → train → deploy) | < 1 hour |
| Infrastructure code owned by amortized | SQLite, S3 client, K8s Job dispatch | Same — new capabilities come from RHOAI |
| Model Registry models | All trained models registered | With version history and lineage |
| Lines of code eliminated | vLLM orchestration, custom artifact tracking | Auth system, GPU scheduling code |

## References

- [RHOAI 3.3 Model Registries](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/pdf/managing_model_registries/Red_Hat_OpenShift_AI_Self-Managed-3.3-Managing_model_registries-en-US.pdf)
- [Kubeflow Hub REST API](https://www.kubeflow.org/docs/components/hub/reference/rest-api/)
- [KServe on RHOAI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.2/html-single/deploying_models/index)
- [ModelCar on RHOAI](https://developers.redhat.com/articles/2025/01/30/build-and-deploy-modelcar-container-openshift-ai)
- [Kueue PyTorchJobs](https://kueue.sigs.k8s.io/docs/tasks/run/kubeflow/pytorchjobs/)
- [RHOAI Trainer v2](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/html/working_with_distributed_workloads/running-kubeflow-trainerv2_distributed-workloads)
- [RHOAI Supported Configs](https://access.redhat.com/articles/rhoai-supported-configs-3.x)
- [InstructLab on OCP](https://github.com/opendatahub-io/ilab-on-ocp)
- [OpenShift OAuth Proxy](https://github.com/openshift/oauth-proxy)
- Deep research: `docs/rhoai-integration-research.md` (this project)
