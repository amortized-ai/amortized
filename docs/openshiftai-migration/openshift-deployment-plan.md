> **SUPERSEDED** — OpenShift deployment is implemented. See `prd-openshift.md` and `architecture-decisions.md`. This doc is kept for historical reference.

# OpenShift/ROSA Deployment Plan

How a platform engineer deploys amortized for a data science team on OpenShift AI / ROSA.

## Target Persona

Sarah, platform engineer at a financial services company. Runs OpenShift AI on ROSA (AWS). Has 4 data scientists who need to build task models. Evaluating amortized.

## Current Gaps

| What a platform engineer needs | What exists | Gap |
|---|---|---|
| Helm chart or Operator | Nothing | No K8s deployment story |
| PostgreSQL backend | SQLite only | Not needed — Model Registry has its own MySQL. SQLite with PVC sufficient for amortized's job queue. |
| S3 artifact storage | Local filesystem default | Pod restarts = artifacts gone |
| Production Studio build | Vite dev server | Can't run dev server in production |
| OAuth/LDAP auth | Bearer token in localStorage | No SSO, no OpenShift OAuth proxy |
| GPU nodes for training | SSH backend only | No K8s Job/Pod dispatch |
| Model Registry integration | No registry client | Use existing `default-modelregistry` on cluster |
| MLflow experiment tracking | `report_to: none` | Can't connect to OpenShift AI's MLflow |
| Container registry config | Hardcoded `ghcr.io/amortized-ai/*` | Can't use internal Quay/ECR |
| Monitoring | None | No Prometheus metrics, no K8s probes |
| Multi-user | Single-user, no auth | Can't share across a team |

## Target Architecture

```
ROSA Cluster (AWS)
├── Namespace: amortized
│   ├── Deployment: amortized-server (FastAPI)
│   │   ├── SQLite (PVC-backed, job queue + conversations)
│   │   ├── S3 artifact store (via ROSA S3 integration)
│   │   └── Config: MLflow URI, Model Registry, compute namespace, auth
│   ├── Deployment: amortized-studio (nginx serving static React build)
│   │   └── Route: studio.apps.cluster.example.com
│   ├── Route: api.apps.cluster.example.com → amortized-server
│   └── ServiceAccount: amortized-job-dispatcher
│       └── RoleBinding: can create Jobs/Pods in amortized-jobs namespace
│
├── Namespace: amortized-jobs (where training/SDG/eval pods run)
│   ├── ResourceQuota: 8 GPUs max
│   ├── LimitRange: per-pod GPU limits
│   └── NetworkPolicy: can reach MLflow + S3, nothing else
│
├── Namespace: redhat-ods-applications (OpenShift AI)
│   ├── MLflow tracking server (experiment tracking, deployed by operator)
│   ├── Model Registry (Kubeflow Hub, `default-modelregistry`)
│   └── S3 buckets for artifacts
│
└── GPU Node Pool
    └── 4x g5.xlarge (NVIDIA A10G) or p4d.24xlarge (A100)
```

## Execution Plan — Ordered by "Platform Engineer Can Deploy"

### Phase 0: Production-Ready Backend (blocks everything)

| # | Work | Why | Effort |
|---|---|---|---|
| 1 | **S3 as default artifact store** — already has `S3Storage` class, make it the default path, test end-to-end | Pod-local filesystem is ephemeral. S3 is the only option on ROSA. | Medium |
| 2 | **Production Studio build** — `vite build` → static files, Dockerfile with nginx, configurable API URL | Can't run Vite dev server in production. | Small |
| 3 | **Health endpoints** — `/healthz` (liveness), `/readyz` (readiness, checks DB + storage) | K8s needs these to manage pod lifecycle. | Small |
| 4 | **Config via env vars** — ensure all settings work via env vars (pydantic-settings already does this, but verify coverage) | K8s ConfigMaps/Secrets inject env vars, not config files. | Small |

### Phase 1: K8s Deployment Story

| # | Work | Why | Effort |
|---|---|---|---|
| 6 | **Helm chart** — deploys server + studio + configures integrations | The standard way to install on K8s/OpenShift. | Medium |
| 7 | **K8s compute backend** — dispatch training/SDG/eval as K8s Jobs in a configurable namespace | Platform engineer's GPUs are in the cluster, not SSH-accessible. | Large |
| 8 | **Container image registry config** — configurable image registry prefix, image pull secrets support | Enterprise clusters use internal registries (Quay, ECR), not ghcr.io. | Small |
| 9 | **Job namespace isolation** — run jobs in a separate namespace with ResourceQuota | Don't let training pods compete with the control plane. | Small |

### Phase 2: OpenShift AI Integration

| # | Work | Why | Effort |
|---|---|---|---|
| 10 | **Model Registry integration** — register trained models via `model_registry` Python client, use existing `default-modelregistry` on cluster | Trained models need a registry for versioning and promotion. | Medium |
| 11 | **MLflow experiment tracking** — `report_to: mlflow`, tracking URI config, SDG logging | OpenShift AI already has MLflow for experiment tracking. Connect to it. | Medium |
| 12 | **OpenShift OAuth proxy** — sidecar container for Studio that authenticates via OpenShift OAuth | Data scientists log in with corporate SSO, not a bearer token. | Medium |
| 13 | **S3 bucket auto-discovery** — read S3 config from OpenShift AI's data science project secrets | Don't make the platform engineer configure S3 twice. | Small |
| 14 | **KServe serving** — create InferenceService CR instead of raw vLLM pods, target RawDeployment mode | Standard serving path on OpenShift AI; integrates with Model Registry. | Medium |
| 15 | **GPU scheduling via Kueue** — respect cluster-level GPU quotas and fair sharing | Shared clusters need fair GPU allocation across teams. | Medium |

### Phase 3: Team Features

| # | Work | Why | Effort |
|---|---|---|---|
| 16 | **Project scoping** (from multi-tenancy-research.md) — `projects` table, scoped resources | Platform engineer has 4 data scientists working on different models. | Medium |
| 17 | **RBAC** — admin (platform eng) vs member (data scientist) vs viewer (stakeholder) | Platform engineer configures compute. Data scientists submit jobs. Stakeholders view results. | Medium |
| 18 | **Audit log** — who did what when | Enterprise compliance requirement. | Small |

### Phase 4: Production Hardening

| # | Work | Why | Effort |
|---|---|---|---|
| 19 | **Retry framework** (from resilience-research.md) — failure classification, configurable retry, checkpoint resume | Training on spot instances needs auto-recovery. | Medium |
| 20 | **Prometheus metrics** — `/metrics` endpoint with GPU utilization, job queue depth, error rates | Platform engineer monitors via Grafana (standard on OpenShift). | Small |
| 21 | **Structured JSON logs** — feed into OpenShift's logging stack (Loki/EFK) | Platform engineer needs centralized logging. | Small |
| 22 | **Backup/restore** — SQLite PVC snapshots, S3 versioning | Production data protection. | Small |

## Day 1 Experience (After All Phases)

### Platform Engineer Installs

```bash
# Add Helm repo
helm repo add amortized https://charts.amortized.ai

# Install on ROSA cluster
helm install amortized amortized/amortized \
  --namespace amortized --create-namespace \
  --set storage.type=s3 \
  --set storage.s3.bucket=amortized-artifacts \
  --set storage.s3.region=us-east-1 \
  --set mlflow.trackingUri=http://mlflow.redhat-ods-applications.svc:5000 \
  --set compute.type=kubernetes \
  --set compute.namespace=amortized-jobs \
  --set compute.gpu.nodeSelector="nvidia.com/gpu=true" \
  --set auth.openshift.enabled=true \
  --set images.registry=quay.io/mycompany
  # Model Registry auto-discovered from RHOAI operator

# Studio available at:
# https://amortized-studio.apps.cluster.example.com
```

### Data Scientist Uses

```
1. Open Studio → SSO login via OpenShift OAuth
2. Chat: "I want to build a ticket classifier"
3. Agent presents option cards → user selects domain, categories, output format
4. Agent proposes SDG job → user clicks [Confirm]
5. SDG runs as K8s Job on GPU node
   → metrics logged to MLflow
   → artifacts stored in S3
6. Agent: "Data looks good. 92% quality score. Ready to train?" → [Confirm]
7. Training runs as K8s Job
   → loss curves streamed to Studio + MLflow
   → checkpoint saved to S3
8. Agent: "Training complete. 94% accuracy on held-out set. Deploy?" → [Confirm]
9. Model registered in Model Registry (Kubeflow Hub)
   → versioned, tagged, ready for promotion
10. Platform engineer sees:
    → All runs in MLflow UI
    → All jobs in Studio
    → GPU utilization in Grafana
    → Audit trail of who trained what
```

## Helm Chart Structure

```
charts/amortized/
├── Chart.yaml
├── values.yaml                    # Default values
├── values-openshift.yaml          # OpenShift-specific overrides
├── templates/
│   ├── deployment-server.yaml     # amortized-server
│   ├── deployment-studio.yaml     # amortized-studio (nginx)
│   ├── service-server.yaml
│   ├── service-studio.yaml
│   ├── route-studio.yaml          # OpenShift Route (or Ingress for vanilla K8s)
│   ├── route-api.yaml
│   ├── configmap.yaml             # Server config via env vars
│   ├── secret.yaml                # API keys, DB credentials
│   ├── serviceaccount.yaml        # For job dispatch
│   ├── role.yaml                  # RBAC for creating Jobs in compute namespace
│   ├── rolebinding.yaml
│   ├── namespace-jobs.yaml        # Optional: create compute namespace
│   ├── resourcequota-jobs.yaml    # Optional: GPU quota in compute namespace
│   ├── _helpers.tpl
│   └── NOTES.txt
```

### Key values.yaml Fields

```yaml
# Server
server:
  image:
    registry: ghcr.io/amortized-ai
    repository: amortized
    tag: latest
  replicas: 1
  resources:
    requests: { cpu: 500m, memory: 512Mi }
    limits: { cpu: 2, memory: 2Gi }

# Studio
studio:
  image:
    registry: ghcr.io/amortized-ai
    repository: studio
    tag: latest
  replicas: 1

# Model Registry
modelRegistry:
  autoDiscover: true       # Auto-discover from RHOAI operator
  # OR manual:
  host: ""                 # e.g., default-modelregistry.redhat-ods-applications.svc
  port: 8080

# Storage
storage:
  type: s3                 # local | s3 | gcs
  s3:
    bucket: amortized-artifacts
    region: us-east-1
    endpoint: ""           # For MinIO
    accessKeySecret: ""    # K8s secret name

# MLflow
mlflow:
  enabled: true
  trackingUri: http://mlflow:5000
  # OR auto-discover from OpenShift AI:
  autoDiscover: true

# Compute
compute:
  type: kubernetes         # kubernetes | ssh
  namespace: amortized-jobs
  createNamespace: true
  gpu:
    nodeSelector: { "nvidia.com/gpu": "true" }
    quota: 8               # Max GPUs for this installation
  images:
    training: ghcr.io/amortized-ai/training:latest
    sdg: ghcr.io/amortized-ai/asynth:latest
    eval: ghcr.io/amortized-ai/asynth:latest
    serve: docker.io/vllm/vllm-openai:latest

# Auth
auth:
  type: token              # token | openshift
  openshift:
    enabled: false
    # Uses oauth-proxy sidecar when enabled

# Monitoring
monitoring:
  prometheus:
    enabled: true
    serviceMonitor: true   # Create ServiceMonitor for Prometheus Operator
```

## Database Strategy

PostgreSQL migration eliminated. Model metadata lives in Model Registry (MySQL). Experiment data lives in MLflow (PostgreSQL). Amortized keeps SQLite for job queue + chat conversations — lightweight data that doesn't justify a separate database. SQLite is backed by a PVC in K8s for persistence across pod restarts.

## References

- OpenShift AI MLflow: https://docs.redhat.com/en/documentation/red_hat_openshift_ai
- Kubeflow Model Registry: https://www.kubeflow.org/docs/components/hub/reference/rest-api/
- RHOAI Model Registries: https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/pdf/managing_model_registries/
- OpenShift OAuth Proxy: https://github.com/openshift/oauth-proxy
- Kueue for GPU scheduling: https://kueue.sigs.k8s.io/
- Related docs: compute-backend-upgrade.md, mlflow-integration-plan.md, multi-tenancy-research.md, resilience-research.md
