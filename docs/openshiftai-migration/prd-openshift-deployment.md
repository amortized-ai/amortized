# PRD: Deploy Amortized on OpenShift AI / ROSA

## Problem Statement

Amortized currently runs only as a local development tool -- a Python FastAPI server on a laptop that dispatches training jobs to GPU nodes via SSH. It cannot be deployed on Kubernetes or OpenShift, which means:

- Platform engineers at enterprises cannot install amortized for their data science teams
- There is no production deployment story (no Helm chart, no container images, no K8s manifests)
- The KubernetesBackend is a stub (all methods raise NotImplementedError)
- Studio has no production build (no Dockerfile, no nginx config)
- GPU nodes must be SSH-accessible -- K8s-managed GPUs are unusable
- No integration with OpenShift AI's existing services (MLflow, KServe, Model Registry)
- SQLite database and local filesystem storage don't survive pod restarts

Amortized will eventually be a Red Hat product. It needs to run natively on OpenShift AI / ROSA (Red Hat OpenShift Service on AWS) and integrate with the ML infrastructure that enterprises already have deployed.

## Solution

Deploy amortized as a thin orchestration layer on OpenShift AI that owns what OpenShift AI lacks (SDG, agent chat, recipes, LLM-as-judge eval) and delegates infrastructure to what OpenShift AI already provides (MLflow, KServe, Kueue, S3, OAuth).

The deployment consists of:
- **amortized-server**: FastAPI API server running as a Deployment with a PVC for SQLite + artifacts
- **amortized-studio**: React SPA served by nginx with reverse proxy to the server, exposed via OpenShift Route
- **Job pods**: Training, SDG, and eval jobs dispatched as K8s Jobs in a dedicated namespace with GPU access

Target cluster: ROSA on AWS with RHOAI 3.3.1, 4x NVIDIA L40S GPUs (48GB each), KServe + Model Registry + Training Operator already running.

```
                    Amortized (thin layer)
 ┌──────────┐ ┌──────┐ ┌───────┐ ┌───────┐
 │  Studio   │ │Agent │ │asynth │ │Judges │
 │  (React)  │ │(chat)│ │(SDG)  │ │(eval) │
 └────┬──────┘ └──┬───┘ └───┬───┘ └───┬───┘
 ┌────▼───────────▼─────────▼─────────▼───┐
 │  Amortized API (FastAPI)               │
 └────────────────┬───────────────────────┘
                  │ dispatches K8s Jobs
                  ▼
          OpenShift AI (infrastructure)
 KServe │ Model Registry │ Training Op │ S3
                              (Kubeflow Hub)
```

## User Stories

### Platform Engineer (installs and operates amortized)

1. As a platform engineer, I want to deploy amortized on my ROSA cluster with a single `oc apply` or `helm install` command, so that my data science team can start building task models without manual infrastructure setup.
2. As a platform engineer, I want amortized to use my cluster's existing GPU nodes (L40S/A100/H100) for training jobs, so that I don't need separate SSH-accessible GPU servers.
3. As a platform engineer, I want training/SDG/eval jobs to run in an isolated namespace with resource quotas, so that amortized jobs don't interfere with other workloads on the cluster.
4. As a platform engineer, I want the amortized server to persist its state (job history, artifacts, conversations) across pod restarts via a PVC, so that I don't lose data during upgrades or node maintenance.
5. As a platform engineer, I want to configure amortized via K8s ConfigMaps and Secrets (not a local YAML file), so that I can manage configuration through standard K8s tooling and GitOps.
6. As a platform engineer, I want amortized to expose health endpoints (`/healthz`, `/readyz`) compatible with K8s probes, so that the cluster can automatically restart unhealthy pods.
7. As a platform engineer, I want Studio to be accessible via an OpenShift Route with TLS termination, so that data scientists can reach it via a browser without port-forwarding.
8. As a platform engineer, I want to use container images from my organization's internal registry (Quay, ECR) instead of ghcr.io, so that I comply with corporate security policies.
9. As a platform engineer, I want the RBAC to be minimal -- amortized's ServiceAccount only needs permission to create/watch/delete Jobs and read Pods/Logs in the jobs namespace, so that the blast radius is contained.
10. As a platform engineer, I want to configure the LLM provider API key (OpenAI, Anthropic) via a K8s Secret that gets injected into the server and job pods, so that keys never appear in ConfigMaps or logs.

### Data Scientist (uses amortized via Studio)

11. As a data scientist, I want to open Studio in my browser and start building a task model immediately after the platform engineer installs amortized, without any local setup.
12. As a data scientist, I want the AI agent to guide me through the full workflow (generate data, train, evaluate, deploy) via chat with clickable option cards, so that I don't need to know K8s, TRL, or asynth internals.
13. As a data scientist, I want to see my training job's progress (loss curve, training step, estimated time remaining) in real-time in Studio, so that I know if training is converging.
14. As a data scientist, I want SDG jobs to run without requiring a GPU (they only make API calls to an LLM provider), so that GPU resources are reserved for training.
15. As a data scientist, I want to submit a training job from a recipe and have it automatically schedule on a GPU node, so that I don't need to manage compute resources.
16. As a data scientist, I want to see all my artifacts (datasets, model weights, eval results, logs) in Studio, so that I can track what was generated and download results.
17. As a data scientist, I want to cancel a running training job from Studio, so that I can stop a misconfigured run without asking the platform engineer.
18. As a data scientist, I want training metrics (loss, learning rate, gradient norm) to stream in real-time as the job runs, so that I can detect problems early.

### Agent Runner (Claude Code / MCP client)

19. As a Claude Code user, I want to connect to amortized's MCP server running on the cluster and submit jobs via natural language, so that I can use amortized without opening Studio.
20. As an MCP client, I want the same API endpoints to work whether amortized runs locally or on K8s, so that my tools don't need environment-specific logic.

### CI/CD Integration

21. As a CI pipeline, I want to submit a training job to amortized via its REST API and poll for completion, so that I can automate model training as part of a release pipeline.
22. As a CI pipeline, I want to download trained model artifacts via the API after a job succeeds, so that I can push them to a model registry or deployment target.

## Implementation Decisions

### Module 1: KubernetesBackend

Replace the stub `backends/kubernetes.py` with a working implementation of the `ComputeBackend` protocol.

- Uses `kubernetes_asyncio` library with `load_incluster_config()` for ServiceAccount auth
- Creates K8s `batch/v1 Job` resources in a configurable namespace (default: `amortized-jobs`)
- GPU requests via `nvidia.com/gpu` resource limits in the pod spec
- Node selection via `nodeSelector: {"nvidia.com/gpu.present": "true"}`
- Status polling via `read_namespaced_job` -- maps K8s job status (succeeded/failed/active) to `BackendStatus`
- Log streaming via `watch.Watch().stream(read_namespaced_pod_log, follow=True)`
- Cancellation via `delete_namespaced_job` with `propagation_policy=Foreground`
- Cleanup deletes the completed K8s Job resource
- Events URL injected as env var: `AMORTIZED_EVENTS_URL=http://amortized-server.amortized.svc:{port}/api/v1/events/ingest`
- Job config injected as `AMORTIZED_CONFIG_JSON` env var (existing pattern)
- Container image registry is configurable via `AMORTIZED_IMAGE_REGISTRY` env var

### Module 2: Server Containerization

Production Dockerfile for the amortized server.

- Base image: `python:3.12-slim`
- Installs `./server` package via pip
- Copies `recipes/`, `examples/`, `containers/` directories into the image
- Runs as non-root user (UID 1001) for OpenShift SCC compatibility
- PVC mount at `/data` for SQLite DB + artifact storage
- Env var configuration: `AMORTIZED_DB_PATH`, `AMORTIZED_STORAGE_PATH`, `AMORTIZED_RECIPES_PATH`
- Entrypoint: `uvicorn amortized.api.app:create_app --factory --host 0.0.0.0 --port 9400`

### Module 3: Studio Containerization

Production Dockerfile for the React frontend.

- Multi-stage build: `node:22-alpine` for Vite build, `nginxinc/nginx-unprivileged:alpine` for serving
- nginx template with `envsubst` for runtime-configurable backend URL
- SPA routing: `try_files $uri $uri/ /index.html`
- Reverse proxy: `/api/` to backend with WebSocket upgrade support
- SSE support: `proxy_read_timeout 300s` for long-lived chat streaming connections
- Non-root nginx (port 8080) for OpenShift SCC compatibility
- Build with default empty `VITE_API_URL` so all API calls use relative paths

### Module 4: K8s Manifests

Kubernetes/OpenShift resource definitions for the full deployment.

- **Namespaces**: `amortized` (control plane) and `amortized-jobs` (compute)
- **Deployments**: `amortized-server` (1 replica) and `amortized-studio` (1 replica)
- **Services**: ClusterIP for both server and studio
- **Route**: OpenShift Route for studio with TLS edge termination
- **PVC**: 50Gi gp3-csi volume for server data
- **ServiceAccount**: `amortized-server` in `amortized` namespace
- **RBAC**: Role + RoleBinding granting the ServiceAccount permission to manage Jobs and read Pods/Logs in `amortized-jobs` namespace
- **Secret**: Template for API keys (applied separately, not committed to git)
- **ConfigMap**: Server configuration (compute backend type, namespace, image registry)

### Module 5: Backend Configuration via Environment

Replace file-based `~/.amortized/config.yaml` dependency with env var support.

- When `AMORTIZED_COMPUTE_BACKEND=kubernetes` is set, the app startup creates a `KubernetesBackend` instead of loading SSH backends from config.yaml
- `AMORTIZED_COMPUTE_NAMESPACE` -- namespace where K8s Jobs are created
- `AMORTIZED_IMAGE_REGISTRY` -- prefix for container images (default: `ghcr.io/amortized-ai`)
- The `config.yaml` path still works for local development -- env vars take precedence
- LLM API keys injected from K8s Secret via standard `valueFrom.secretKeyRef`

### Module 6: asynth Container Entrypoint

Add a working entrypoint to the asynth container so it can run as a K8s Job.

- Python script that reads config from `AMORTIZED_CONFIG_JSON` env var (or mounted YAML file)
- Calls `SynthesisConfig.from_dict()` then `synthesize()`
- Writes output to the work directory
- Emits events (progress, completion) to `AMORTIZED_EVENTS_URL`
- Uses existing `containers/shared/context.py` RunContext pattern for heartbeat and event emission

### Module 7: Container Image CI

GitHub Actions workflows to build and push all container images.

- Trigger on release or manual dispatch
- Build and push to `ghcr.io/amortized-ai/`: `amortized`, `studio`, `training`, `asynth`
- Multi-arch build (amd64) for OpenShift compatibility
- Images must be public (or pull secrets configured in the K8s manifests)

### Architectural Decisions

- **SQLite with PVC for Phase 1**: Single-replica server with SQLite is acceptable for initial deployment. PostgreSQL migration deferred to Phase 2 when multi-replica is needed.
- **K8s Jobs, not PyTorchJobs**: Use plain `batch/v1 Job` instead of Kubeflow PyTorchJob for simplicity. Training Operator's PyTorchJob adds complexity without benefit for single-node LoRA SFT. Can migrate later for multi-node distributed training.
- **In-process scheduler**: Keep the scheduler as an asyncio task inside the API server process. Extract to a separate worker only when scaling beyond single replica.
- **Event ingest via HTTP**: Job pods POST events to the server's `/api/v1/events/ingest` endpoint via K8s Service DNS. No sidecar containers or message queues needed.
- **No MLflow in Phase 1**: Skip MLflow in Phase 1 because Model Registry (Kubeflow Hub, already deployed as `default-modelregistry`) handles model versioning. MLflow is only needed later for experiment tracking (loss curves, hyperparams, run comparison). Model versioning via Model Registry is Phase 1; MLflow experiment tracking is Phase 2.
- **No OpenShift OAuth in Phase 1**: Use bearer token auth (existing). OAuth proxy sidecar is Phase 2.
- **Amortized owns its data**: The server's SQLite DB and artifact files live on a PVC mounted at `/data`. This is the source of truth for Phase 1. In Phase 2, MLflow becomes the experiment/artifact backend.

## Testing Decisions

Good tests verify external behavior -- submit a job, check it transitions through states, verify artifacts are registered. Don't test K8s client internals or nginx config parsing.

### What to test

**KubernetesBackend** (integration test):
- Submit a job: verify K8s Job resource is created in the correct namespace with correct image, env vars, GPU limits
- Poll status: verify state mapping (active to running, succeeded to succeeded, failed to failed)
- Cancel: verify K8s Job is deleted
- Logs: verify log lines are streamed from the pod
- Use `kubernetes_asyncio` mock or a test namespace on the ROSA cluster

**Server Dockerfile** (build test):
- `docker build` succeeds
- Container starts and `/api/v1/health` returns 200
- Recipes are loadable inside the container

**Studio Dockerfile** (build test):
- `docker build` succeeds
- nginx serves index.html at `/`
- nginx proxies `/api/v1/health` to the backend (requires backend to be running)

**End-to-end on ROSA** (manual acceptance test):
- Deploy all manifests: Studio loads in browser
- Submit SDG job via Studio: K8s Job created in `amortized-jobs`, pod runs (CPU, no GPU), output artifact registered
- Submit training job via Studio: K8s Job with GPU request, scheduled on L40S node, training metrics stream, model artifact registered
- Cancel a running job: K8s Job deleted, job status updated in Studio

### Prior art
- Existing tests in `tests/` use `pytest-asyncio` with the `LocalStubBackend` for unit tests
- E2E tests in `studio/e2e/` use Playwright + MSW mock handlers
- The KubernetesBackend tests should follow the same pattern as SSH backend tests but with mocked K8s API

## Out of Scope

The following are explicitly deferred to Phase 2:

- **PostgreSQL migration** -- SQLite with PVC is sufficient for single-replica Phase 1
- **MLflow experiment tracking** -- `report_to: mlflow`, loss curves, run comparison (Phase 2)
- **Model Registry integration** -- register trained models in Kubeflow Hub via `model_registry` Python client (Phase 1 scope)
- **KServe model serving** -- creating InferenceService CRs for model deployment. Target RawDeployment mode (Serverless deprecated in RHOAI 3.3). RHOAI 3.4 introduces LLMInferenceService CRDs.
- **Kueue GPU scheduling** -- fair-sharing and quotas (direct K8s Jobs with nodeSelector for now)
- **S3 artifact storage** -- PVC storage is sufficient for Phase 1
- **OpenShift OAuth** -- SSO via oauth-proxy sidecar
- **Auto-discovery of RHOAI services** -- detecting MLflow URI, S3 config from Data Science Project secrets
- **Multi-replica server deployment** -- requires PostgreSQL and persistent job handle storage
- **Helm chart** -- raw K8s manifests for Phase 1, Helm chart for Phase 2 when parameterization is needed
- **Retry framework** -- failure classification, checkpoint-based resume
- **Multi-tenancy / project scoping** -- single-tenant for Phase 1

## Further Notes

### GPU Capacity on Target Cluster

4x NVIDIA L40S (48GB VRAM each). Training capacity per GPU:
- Qwen3-0.6B LoRA: ~8GB, fits easily
- Qwen3-4B LoRA: ~20GB, fits
- Qwen 2.5 7B QLoRA: ~16GB with 4-bit, fits
- Llama 3.1 8B LoRA: ~24GB, fits
- Llama 3.1 8B full SFT: ~40GB, tight fit

### Container Images Needed

| Image | Base | GPU | Purpose |
|---|---|---|---|
| `amortized` | python:3.12-slim | No | API server |
| `studio` | nginx-unprivileged:alpine | No | Frontend |
| `training` | nvidia/cuda:12.4.1 | Yes | LoRA SFT, DPO, GRPO via TRL |
| `asynth` | python:3.11-slim | No | Synthetic data generation |
| `eval` | python:3.12-slim | No | LLM-as-judge evaluation |

### OpenShift-Specific Considerations

- All containers must run as non-root (OpenShift default SCC `restricted-v2`)
- Use `nginxinc/nginx-unprivileged` instead of standard `nginx` image
- SecurityContext: no `privileged`, no `hostNetwork`, no `hostPID`
- GPU access via NVIDIA GPU Operator (already installed on the cluster via `nvidia-gpu-operator` namespace)
- Routes use TLS edge termination (OpenShift default)

### Superseded Sections

> **Note**: This PRD predates the RHOAI integration research (2026-06-19). The companion PRD `prd-rhoai-integration.md` supersedes this document for RHOAI-specific integration decisions (Model Registry, KServe, MLflow scoping). This PRD remains valid for the K8s deployment mechanics (Dockerfiles, manifests, KubernetesBackend).

### Related Documentation

- `docs/rosa-implementation-plan.md` -- detailed implementation plan with code snippets
- `docs/openshift-ai-integration.md` -- full OpenShift AI integration strategy
- `docs/rosa-cluster-inventory.md` -- cluster scan results
- `docs/compute-backend-upgrade.md` -- compute backend protocol research
- `docs/mlflow-integration-plan.md` -- Phase 2 MLflow integration plan
- `docs/agent-architecture.md` -- agent architecture decisions
