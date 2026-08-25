# Amortized

## What We're Building

A control plane for building task models on OpenShift. The user describes a task their AI agent does with a frontier model (classification, extraction, routing, summarization), and amortized produces a small fine-tuned model that does it cheaper, faster, and on their infrastructure.

Amortized is a **thin orchestration layer**. It translates user intent into tool-native YAML configs, dispatches K8s Jobs/Deployments, and tracks job lifecycle. Everything else is delegated: MLflow for artifacts/lineage, S3 for storage, K8s for compute, TRL/asynth/vLLM for ML logic.

## Dev Commands

```bash
# Backend
uv pip install -e '.[dev]'    # install
amortized up                   # start server on :8000
amortized config               # configure compute backend
ruff check src/ tests/         # lint
ruff format src/ tests/        # format
mypy src/                      # type check
pytest tests/ -x -q            # test

# Studio (frontend)
cd studio && npm install       # install
cd studio && npm run dev       # dev server on :5173
cd studio && npm run lint      # lint
cd studio && npm run typecheck # type check
cd studio && npm test          # test
cd studio && npm run build     # production build
```

## Architecture

2 job types (v1), each dispatched as a K8s Job, configured via YAML:

- Training: `ghcr.io/amortized-ai/training:latest` — `thub <algo> --config config.yaml` (training-hub) or `trl <algo> --config config.yaml` (TRL)
- SDG: `ghcr.io/amortized-ai/data-designer:latest` — Data Designer microservice for synthetic data generation

Config-only, no generated Python scripts. Single code path for all backends (K8s, SSH, local).

### Compute Backends

- **Kubernetes** (`backends/kubernetes.py`) — K8s Jobs for training/SDG. ConfigMaps for config delivery, init containers for S3 data download.
- **SSH** (`backends/ssh.py`) — podman/docker on remote GPU nodes. Same YAML configs, different delivery mechanism.
- **Local** (`backends/local.py`) — subprocess on the local machine. For development.

### Artifact Flow (MLflow)

All artifacts flow through MLflow (AD-3). Amortized does not manage artifacts itself.

```
SDG → mlflow.log_artifact(data.jsonl) → MLflow → S3
Training → report_to: mlflow (TRL auto-logs) → MLflow → S3
Serve → training_job_id → resolve MLflow run → init container downloads adapter from S3
```

Amortized stores `mlflow_run_id` per job. Everything else — storage, versioning, lineage — is MLflow's responsibility.

### K8s Deployment

```
studio/                ← React/Vite frontend (Amortized Studio)
k8s/base/              ← kustomize base (server, studio, RBAC, configmap)
k8s/services/          ← MLflow, MinIO, PostgreSQL service definitions
k8s/overlays/dev/      ← single-user development overlay
k8s/overlays/rosa/     ← OpenShift/ROSA production overlay
```

Deploy with `kubectl apply -k k8s/overlays/dev` or `make deploy-dev`.

## Key Patterns

- **Config translation**: `_training_hub_config_yaml()` and `_build_synth_config()` in `worker.py`
- **MLflow run tracking**: `mlflow_run_id` stored on job record, resolved via `_resolve_mlflow_artifact_uri()`
- **Parent job chaining**: `parent_job_id` links SDG→Training; worker resolves upstream MLflow artifacts
- **Init containers**: S3 data download via `aws s3 cp/sync` for training data from MLflow artifact store
- **Recipes**: starter templates loaded from `agents/*/skills/` reference payloads via `core/recipes.py`
- **Credentials**: API keys stripped from config before DB storage, injected as per-job K8s Secrets

## Architecture Decisions

Key decisions for v1:
- AD-1: Plug into infrastructure, don't bundle it
- AD-2: Single code path, no branching
- AD-3: MLflow is the artifact store (no direct S3 writes)
- AD-4: Polling for now, push-based later
- AD-11: No serve jobs (MaaS handles serving)

## Code Style

- Python 3.12+, strict mypy, ruff enforced
- `src/` layout — all code under `src/amortized/`
- FastAPI with pydantic-settings for config
- All API endpoints under `/api/v1/`
- PostgreSQL for persistence (async via asyncpg, Alembic migrations)
- No comments unless the WHY is non-obvious
- uv for Docker builds (not pip)

## Git

- Pre-commit hook regenerates `openapi/v1.json` if API files change
- `git config core.hooksPath .githooks`

## Gotchas

- `yaml.safe_load()` parses `2e-4` as string, not float — use `0.0002` in YAML configs
- vLLM image entrypoint is `["vllm", "serve"]` — container CMD should be just `["--config", "path"]`, not `["vllm", "serve", "--config"]`
- TRL field names differ from common conventions: `num_train_epochs` not `num_epochs`, `per_device_train_batch_size` not `batch_size`, `max_length` not `max_seq_len`
- SDG output must use `messages` column (not `conversation`) for TRL compatibility
- `report_to` defaults to `mlflow` when `mlflow_tracking_uri` is configured, `none` otherwise
- TRL 1.5.1+ has a `kernels` import bug — pin to 1.5.0
- `MLFLOW_S3_ENDPOINT_URL` (not `AWS_S3_ENDPOINT_URL`) is what MLflow reads for MinIO
- `imagePullPolicy: Always` on K8s job containers — otherwise cached stale images cause silent failures
- asynth passes S3 `endpoint_url` via `client_kwargs` (not env var) for compatibility with old s3fs versions
