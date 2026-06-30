# Amortized

## What We're Building

A control plane for building task models on OpenShift. The user describes a task their AI agent does with a frontier model (classification, extraction, routing, summarization), and amortized produces a small fine-tuned model that does it cheaper, faster, and on their infrastructure.

Amortized is a **thin orchestration layer**. It translates user intent into tool-native YAML configs, dispatches K8s Jobs/Deployments, and tracks job lifecycle. Everything else is delegated: MLflow for artifacts/lineage, S3 for storage, K8s for compute, TRL/asynth/vLLM for ML logic.

## Dev Commands

```bash
uv pip install -e '.[dev]'    # install
amortized up                   # start server on :8000
amortized config               # configure compute backend
ruff check src/ tests/         # lint
ruff format src/ tests/        # format
mypy src/                      # type check
pytest tests/ -x -q            # test
```

## Architecture

3 job types (v1), each dispatched as a K8s Job, configured via YAML:

- Training: `ghcr.io/amortized-ai/training:latest` — `thub <algo> --config config.yaml` (training-hub) or `trl <algo> --config config.yaml` (TRL)
- SDG: `ghcr.io/amortized-ai/asynth:latest` — `asynth synthesize --config config.yaml`
- Eval: `ghcr.io/amortized-ai/asynth:latest` — `asynth judge --config config.yaml`

Config-only, no generated Python scripts. Single code path for all backends (K8s, SSH, local).

### Compute Backends

- **Kubernetes** (`backends/kubernetes.py`) — K8s Jobs for training/SDG/eval, Deployments for serve. ConfigMaps for config delivery, init containers for S3 data download.
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

### OpenShift Deployment

```
k8s/base/    ← amortized only (server, studio, RBAC, configmap)
k8s/dev/     ← MinIO + MLflow for development/testing
```

Amortized plugs into existing S3 and MLflow (AD-1). Platform engineer provides endpoints.

## Key Patterns

- **Config translation**: `_trl_config_yaml()`, `_serve_config_yaml()`, `_build_synth_config()`, `_eval_config_yaml()` in `worker.py`
- **MLflow run tracking**: `mlflow_run_id` stored on job record, resolved for serve jobs via `_resolve_mlflow_artifact_uri()`
- **Serve from training**: `training_job_id` in serve config → worker resolves base model + adapter from MLflow
- **Init containers**: S3 data download (`_s3_data_path`) and model download (`_s3_model_path`) via `aws s3 cp/sync`
- **Judge templates**: loaded from `templates/eval/` by `core/judge_templates.py`
- **Recipes**: loaded from `templates/` and `examples/` via `core/recipes.py`, support `extends:` for inheritance
- **Credentials**: API keys stored encrypted in DB, injected as per-job K8s Secrets at dispatch time

## Architecture Decisions

See `docs/openshiftai-migration/architecture-decisions.md` for the full list:

- AD-1: Plug into infrastructure, don't bundle it
- AD-2: Single code path, no branching
- AD-3: MLflow is the artifact store
- AD-4: mlflow_run_id on jobs
- AD-5: TRL 1.5.0 with thin custom image
- AD-6: Polling for now, push-based later
- AD-7: No serve monitor on K8s
- AD-8: Reuse K8s API client

## Code Style

- Python 3.12+, strict mypy, ruff enforced
- `src/` layout — all code under `src/amortized/`
- FastAPI with pydantic-settings for config
- All API endpoints under `/api/v1/`
- SQLite for persistence (no ORM, raw aiosqlite)
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
