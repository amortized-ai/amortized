# Amortized — Behavioral Specification

Version: 1.0.0
Status: Draft
Last updated: 2026-07-28

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119).

---

## 1. System Overview

Amortized is a control plane for building task-specific fine-tuned models on OpenShift. It translates user intent into tool-native YAML configs, dispatches compute jobs (K8s Jobs, SSH containers, or local subprocesses), and tracks job lifecycle. All ML logic is delegated to external tools (TRL, NVIDIA Data Designer, vLLM); all artifact storage is delegated to MLflow backed by S3.

The system comprises two deployable units:

- **Server** — a FastAPI application (`src/amortized/`) exposing a REST API on `:8000`, with an embedded background worker and MCP server.
- **Studio** — a React/Vite single-page application (`studio/`) serving the frontend on `:5173`, proxying API calls to the server.

---

## 2. Domain Model

### 2.1 Job

The central entity. Represents a unit of compute work dispatched to a backend.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `id` | UUID string | PRIMARY KEY, auto-generated | Unique job identifier |
| `type` | `JobType` enum | NOT NULL | `training` or `sdg` |
| `status` | `JobStatus` enum | NOT NULL, default `queued` | Current lifecycle state |
| `config` | JSON dict | NOT NULL, default `{}` | Job-type-specific configuration |
| `recipe` | string | default `""` | Recipe name used to create this job |
| `parent_job_id` | string | default `""` | Links SDG→Training chains |
| `user_id` | string | default `""` | From `X-Forwarded-User` header |
| `k8s_job_name` | string | default `""` | Kubernetes Job resource name |
| `k8s_namespace` | string | default `""` | Kubernetes namespace |
| `mlflow_run_id` | string | default `""` | MLflow run ID for artifact linkage |
| `mlflow_experiment` | string | default `""` | MLflow experiment name |
| `error` | string or null | default `""` | Error message on failure |
| `created_at` | ISO 8601 string | NOT NULL | Creation timestamp |
| `started_at` | ISO 8601 string or null | | When backend submission began |
| `completed_at` | ISO 8601 string or null | | When job reached terminal state |
| `backend_handle` | JSON string | default `""` | Serialized `BackendHandle` |

**Storage**: SQLite `jobs` table. Config is JSON-serialized. Indexed on `status`, `type`, `created_at`, `user_id`.

**Sanitization invariant**: The repository layer MUST normalize `error` values of `""` or `"None"` to `null`, and empty string `started_at`/`completed_at` to `null`, before returning rows to callers (`repository.py:180-186`).

### 2.2 JobType

An enum with exactly two values in v1:

- `training` — fine-tuning via training-hub or TRL
- `sdg` — synthetic data generation via NVIDIA Data Designer

### 2.3 JobStatus

An enum defining the job lifecycle:

- `queued` — awaiting worker pickup
- `provisioning` — submitted to backend, not yet running
- `running` — backend reports process is alive
- `succeeded` — exit code 0
- `failed` — nonzero exit code or exception
- `cancelled` — user-initiated or negative exit code

### 2.4 Document

| Field | Type | Description |
|---|---|---|
| `document_id` | string | PRIMARY KEY; equals `mlflow_run_id` when MLflow is configured |
| `mlflow_run_id` | string | MLflow run tracking this document's artifacts |
| `filename` | string | Sanitized original filename |
| `format` | `OutputFormat` | `md`, `text`, `json`, or `html` |
| `content` | string | Parsed document content |
| `created_at` | ISO 8601 string | Processing timestamp |

**Dual storage**: Documents are stored both in the local SQLite `documents` table and as MLflow artifacts. The MLflow run is the authoritative artifact store; local storage is a cache.

### 2.5 Recipe

Recipes are YAML files on disk (not database entities). They define reusable job configurations.

| Property | Description |
|---|---|
| `name` | Path-like identifier (e.g., `templates/custom/my-recipe`) |
| `type` | `training` or `sdg` |
| `description` | Human-readable description |
| `extends` | Parent recipe name for inheritance |
| `config` | Job configuration defaults |

**Inheritance**: Recipes MAY use `extends:` to inherit from a parent. The system MUST perform deep merge (child overrides parent). Circular `extends:` chains MUST be detected and rejected with `CircularRecipeError`.

**Protection**: Only recipes under `templates/custom/` MAY be deleted. Attempts to delete built-in recipes MUST raise `ProtectedRecipeError`.

### 2.6 ComputeSpec

Request-level compute preferences:

| Field | Type | Default | Description |
|---|---|---|---|
| `backend` | string | `"local"` | Compute backend name |
| `gpus` | int | `0` | Number of GPUs (>= 0) |
| `gpu_type` | string or null | `null` | GPU type constraint |

### 2.7 BackendHandle

Opaque handle returned by `backend.submit()`. Serialized as JSON in the `backend_handle` column.

| Field | Type | Description |
|---|---|---|
| `backend_name` | string | Which backend owns this handle |
| `job_id` | string | Job identifier |
| `remote_pid` | int or null | PID on remote host (SSH backend) |
| `remote_dir` | string or null | Working directory on remote/local |
| `container_id` | string or null | Container ID (SSH/podman backend) |
| `scheduler_id` | string or null | K8s Job name |
| `secret_names` | list of (name, namespace) tuples or null | K8s Secrets to clean up |

### 2.8 BackendStatus

Status report from a compute backend:

| Field | Type | Description |
|---|---|---|
| `running` | bool | Whether the process is still alive |
| `exit_code` | int or null | Process exit code (null if still running) |
| `error` | string or null | Error message from the backend |

### 2.9 JobSpec

The contract between the worker and compute backends:

| Field | Type | Default | Description |
|---|---|---|---|
| `job_id` | string | required | Job identifier |
| `command` | list[string] | required | Command to execute |
| `env` | dict | `{}` | Environment variables |
| `work_dir` | string | `"."` | Working directory |
| `image` | string or null | `null` | Container image |
| `timeout` | int or null | `null` | Timeout in seconds |
| `resources` | `Resources` | `Resources()` | Compute resources |
| `config_files` | dict[str, str] | `{}` | Filename→content pairs to mount |
| `s3_downloads` | list[S3Download] | `[]` | S3 artifacts to download before execution |
| `job_type` | string | `""` | Job type string |
| `user_id` | string | `""` | User identifier |

**Invariant**: Each `JobSpec` instance MUST get a fresh `Resources` default (via `field(default_factory=Resources)`), not a shared mutable instance.

### 2.10 Resources

| Field | Type | Default |
|---|---|---|
| `gpus` | int | `1` |
| `gpu_type` | string or null | `null` |
| `cpus` | int or null | `null` |
| `memory_gb` | int or null | `null` |
| `nodes` | int | `1` |

### 2.11 Settings (singleton)

Application configuration via `pydantic-settings` with `AMORTIZED_` environment variable prefix. Key fields:

| Field | Default | Description |
|---|---|---|
| `host` | `"0.0.0.0"` | Server bind address |
| `port` | `8000` | Server bind port |
| `db_path` | `./data/amortized.db` | SQLite database path |
| `api_key` | `""` | Bearer token for auth (empty = no auth) |
| `compute_backend` | `"local"` | Primary compute backend |
| `mlflow_tracking_uri` | `""` | MLflow tracking URI (empty = disabled) |
| `gateway_url` | `""` | MLflow AI Gateway URL |
| `docling_url` | `""` | Docling-serve URL |

**Resolved default backend**: `default_backend || compute_backend || "local"`.

### 2.12 GatewayModel

| Field | Type | Description |
|---|---|---|
| `name` | string | Endpoint name (used as `model` in job config) |
| `provider` | string | Model provider (e.g., `openai`, `anthropic`) |
| `model_name` | string | Underlying model identifier |

---

## 3. State Machines

### 3.1 Job Lifecycle State Machine

```
                    ┌──────────────────────────────────────────┐
                    │     User cancel (any active state)        │
                    ▼                                          │
  ┌────────┐   ┌──────────────┐   ┌─────────┐   ┌───────────┐│
  │ queued │──▶│ provisioning │──▶│ running │──▶│ succeeded ││
  └────────┘   └──────────────┘   └─────────┘   └───────────┘│
       │              │               │                        │
       │              │               │          ┌──────────┐  │
       │              │               ├─────────▶│  failed  │  │
       │              │               │          └──────────┘  │
       │              │               │                        │
       └──────────────┴───────────────┴───────▶┌───────────┐◀─┘
                                               │ cancelled │
                                               └───────────┘
```

**Terminal states**: `succeeded`, `failed`, `cancelled`.

#### 3.1.1 Transition Rules

| From | To | Trigger | Source |
|---|---|---|---|
| `queued` | `provisioning` | Worker picks job, calls `backend.submit()` | `worker.py:507-510` |
| `provisioning` | `running` | `backend.status()` reports `running=True` | `worker.py:522-523` |
| `running` | `succeeded` | `exit_code == 0` | `worker.py:534-551` |
| `running` | `failed` | `exit_code > 0` or exception | `worker.py:560-568` |
| `running` | `cancelled` | `exit_code < 0` | `worker.py:552-559` |
| `{queued, provisioning, running}` | `cancelled` | User calls cancel API | `core/jobs.py:70-96` |
| any active | `failed` | Unhandled exception during `_run_job()` | `worker.py:570-594` |
| `running` (orphaned) | `failed` | Startup orphan cleanup finds no live process | `worker.py:597-625` |

#### 3.1.2 Cancel Semantics

- Cancelling a `queued` or `provisioning` job MUST set status to `cancelled` without backend interaction.
- Cancelling a `running` job MUST attempt `backend.cancel(handle)`. If backend cancellation fails, the job MUST still be marked `cancelled` in the database.
- Cancelling an already `cancelled` job MUST be idempotent (return current state, no error).
- Cancelling a `succeeded` or `failed` job MUST raise `InvalidJobStateError` (HTTP 400).

#### 3.1.3 Delete Semantics

- Only jobs in terminal states (`succeeded`, `failed`, `cancelled`) MAY be deleted.
- Deleting a non-terminal job MUST raise `InvalidJobStateError` with message "Cannot delete job in state '...' — cancel it first".
- Deleting a nonexistent job MUST raise `JobNotFoundError` (HTTP 404).

### 3.2 Worker Loop State Machine

```
  ┌─────────┐     job found     ┌──────────┐
  │  poll   │──────────────────▶│ _run_job │
  └─────────┘                   └──────────┘
       ▲    no job: sleep 2s         │
       │◀────────────────────────────┘
       │
       │  CancelledError → shutdown
       ▼
  ┌──────────┐
  │ shutdown │
  └──────────┘
```

- The worker MUST pick the oldest `queued` job first (FIFO via `ORDER BY created_at ASC LIMIT 1`).
- On exception, the worker MUST log the error and continue after sleeping `poll_interval` seconds.
- On `CancelledError`, the worker MUST exit cleanly.

### 3.3 Document Processing State Machine

```
  upload/URL ──▶ call docling-serve ──▶ extract content
                                            │
                                   ┌────────┴────────┐
                                   ▼                  ▼
                              MLflow store        return result
                         (create experiment       (no MLflow)
                          → create run
                          → upload artifacts
                          → mark FINISHED)
                                   │
                              on failure:
                          mark run FAILED
```

### 3.4 Studio Chat State Machine

```
  idle ──▶ streaming ──▶ done
                    └──▶ error
```

- `sendMessage` MUST be a no-op when state is `streaming` (guard against double-send).
- On API error, the placeholder assistant message MUST be removed from state.

### 3.5 Studio Session Reconnection

```
  (no session) ──▶ POST /agent/session ──▶ connected
  connected ──▶ message OK ──▶ connected
  connected ──▶ 404/500 ──▶ reconnecting ──▶ context replay ──▶ rebuilt
  connected ──▶ 502/network ──▶ reset session ──▶ retry
  any error after MAX_RETRIES(2) ──▶ clear session ──▶ throw
```

- Context replay MUST summarize prior messages to a maximum of 6000 characters, using reverse chronological order.

---

## 4. Module Behavioral Contracts

### 4.1 API Layer (`api/`)

#### 4.1.1 Authentication Middleware (`main.py:146-167`)

- If `settings.api_key` is empty, all requests MUST be allowed without authentication.
- If `settings.api_key` is set, requests to paths NOT in `_AUTH_SKIP_PATHS` MUST include `Authorization: Bearer <token>`.
- Token comparison MUST use constant-time comparison (`hmac.compare_digest`).
- Unauthenticated requests MUST receive HTTP 401 with `ErrorResponse{code: "unauthorized"}`.
- Paths `/api/v1/health`, `/docs`, `/openapi.json`, `/redoc` MUST be exempt from authentication.

#### 4.1.2 Jobs Endpoint (`api/jobs.py`)

**POST `/api/v1/jobs`** (create_job)
- MUST validate config via `_validate_config()` before creation.
- If `dry_run=true`, MUST return HTTP 200 with `DryRunResponse` (no job created).
- If validation fails and not dry_run, MUST return HTTP 422.
- MUST strip sensitive keys (`api_key`, `api_secret`, `token`, `password`) from config before storage, storing them separately as `_secrets`.
- MUST extract `user_id` from `X-Forwarded-User` header.
- MUST return HTTP 201 with the created `Job` on success.
- Returned config MUST be redacted via `redact_config()`.

**GET `/api/v1/jobs`** (list_jobs)
- MUST support optional `status` and `type` query parameters.
- MUST return jobs in reverse chronological order (`ORDER BY created_at DESC`).

**GET `/api/v1/jobs/{job_id}`** (get_job)
- MUST return HTTP 404 if job does not exist.
- Returned config MUST be redacted.

**DELETE `/api/v1/jobs/{job_id}`** (cancel_job)
- MUST return HTTP 404 if job does not exist.
- MUST return HTTP 400 if job is in `succeeded` or `failed` state.
- MUST be idempotent for already-cancelled jobs.

**POST `/api/v1/jobs/{job_id}/delete`** (delete_job)
- MUST return HTTP 204 on success.
- MUST return HTTP 404 if job does not exist.
- MUST return HTTP 400 if job is not in a terminal state.

**GET `/api/v1/jobs/{job_id}/logs`** (get_job_logs)
- MUST return empty logs with message if no backend handle exists.
- MUST default `tail` to 100 lines.
- MUST NOT propagate backend log fetch exceptions to the client; MUST return empty logs with error message instead.

**GET `/api/v1/jobs/{job_id}/artifacts`** (get_job_artifacts)
- MUST return the MLflow artifact URI resolved from the job's `mlflow_run_id`.
- If no `mlflow_run_id` exists, MUST return empty artifact_uri with explanatory message.

#### 4.1.3 Recipes Endpoint (`api/recipes.py`)

- **GET `/api/v1/recipes`**: MUST list recipes from `templates/` and `examples/` directories.
- **GET `/api/v1/recipes/{name}`**: MUST resolve `extends:` inheritance before returning.
- **POST `/api/v1/recipes`**: MUST save to `templates/custom/` directory only.
- **DELETE `/api/v1/recipes/{name}`**: MUST reject deletion of non-custom recipes with HTTP 403.
- **POST `/api/v1/jobs/recipe`**: MUST flatten recipe overrides into config, strip secrets, and create a job.

#### 4.1.4 Documents Endpoint (`api/documents.py`)

**POST `/api/v1/documents/convert`** (upload)
- Empty uploads MUST be rejected with HTTP 400.
- Files exceeding 100 MB MUST be rejected with HTTP 413.
- Filenames MUST be sanitized: path traversal stripped, null bytes removed, truncated to 255 chars.
- If `docling_url` is not configured, MUST return HTTP 503.

**POST `/api/v1/documents/convert/url`** (URL conversion)
- MUST validate URL scheme (only `http://` and `https://`).
- MUST block SSRF targets: cloud metadata endpoints (`169.254.169.254`, `metadata.google.internal`, `metadata.azure.com`), loopback addresses, private IP ranges, and internal service domains (`*.svc.cluster.local`, `*.local`, `*.internal`).

**Upstream error mapping**:
- `httpx.ConnectError` → HTTP 502
- `httpx.TimeoutException` → HTTP 504
- `httpx.TransportError` → HTTP 502
- Non-JSON response from docling → HTTP 502

#### 4.1.5 Error Response Envelope

All API errors MUST use the `ErrorResponse` envelope:

```json
{
  "code": "string",
  "message": "string",
  "details": []
}
```

Validation errors (HTTP 422) MUST include structured detail objects with `loc`, `msg`, and `type` fields.

### 4.2 Worker (`worker.py`)

#### 4.2.1 Job Dispatch Pipeline

The worker MUST execute the following pipeline for each picked job:

1. Pop `_secrets` from stored config.
2. Resolve output directory based on job type and backend.
3. Resolve compute backend from settings.
4. Expand `~` in all string config values.
5. Build environment variables (forwarded env, secrets, MLflow config).
6. Resolve parent job artifacts for chaining (if `parent_job_id` set).
7. Build command and config files based on job type.
8. Construct `JobSpec` and call `backend.submit()`.
9. Transition to `provisioning`, record `started_at` and `backend_handle`.
10. Poll `backend.status()` every 2 seconds until not running.
11. On first `running=True` poll, transition to `running`.
12. On completion, determine terminal state from exit code.
13. For `exit_code == 0`: extract MLflow run ID from logs, set tags, register model (training only).
14. Clean up K8s secrets if applicable.

#### 4.2.2 Training Job Config Translation

For training jobs using training-hub (`thub`):

- Field names MUST be translated per `_TRAINING_HUB_FIELD_MAP`:
  - `model_name_or_path` → `model_path`
  - `num_train_epochs` → `num_epochs`
  - `per_device_train_batch_size` → `micro_batch_size`
  - `max_length` → `max_seq_len`
  - `output_dir` → `ckpt_output_dir` (except for `gepa` algorithm)
- Keys in `_TRAINING_HUB_SKIP_KEYS` MUST be excluded from the generated YAML.
- For `sft`/`lora_sft`: `micro_batch_size` MUST be popped and converted to `effective_batch_size` (batch × 4).
- Algorithm aliases MUST be resolved: `lora` → `lora_sft`, `qlora` → `lora_sft`, `qlora_sft` → `lora_sft`.

#### 4.2.3 SDG Job Config Translation

For SDG jobs using NVIDIA Data Designer:

- Config MUST be wrapped as `{data_designer: config}`.
- Stale keys (`model`, `api_base`, `api_key`, `num_samples`, etc.) MUST be stripped before wrapping.
- `num_records` MUST be extracted (default 100) and passed as CLI argument.
- Document content MUST be fetched from MLflow and injected as config files if `document_ids` is present.
- The command MUST chain `data-designer create` and `upload_to_mlflow.py` via `sh -c`.

#### 4.2.4 Parent Job Artifact Resolution

- If a training job has an SDG parent, the worker MUST resolve the parent's MLflow artifact URI and inject an S3 download for the `generated_data/` directory.
- If the parent has no `mlflow_run_id` or the URI cannot be resolved, the worker MUST log a warning and continue with the existing config.
- Existing explicit `data_path` values starting with `s3://` MUST NOT be overridden.

#### 4.2.5 MLflow Integration

- If `mlflow_tracking_uri` is configured, the worker MUST set `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` environment variables on every job.
- Experiment name format MUST be `amortized/{job_type}/{job_id[:8]}`.
- For training jobs, `HF_MLFLOW_LOG_ARTIFACTS=true` MUST be set.
- For training jobs, `report_to` MUST default to `"mlflow"` when MLflow is configured.
- On success, the worker MUST extract the MLflow run ID from logs (pattern: `AMORTIZED_MLFLOW_RUN_ID=<hex32>` or `/runs/<hex32>`).
- On success, the worker MUST tag the MLflow run with `job_type` and `job_id`.
- For successful training jobs, the worker MUST register a model version in the MLflow model registry. Registration failure MUST NOT cause the job to be marked as failed.

#### 4.2.6 Gateway-Managed Secrets

- When `gateway_url` is configured and job type is `sdg`, LLM-related secrets (`api_key`, `api_secret`, `token`) MUST NOT be injected into the job environment.
- Instead, `OPENAI_API_KEY=gateway-managed` MUST be set as a placeholder.

#### 4.2.7 Orphan Cleanup

On server startup, `cleanup_orphaned_jobs()` MUST:

1. Query all jobs with status `running`.
2. For each, attempt to check backend status via the stored handle.
3. If the backend reports the process is still alive, log "Re-adopted running job".
4. If the process is not alive or the backend is unreachable, mark the job as `failed` with error "Orphaned job — process no longer running".

#### 4.2.8 Error Recovery

If `_run_job()` raises an unhandled exception:

1. The error text MUST be written to `stderr.log` in the job's output directory.
2. A fallback `BackendHandle` MUST be created pointing to the output directory.
3. The job MUST be marked `failed` with the exception message.

### 4.3 Compute Backend Protocol (`backends/__init__.py`)

All compute backends MUST implement the `ComputeBackend` protocol:

| Method | Signature | Contract |
|---|---|---|
| `capabilities()` | `() → set[Capability]` | Return supported capabilities |
| `submit()` | `(JobSpec) → BackendHandle` | Start job execution; MUST NOT block until completion |
| `status()` | `(BackendHandle) → BackendStatus` | Check if job is still running |
| `cancel()` | `(BackendHandle) → None` | Request job termination |
| `logs()` | `(BackendHandle) → AsyncIterator[str]` | Stream log lines |

**Capabilities** (enum): `GPU`, `MULTI_NODE`, `LOG_STREAM`, `STOP`, `RESUME`.

**Backend registry** (`core/compute.py`):
- `register_backend(backend)` MUST store the backend keyed by `backend.name`.
- `get_backend(name)` MUST raise `KeyError` if the backend is not registered.
- `check_capabilities(backend, required)` MUST raise `MissingCapabilityError` if required capabilities are not a subset of the backend's capabilities.

### 4.4 Credential Management

#### 4.4.1 Secret Stripping (`api/jobs.py:_strip_secrets`)

Before database storage, the API layer MUST:

1. Extract values for keys in `{api_key, api_secret, token, password}` from the config.
2. Store the original values in a separate `secrets` dict.
3. Pass the clean config and secrets separately to `core_create_job()`.
4. The core layer MUST store secrets as `config["_secrets"]`.

#### 4.4.2 Config Redaction (`core/redact.py`)

Before returning config to clients, `redact_config()` MUST:

1. Replace values for keys in `{api_key, api_secret, token, password, secret}` with `"***redacted***"`.
2. Recursively traverse nested dicts and lists.
3. MUST NOT mutate the original config dict.

#### 4.4.3 Log Text Redaction (`core/redact.py`)

`redact_text()` MUST replace credential values in log text matching the pattern `<KEY_NAME>=<value>` or `<KEY_NAME>: <value>` where the key name contains `API_KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or `CREDENTIAL`.

#### 4.4.4 Secret Injection (Worker)

When running a job, the worker MUST:

1. Pop `_secrets` from the stored config.
2. Map secret keys to environment variable names (e.g., `api_key` → `OPENAI_API_KEY`).
3. For K8s backends, inject secrets as K8s Secret resources.
4. After job completion, clean up K8s secrets via `backend.cleanup_secrets()`.

### 4.5 Recipe System (`core/recipes.py`)

#### 4.5.1 Loading

- `load_recipe(name)` MUST resolve the recipe file at `{recipes_dir}/{name}.yaml`.
- If the file does not exist, MUST raise `RecipeNotFoundError`.
- If `extends:` is present, MUST recursively load the parent and deep-merge.
- Deep merge: for dict values, merge recursively; for all other types, child overrides parent.

#### 4.5.2 Circular Reference Detection

- The loader MUST track visited recipe names in a `_chain` set.
- If a recipe name appears in `_chain`, MUST raise `CircularRecipeError`.
- This detects both direct self-reference and transitive cycles.

#### 4.5.3 Override Application

`apply_overrides(recipe, overrides)` MUST:

1. Support dot-notation keys (e.g., `"config.learning_rate"` → `recipe["config"]["learning_rate"]`).
2. Auto-prefix non-meta keys with `"config."` if they don't already start with `"config"`.
3. Skip meta keys (`type`, `description`, `extends`, `name`).
4. Skip falsy values (except `0` and `False`).
5. Create intermediate dicts for nested paths that don't exist.
6. Deep-copy the recipe before modification (MUST NOT mutate original).

#### 4.5.4 Flattening

`flatten_recipe_to_config()` MUST:

1. Merge top-level recipe keys (excluding meta keys) into the `config` sub-dict.
2. Normalize `teacher_model` → `model` for SDG compatibility.

### 4.6 Database Layer (`db/`)

#### 4.6.1 Schema

Two tables: `jobs` and `documents`. Schema is defined in `schema.sql` and applied via `init_db()`.

#### 4.6.2 Repository

- `create_job()` MUST always insert with status `queued`.
- `update_job()` MUST only allow updates to columns in `_UPDATABLE_COLUMNS`. Attempting to update an unlisted column MUST raise `ValueError`.
- `pick_pending_job()` MUST select the oldest `queued` job (`ORDER BY created_at ASC LIMIT 1`).
- `list_jobs()` MUST return results in reverse chronological order (`ORDER BY created_at DESC`).
- All write operations MUST commit immediately.

### 4.7 Application Lifecycle (`main.py`)

On startup (lifespan context manager), the application MUST:

1. Initialize the database (`init_db()`).
2. Load and register compute backends (`_load_backends()`).
3. Run orphan cleanup (`cleanup_orphaned_jobs()`).
4. Start the worker loop as a background task.

On shutdown:

1. Cancel the worker task.
2. Close the database connection.

#### 4.7.1 Backend Loading

`_load_backends()` MUST:

1. Always register a `LocalBackend`.
2. If `compute_backend == "kubernetes"`, register a `KubernetesBackend`.
3. Read `~/.amortized/config.yaml` for additional SSH backends and settings overrides.
4. Unknown backend types in config.yaml MUST be silently skipped.
5. Invalid YAML in config.yaml MUST be logged and skipped (not crash).

#### 4.7.2 CORS

- Origins MUST be parsed from comma-separated `cors_origins` setting.
- If origins list is `["*"]`, `allow_credentials` MUST be `False`.
- Otherwise, `allow_credentials` MUST be `True`.

---

## 5. Failure Model

### 5.1 Error Hierarchy

| Error Class | Module | HTTP Status | Condition |
|---|---|---|---|
| `JobNotFoundError` | `core/jobs.py` | 404 | Job ID does not exist |
| `InvalidJobStateError` | `core/jobs.py` | 400 | Cancel/delete of job in invalid state |
| `RecipeNotFoundError` | `core/recipes.py` | 404 | Recipe YAML file not found |
| `CircularRecipeError` | `core/recipes.py` | 422 | Circular `extends:` chain detected |
| `ProtectedRecipeError` | `core/recipes.py` | 403 | Attempt to delete built-in recipe |
| `MissingCapabilityError` | `core/compute.py` | — | Backend lacks required capability |
| `KeyError` (backend) | `core/compute.py` | — | Unknown compute backend name |
| `ValidationError` (Pydantic) | `models.py` | 422 | Invalid config fields |

### 5.2 HTTP Error Mapping

| Status | Meaning | Trigger |
|---|---|---|
| 400 | Bad request | Invalid URL, empty upload, cancel of terminal job, invalid recipe name |
| 401 | Unauthorized | Missing or invalid Bearer token |
| 404 | Not found | Nonexistent job, recipe, document, or artifact |
| 413 | Payload too large | Document upload exceeds 100 MB |
| 422 | Validation error | Invalid config, unknown job type, circular recipe |
| 500 | Internal server error | Unhandled exception |
| 502 | Bad gateway | Upstream failure (docling-serve, MLflow, S3) |
| 503 | Service unavailable | Required service not configured (docling, MLflow) |
| 504 | Gateway timeout | Upstream timeout |

### 5.3 Upstream Failure Handling

#### MLflow Failures

- MLflow unavailability during document upload MUST NOT prevent document conversion; a warning MUST be appended to the response.
- MLflow unavailability during document listing MUST return HTTP 502.
- MLflow unavailability during model registration MUST NOT cause a succeeded job to be marked failed.

#### Docling-serve Failures

- `ConnectError` MUST map to HTTP 502 with "temporarily unavailable".
- `TimeoutException` MUST map to HTTP 504 with "request timed out".
- Non-JSON responses MUST map to HTTP 502.

#### Backend Failures

- If the compute backend is unknown, the job MUST be marked `failed` immediately.
- Backend cancellation failure during job cancel MUST be logged but MUST NOT prevent the job from being marked `cancelled`.

### 5.4 Idempotency Guarantees

- Cancelling an already-cancelled job MUST be idempotent (return current state).
- Creating a job with `dry_run=true` MUST NOT create any state.
- Worker loop errors MUST NOT crash the worker; it MUST retry after sleeping.

---

## 6. Security Model

### 6.1 Authentication

- Bearer token authentication via `Authorization` header.
- Constant-time token comparison to prevent timing attacks.
- Health, docs, and OpenAPI endpoints exempt from auth.

### 6.2 SSRF Protection (Documents)

URL conversion MUST block:

- Non-HTTP(S) schemes
- Cloud metadata endpoints (`169.254.169.254`, `metadata.google.internal`, `metadata.azure.com`)
- Loopback and localhost addresses
- Private IP ranges (RFC 1918)
- Link-local addresses
- Internal service domains (`*.svc.cluster.local`, `*.local`, `*.internal`)

### 6.3 Path Traversal Protection (Recipes)

- Recipe names containing `..` path segments MUST be rejected.
- Resolved paths MUST be validated as relative to the recipes directory.
- Only `templates/custom/` recipes may be deleted.

### 6.4 Filename Sanitization (Documents)

- `os.path.basename()` to strip directory components.
- Null bytes and path separators replaced with `_`.
- Truncated to 255 characters.
- Empty names replaced with `upload-<random>`.

### 6.5 Credential Isolation

- Credentials MUST be stripped from config before database storage.
- Credentials MUST be redacted in all API responses.
- Credentials MUST be scrubbed from log text.
- Credentials MUST be injected per-job as environment variables or K8s Secrets.
- K8s Secrets MUST be cleaned up after job completion.

---

## 7. Studio Frontend Contracts

### 7.1 Chat System

- Message sending MUST be serialized: `sendMessage` is a no-op during `streaming` state.
- A placeholder assistant message (empty content) MUST be immediately appended on send for optimistic UI.
- On error, the placeholder MUST be removed.
- Conversation auto-titling MUST occur on first user message, falling back to first 40 chars on failure.
- Option cards MUST lock (non-interactive) after selection.

### 7.2 Workflow Step Detection

The chat system implements a guided workflow with 9 steps:

`sdg-domain → sdg-categories → sdg-urgency → sdg-samples → sdg-teacher-model → sdg-confirm → training-student-model → training-method → training-confirm`

- Phase tags (`<phase>sdg:step_name</phase>`) in agent responses MUST take precedence over regex heuristic detection.
- Tags MUST be stripped from displayed content.

### 7.3 Job Monitoring

- `JobMonitorCard` MUST poll at 3-second intervals.
- Polling MUST stop when the job reaches a terminal state.
- Progress bar values: `queued=10%`, `provisioning=20%`, `running=35-80%` (time-based ramp), `terminal=100%`.
- On completion, contextual option cards MUST be auto-generated (different cards for success vs failure, SDG vs training).

### 7.4 Data Persistence

All Zustand stores MUST use `persist` middleware with `localStorage`:

| Store | Key | Contents |
|---|---|---|
| `settings-store` | `amortized-settings` | API key, chat model selection, enabled providers |
| `chat-store` | `amortized-chat` | Conversations, messages, session map |
| `entity-names-store` | `amortized-entity-names` | Custom display names for entities |
| `ui-store` | `amortized-ui` | Sidebar state, theme, tutorial progress |

### 7.5 API Client

- Every request MUST include an `X-Request-ID` header (`crypto.randomUUID()`).
- Request duration MUST be logged.
- Auth headers MUST be derived from `settings-store.apiKey`.
- Session reconnection MUST retry up to 2 times with context replay.

### 7.6 System Health

- The frontend MUST poll `/api/v1/health` every 30 seconds.
- Status MUST be one of: `ok`, `error`, `loading`.

---

## 8. External Dependencies

| Dependency | Role | Protocol | Failure Mode |
|---|---|---|---|
| SQLite | Job and document persistence | File-based | Fatal (server won't start) |
| MLflow | Artifact storage, experiment tracking, model registry | HTTP REST | Degraded (jobs succeed but artifacts unlinked) |
| MinIO/S3 | Object storage (behind MLflow) | S3 API | Degraded (artifact download fails) |
| Docling-serve | Document parsing (PDF, DOCX, etc.) | HTTP REST | Document processing unavailable (503) |
| MLflow AI Gateway | LLM routing for SDG | HTTP REST | SDG requires direct API key |
| Kubernetes API | Job/Deployment/ConfigMap/Secret management | K8s client | Jobs cannot be dispatched |
| Container images | `ghcr.io/amortized-ai/{training,data-designer}:latest` | OCI registry | ImagePullBackOff |

---

## 9. Configuration Reference

### 9.1 Environment Variables

All prefixed with `AMORTIZED_`:

| Variable | Default | Description |
|---|---|---|
| `AMORTIZED_HOST` | `0.0.0.0` | Server bind address |
| `AMORTIZED_PORT` | `8000` | Server bind port |
| `AMORTIZED_DB_PATH` | `./data/amortized.db` | SQLite database path |
| `AMORTIZED_API_KEY` | `""` | Bearer token (empty = no auth) |
| `AMORTIZED_COMPUTE_BACKEND` | `local` | Primary compute backend |
| `AMORTIZED_COMPUTE_NAMESPACE` | `amortized-jobs` | K8s namespace |
| `AMORTIZED_MLFLOW_TRACKING_URI` | `""` | MLflow URI (empty = disabled) |
| `AMORTIZED_GATEWAY_URL` | `""` | AI Gateway URL |
| `AMORTIZED_DOCLING_URL` | `""` | Docling-serve URL |
| `AMORTIZED_IMAGE_REGISTRY` | `ghcr.io/amortized-ai` | Container image registry |
| `AMORTIZED_IMAGE_PULL_POLICY` | `Always` | K8s image pull policy |
| `AMORTIZED_CORS_ORIGINS` | `*` | Comma-separated CORS origins |

### 9.2 Forwarded Environment Variables

The following env vars, when set on the server, are forwarded to job containers:

- Any names listed in `forward_env` from `~/.amortized/config.yaml`
- `MLFLOW_TRACKING_URI` (when MLflow is configured)
- `MLFLOW_EXPERIMENT_NAME`
- `MLFLOW_S3_ENDPOINT_URL` / `FSSPEC_S3_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_S3_BUCKET`

### 9.3 Container Images

| Job Type | Image | Entry Point |
|---|---|---|
| `training` | `ghcr.io/amortized-ai/training:latest` | `thub <algorithm> --config /amortized/config.yaml` |
| `sdg` | `ghcr.io/amortized-ai/data-designer:latest` | `sh -c "data-designer create ... && upload_to_mlflow.py ..."` |

---

## 10. Invariants

1. **Config-only dispatch**: The system MUST NOT generate Python scripts. All job behavior is defined by YAML configuration files delivered via ConfigMaps (K8s), file writes (SSH/local), or command-line arguments.

2. **Single code path**: The worker MUST use the same dispatch logic regardless of backend type. Backend-specific behavior is encapsulated behind the `ComputeBackend` protocol.

3. **MLflow is the artifact store**: The system MUST NOT write directly to S3 (except via the artifact proxy endpoint). All artifacts MUST flow through MLflow's API.

4. **Polling-based lifecycle**: In v1, the system uses polling (2-second interval) for job status. Push-based notifications are explicitly deferred.

5. **FIFO job ordering**: The worker MUST pick the oldest queued job first. There is no priority system in v1.

6. **Credential lifecycle**: Secrets MUST be stripped before storage → injected per-job at runtime → cleaned up after completion. At no point SHOULD credentials be visible in API responses or logs.

7. **Terminal state guard**: Only jobs in terminal states (`succeeded`, `failed`, `cancelled`) may be deleted. Active jobs MUST be cancelled before deletion.

8. **Idempotent cancel**: Cancelling an already-cancelled job MUST succeed without error.
