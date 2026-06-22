# Studio Migration — Aligning with the OpenShift Backend

## Context

Studio was built for the local SSH-based amortized backend. The backend has since been rebuilt for OpenShift with MLflow as the artifact store, K8s for compute, and config-only dispatch. Studio's TypeScript types, API calls, and UI features are out of sync with the current backend. This doc specifies every change needed.

No backward compatibility. This is a clean break.

## Backend Contract (Source of Truth)

The amortized API is the contract. Studio conforms to it. The OpenAPI spec at `/openapi.json` is auto-generated from FastAPI — Studio's types should match it exactly.

### Job Response Shape

```typescript
interface Job {
  id: string
  type: "training" | "sdg" | "eval" | "serve"
  status: "queued" | "provisioning" | "running" | "succeeded" | "failed" | "cancelled"
  config: Record<string, unknown>
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string | null
  started_at: string | null
  completed_at: string | null
  error: string | null
  output_dir: string | null
  mlflow_run_id: string  // NEW — links to MLflow experiment run
}
```

Key changes:
- `mlflow_run_id` is a new field (string, empty when no MLflow)
- Remove `backend_handle` (never exposed)
- Remove `progress` (never existed on the backend)

### Job Submission

```typescript
// POST /api/v1/jobs
interface JobRequest {
  type: string
  config: Record<string, unknown>
  compute?: { backend?: string; gpus?: number; gpu_type?: string }
  metadata?: Record<string, unknown>
  dry_run: boolean  // MUST be explicitly set to false to create a job
  depends_on?: string[]
}
```

**Critical**: `dry_run` defaults to `true` on the backend. Studio MUST send `dry_run: false` for actual job creation. Every `createJob()` call needs this.

### Serve Job with Training Reference

```typescript
// Serve a model from a completed training job
const serveConfig = {
  type: "serve",
  config: {
    training_job_id: "abc123",  // NEW — auto-resolves base model + adapter from MLflow
    // model_name_or_path: optional override
    // adapter_path: auto-set from MLflow
  },
  dry_run: false,
}
```

The backend resolves `training_job_id` → looks up the training job → gets `mlflow_run_id` → queries MLflow for artifact URI → sets base model + adapter path automatically.

### Artifact Response Shape

```typescript
interface Artifact {
  id: string
  job_id: string | null
  artifact_type: string  // "dataset" | "model" | "results" | "log" | "file"
  path: string           // may be S3 URI (s3://...) or local path
  size: number
  name: string
  location: string
  metadata: Record<string, unknown>
  producer_job: string | null
  created_at: string
}
```

Key changes:
- `path` may be an S3 URI now (not just a filesystem path)
- `name` and `location` are new fields
- `artifact_type` values may differ from Studio's current enum

### Health Response Shape

```typescript
interface HealthResponse {
  status: "ok"
  timestamp: string
  gpu: {
    available: boolean
    count: number
    devices: Array<{ name: string; memory: number }>
    note?: string  // e.g., "torch not installed"
  }
}
```

Not the old `{ name, vram_total, vram_used, vram_free, driver_version, cuda_version }` shape.

### Compute Backend Response Shape

```typescript
interface ComputeBackend {
  name: string
  capabilities: string[]  // ["gpu", "log_stream", "stop"]
}

interface ComputeStatus {
  name: string
  capabilities: string[]
  healthy: boolean
}
```

No `type`, `host`, `status`, `last_checked` fields. The `type` (local/ssh/kubernetes) is not exposed — backends are identified by name and capabilities.

### Evaluator Response Shape

```typescript
interface Evaluator {
  id: string
  name: string
  type: "llm" | "rule_based"
  description: string
  prompt: string
  judgment_type: "bool" | "int" | "float" | "enum"  // NOT "binary" | "likert" | "numeric"
  model: string
  temperature: number
  variables: string[]  // list of variable names, NOT a dict
  rule_config: Record<string, unknown> | null
  created_at: string
}
```

### Evaluation Create Request

```typescript
// POST /api/v1/evaluations
interface EvaluationCreate {
  evaluator_id: string      // singular, NOT evaluator_ids[]
  dataset: string           // dataset path or artifact ref, NOT dataset_artifact_id
  model_override?: string
  inference_params_override?: Record<string, unknown>
}
```

## Changes Required

### 1. Fix `types/api.ts` — Match Backend Exactly

Rewrite the entire type file to match the shapes above. Don't try to merge — replace.

### 2. Fix `createJob()` — Send `dry_run: false`

In `src/lib/api-client.ts` (or wherever `createJob` is defined):

```typescript
export async function createJob(config: JobRequest): Promise<Job> {
  return post<Job>("/api/v1/jobs", { ...config, dry_run: false })
}
```

Every job submission path (chat agent confirm, recipe execute, model deploy, evaluation run) must go through this and include `dry_run: false`.

### 3. Remove Global WebSocket — Use Polling

The backend has NO global `/api/v1/ws` endpoint. It has per-job WebSockets at `/api/v1/jobs/{id}/stream`.

Two options:
- **Option A (recommended)**: Remove the global `JobEventSocket` entirely. Use React Query's `refetchInterval` for job list polling (already exists as fallback). Per-job streaming can use SSE at `GET /api/v1/jobs/{id}/events?stream=true`.
- **Option B**: Add a global WebSocket endpoint to the backend. But this contradicts the "no backward compat" directive.

Go with Option A. The polling fallback already works — Studio just needs to not try connecting to the non-existent WebSocket.

### 4. Add MLflow Integration to UI

**Job detail panel — Overview tab:**
- Show `mlflow_run_id` when present
- Add "Open in MLflow" link: `{MLFLOW_TRACKING_URI}/#/experiments/{exp_id}/runs/{mlflow_run_id}`
- The MLflow tracking URI can come from the health/config endpoint, or be stored in Studio's settings

**Model detail page:**
- Show MLflow run link for the training job that produced the model
- Link to MLflow artifact browser for model weights

**How to get MLflow URI**: Add it to the health response or a config endpoint so Studio knows where MLflow is. Or add a setting in Studio's settings page.

### 5. Fix Deploy Dialog — Use `training_job_id`

In `use-models.ts`, `useDeployModel`:

```typescript
// Old (broken):
createJob({ type: "serve", model_artifact_id: model.id, model_path: model.path })

// New:
createJob({
  type: "serve",
  config: {
    training_job_id: model.producer_job,  // the training job ID
  },
  dry_run: false,
})
```

The backend resolves everything from `training_job_id` — base model, adapter path, MLflow artifacts.

### 6. Fix Dataset Preview

The backend returns a different shape for artifact previews. Either:
- Update Studio to match the backend's response shape
- Or update the backend's preview endpoint to return `{ rows, total_rows }` for JSONL files

Check what `GET /api/v1/jobs/{job_id}/artifacts/{artifact_id}/preview` actually returns and match it.

### 7. Fix Evaluator Types

- Change judgment type enum: `"binary"` → `"bool"`, `"likert"` → `"int"`, `"numeric"` → `"float"`
- Change `variables` from `Record<string, string>` to `string[]`
- Remove `is_system` (doesn't exist)
- Add `description`, `rule_config`

### 8. Fix Evaluation Submission

- Change `evaluator_ids: string[]` to `evaluator_id: string`
- Change `dataset_artifact_id: string` to `dataset: string`
- Change `model_artifact_id: string` to `model_override?: string`

### 9. Fix Health/GPU Display

Update `GpuInfoPanel` to handle the new shape:
```typescript
// Old: gpu.name, gpu.vram_total, gpu.vram_used
// New: gpu.available, gpu.count, gpu.devices[].name, gpu.devices[].memory
```

On K8s, `gpu.available` will be `false` (GPUs are on worker nodes, not the server pod). Show "GPUs available on compute nodes" instead of trying to show VRAM usage.

### 10. Fix Compute Backend Display

Update settings page backend table:
- Remove `type`, `host`, `status`, `last_checked` columns
- Show `name` and `capabilities` instead
- Remove "Add SSH Backend" dialog (K8s is the primary backend on OpenShift)
- Or keep it for hybrid deployments where both SSH and K8s backends exist

### 11. Add `kubernetes` Backend Type

The settings page should recognize and display the kubernetes backend:
- Show capabilities: `gpu`, `log_stream`, `stop`
- Show namespace: from backend config
- Don't show an "Add" button for K8s (it's configured via env vars, not the API)

### 12. Fix VRAM Estimator

On K8s, the server pod doesn't have GPU info. The VRAM estimator should:
- Still work as a calculator (model size × method → VRAM estimate)
- Not depend on local GPU detection
- Show estimated VRAM vs available VRAM on the cluster (if known)

### 13. Studio Dockerfile

The Studio Dockerfile in the amortized repo (`studio/Dockerfile`) is a template. The actual build must happen from the Studio repo. The nginx config (`studio/nginx.conf.template`) is correct — it proxies `/api/` to the backend.

For OpenShift deployment:
1. Build from Studio repo: `docker build -t ghcr.io/amortized-ai/studio:latest .`
2. Use the nginx template for SPA routing + API proxy
3. Set `BACKEND_HOST` and `BACKEND_PORT` env vars in the K8s deployment

### 14. Runtime API URL Configuration

Currently `VITE_API_URL` is build-time only. For OpenShift, Studio is served by nginx which proxies `/api/` to the backend — so same-origin works and no URL config is needed.

Remove the `apiUrl` setting from the settings store. It's vestigial — the actual API client doesn't use it.

## Implementation Priority

### Phase 1 — Make it work (critical fixes)

1. Fix `types/api.ts` — match backend shapes
2. Fix `createJob()` — `dry_run: false`
3. Remove global WebSocket — use polling
4. Fix evaluation create — correct field names
5. Fix deploy dialog — use `training_job_id`

### Phase 2 — OpenShift features

6. Add `mlflow_run_id` display + MLflow links
7. Fix health/GPU display for K8s
8. Fix compute backend display
9. Fix evaluator types

### Phase 3 — Polish

10. Fix dataset preview shape
11. Fix VRAM estimator for K8s
12. Remove vestigial `apiUrl` setting
13. Add "Open in MLflow" links throughout

## Testing

After each phase:
1. `npm run typecheck` — TypeScript compilation
2. `npm test` — unit tests
3. `npm run build` — production build succeeds
4. Manual: submit a job via chat, verify it creates (not dry-run)
5. Manual: check job detail shows `mlflow_run_id`
6. Manual: deploy a model via training_job_id
7. Playwright E2E: update MSW handlers to match new backend shapes
