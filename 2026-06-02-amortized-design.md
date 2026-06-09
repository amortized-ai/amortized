# Amortized — API-First ML Control Plane

*Amortize the cost of running with frontier models.*

## Overview

Amortized is an API-first **job orchestration control plane** for ML workloads. Everything is a **Job**, submitted to a unified API, dispatched to **Compute Backends**, and executed in containers that wrap ML libraries (Training Hub, Amortized Synth, vLLM). The control plane manages 4 job types (training, synthesis, inference, evaluation) with extensibility handled by the underlying dispatch layers, not by Amortized itself. Humans, SDKs, and AI agents all use the same API.

**Target users**: ML engineers at companies running ML workloads in production.

**Core philosophy**: The API is the product. The control plane owns job lifecycle, compute dispatch, and state — not ML logic. ML capabilities live in container images that wrap dispatch layers (Training Hub for training, Amortized Synth for synthesis) or direct libraries (vLLM for inference).

**Deployment**: Hybrid — local CLI for individual use, persistent server for teams. SQLite for v1, repository pattern enables PostgreSQL later. Same binary, config-toggled.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Clients                               │
│  CLI (REST client)  │  Python SDK  │  MCP (AI agents)   │
└────────────┬────────┴──────┬───────┴──────┬─────────────┘
             │               │              │
             ▼               ▼              ▼
┌─────────────────────────────────────────────────────────┐
│                  API Server (FastAPI)                     │
│  POST /jobs  GET /jobs/{id}  GET /job-types  SSE logs    │
│  Auth middleware (optional)  │  MCP auto-generated       │
├─────────────────────────────────────────────────────────┤
│                    Core Domain                           │
│  Job Scheduler  │  Job Type Registry  │  Artifact Store   │
│  Event Log      │  Recipe Engine      │  Config Validator │
├─────────────────────────────────────────────────────────┤
│                 Compute Backends                         │
│  Slurm (profiles)  │  SkyPilot  │  Kubernetes             │
├─────────────────────────────────────────────────────────┤
│              Container Runners (4 job types)              │
│  Training Hub  │  Amortized Synth  │  vLLM  │  LLM-as-Judge   │
└─────────────────────────────────────────────────────────┘
```

## Core Abstractions

### Job

The universal unit of work. Training runs, evaluations, inference batches, data synthesis — all are jobs. The control plane manages their lifecycle uniformly without type-specific code paths.

```python
@dataclass
class JobRequest:
    type: str                          # "training", "synth", "inference", "eval"
    config: dict                       # validated against the job type's JSON Schema
    compute: ComputeSpec               # where to run (Slurm, SkyPilot, K8s)
    depends_on: list[ArtifactRef] = [] # artifact dependencies from prior jobs
    metadata: dict = {}                # user tags, notes
    dry_run: bool = True               # preview by default, explicit opt-in to execute
```

The control plane validates `config` against the JSON Schema for the given `type` (loaded from `containers/{type}/schema.json`). It never interprets the config contents — that's the container runner's job.

### Container Runners

Each job type has a container image with a runner script. The runner reads config, calls the underlying library (Training Hub, Amortized Synth, vLLM, or a judge), and emits events. A shared Python module (`containers/shared/`) provides the common utilities.

```python
# containers/shared/context.py — copied into every container image
@dataclass
class RunContext:
    job_id: str
    work_dir: Path                    # scratch directory for this job
    config: dict                      # parsed from /amortized/config.json
    artifacts: dict[str, str]         # resolved input artifact paths
    
    def emit(self, event: Event) -> None:
        """Emit via stdout (JSON line) + HTTP POST to AMORTIZED_EVENTS_URL."""

    def save_artifact(self, name: str, path: Path) -> None:
        """Upload to shared storage and emit artifact event."""

    def is_cancelled(self) -> bool:
        """Check cancellation signal."""

    @classmethod
    def from_environment(cls) -> "RunContext":
        """Bootstrap from /amortized/config.json + env vars."""
```

```python
# containers/training/runner.py — example runner
from shared.context import RunContext
import training_hub

def main():
    ctx = RunContext.from_environment()
    ctx.emit(Event(type="progress", data={"message": "Starting training"}))
    
    training_hub.train(
        algorithm=ctx.config["algorithm"],
        backend=ctx.config.get("backend"),
        **ctx.config["params"],
    )
    
    ctx.save_artifact("model", ctx.config["output_dir"])

if __name__ == "__main__":
    main()
```

Containers always run on remote GPU instances. Slurm uses Singularity/Apptainer, K8s uses pods, SkyPilot provisions cloud VMs. There is no local compute backend — the control plane runs locally (or as a team server), all execution happens remotely. For container developers, `amortized submit --dev` mounts source code as a volume — no image rebuild needed.

### Compute Backend

Where jobs physically execute. Backends handle provisioning and process management, not ML logic.

```python
class ComputeBackend(Protocol):
    name: str

    def capabilities(self) -> set[Capability]:
        """What this backend supports: STOP, RESUME, LOG_STREAM, GPU, MULTI_NODE"""

    async def submit(self, job: JobSpec) -> BackendHandle:
        """Submit a job. Returns a handle for tracking."""

    async def status(self, handle: BackendHandle) -> BackendStatus:
        """Poll current state from the backend."""

    async def cancel(self, handle: BackendHandle) -> None:
        """Cancel a running job."""

    async def logs(self, handle: BackendHandle) -> AsyncIterator[str]:
        """Stream logs. Only available if LOG_STREAM in capabilities."""
```

Backends declare capabilities instead of implementing no-ops. The control plane checks capabilities before calling optional methods.

### Artifact

Typed references to data that flow between jobs.

```python
@dataclass
class Artifact:
    id: str                    # "art_abc123"
    type: str                  # open string: "model", "dataset", "checkpoint", "endpoint", etc.
    name: str                  # "llama-8b-sft-v2"
    location: str              # local path, S3 URI, HuggingFace repo ID
    producer_job: str | None   # job that created this artifact
    metadata: dict             # size, format, metrics — type-specific, schema-free
    created_at: datetime
```

Artifacts are references, not containers. The control plane tracks what exists and where. A training job consumes a dataset artifact and produces a model artifact. An eval job consumes a model and produces results. Jobs link via artifact references — lightweight pipelines without a DAG engine.

### Event

The universal communication channel from container runners to the control plane.

```python
@dataclass
class Event:
    type: str          # "state_change", "artifact", "progress", "tracking", "error", "heartbeat"
    timestamp: float
    data: dict
```

Events are **job lifecycle signals**, not training metrics. Detailed metrics (loss, learning rate, gradients) go to dedicated experiment tracking tools (W&B, MLflow, TensorBoard) directly from the container. The control plane captures a one-time `tracking` event with the experiment URL (e.g., W&B run link) so users and agents can navigate from job status to experiment dashboard.

Event types are open strings — container runners can emit custom event types. The control plane persists all events without interpreting them.

## Job Lifecycle

```
VALIDATING → QUEUED → PROVISIONING → RUNNING → SUCCEEDED
                │          │            │
                │          │            ├→ FAILED
                │          │            └→ CANCELLED
                │          └→ FAILED (compute error)
                └→ FAILED (config error)
```

### Phases

**VALIDATING**: Control plane runs the job type's validation function. Schema check first (JSON Schema), then semantic pre-flight checks (model exists? dataset accessible? GPU type compatible?). Synchronous, fast (<5s).

**QUEUED**: Job is persisted and waiting for compute. Jobs can be reordered or cancelled here.

**PROVISIONING**: Compute backend is setting up the environment (SkyPilot launching a VM, Slurm allocating nodes, K8s creating a pod). Tracked separately from RUNNING so users know "waiting for a GPU" vs "training is running."

**RUNNING**: Container runner actively working, yielding events. Events are persisted to an append-only log.

**Terminal states**: SUCCEEDED (artifacts registered), FAILED (structured error with phase), CANCELLED (graceful shutdown signal).

### Resumability

Jobs that fail during RUNNING can be resumed if the container runner supports it:

```
POST /api/v1/jobs/{id}/resume
```

The control plane passes the last checkpoint artifact to the container runner. Checkpoints are just artifacts — explicit, typed, and tracked.

### Job Dependencies

Jobs can reference artifacts from other jobs:

```python
eval_job = client.submit("eval", config={
    "model": "artifact:job-abc/model",  # wait for job-abc to produce a model
    "benchmarks": ["mmlu"],
})
```

The control plane holds the job in QUEUED until referenced artifacts exist.

## API Design

### REST API

```
# Jobs
POST   /api/v1/jobs                    # Submit a job (dry_run=true by default)
GET    /api/v1/jobs                    # List jobs (filter by type, status, user)
GET    /api/v1/jobs/{id}               # Job detail + status + events
GET    /api/v1/jobs/{id}/logs          # Stream logs (SSE)
GET    /api/v1/jobs/{id}/events        # Structured event stream (SSE)
POST   /api/v1/jobs/{id}/cancel        # Cancel a job
POST   /api/v1/jobs/{id}/resume        # Resume from last checkpoint
DELETE /api/v1/jobs/{id}               # Delete job record

# Artifacts (datasets, models, checkpoints, results — all the same API)
POST   /api/v1/artifacts               # Upload file (multipart, <1GB) or register external reference
GET    /api/v1/artifacts/upload-url    # Pre-signed URL for large file upload (>1GB)
GET    /api/v1/artifacts               # List artifacts (filter by type, job_id)
GET    /api/v1/artifacts/{id}          # Artifact metadata + location
GET    /api/v1/artifacts/{id}/download # Pre-signed download URL
DELETE /api/v1/artifacts/{id}          # Delete artifact record

# Job Types
GET    /api/v1/job-types               # List available job types + descriptions
GET    /api/v1/job-types/{type}/schema # JSON Schema for this job type's config

# Compute
GET    /api/v1/compute                 # List backends + capabilities + status
GET    /api/v1/compute/{backend}/status # Backend health + running jobs

# Config
POST   /api/v1/config/validate         # Validate without submitting
GET    /api/v1/recipes                  # Browse bundled recipes
GET    /api/v1/recipes/{name}           # Get a specific recipe

# Admin (cloud lifecycle, separate from jobs)
POST   /api/v1/admin/clusters              # Provision a cluster
DELETE /api/v1/admin/clusters/{name}       # Tear down (requires confirmation)
POST   /api/v1/admin/clusters/{name}/stop  # Pause (SkyPilot only)
```

### Python SDK

```python
from amortized import Client

client = Client()  # auto-discovers local or remote server

# Submit a training job
job = client.submit(
    type="training",
    config={
        "algorithm": "lora_sft",
        "model_path": "meta-llama/Llama-3.1-8B",
        "data_path": "my-org/instructions",
        "num_epochs": 3,
        "lora_r": 16,
    },
    compute={"backend": "slurm", "gpus": 4, "gpu_type": "H100"},
)

job.wait()
job.stream_logs()
job.status  # SUCCEEDED

# Chain: eval the trained model
eval_job = client.submit(
    type="eval",
    config={
        "model": job.artifact_ref("model"),
        "judge_model": "openai/gpt-4o",
        "dataset": "my-org/eval-set",
    },
    compute={"backend": "slurm", "gpus": 1},
)

```

SDK auto-discovers the server: `AMORTIZED_API_URL` env var → `~/.amortized/config.yaml` → `localhost:9400`.

### MCP Tools (Agent Interface)

Auto-generated from the OpenAPI spec. No hand-written tool surface, no drift.

```
Tools:
  submit_job(type, config, compute)                 # POST /api/v1/jobs
  get_job(job_id)                                   # GET /api/v1/jobs/{id}
  cancel_job(job_id)                                # POST /api/v1/jobs/{id}/cancel
  list_jobs(status?, type?, limit?)                 # GET /api/v1/jobs
  get_logs(job_id, lines?)                          # GET /api/v1/jobs/{id}/logs
  list_job_types()                                  # GET /api/v1/job-types
  get_job_type_schema(type)                         # GET /api/v1/job-types/{type}/schema
  validate_config(type, config)                     # POST /api/v1/config/validate
  list_recipes(query?)                              # GET /api/v1/recipes
  get_recipe(name)                                  # GET /api/v1/recipes/{name}
  list_artifacts(type?, job_id?)                    # GET /api/v1/artifacts
  get_artifact_download_url(artifact_id)            # GET /api/v1/artifacts/{id}/download
  resume_job(job_id)                                # POST /api/v1/jobs/{id}/resume

Resources:
  system://capabilities     # job types, backends, storage
  recipes://{name}          # recipe configs with descriptions
  jobs://recent             # last 10 jobs (quick context)
```

Agents discover job type capabilities via JSON Schema from `get_job_type_schema()`. The schema IS the documentation — no "call get_started() first" pattern.

### Safety: Dry Run by Default

All job submissions default to `dry_run=true`. The response includes:
- Validation result
- Job type + compute resolution
- Estimated cost (if cloud)
- Command preview
- Confirm URL

Destructive operations (cluster teardown) require explicit confirmation fields at the API layer, not per-client.

## Container Runner System

### No Plugin System — Extensibility Lives Below

Amortized has 4 hardcoded job types. This is intentional, not a limitation. When someone adds a new training algorithm, they add it to Training Hub. When someone adds a new synthesis flow, they add it to Amortized Synth. Amortized's job types don't change.

The control plane knows each job type's:
- **JSON Schema** — loaded from `containers/{type}/schema.json` at startup
- **Validation function** — type-specific pre-flight checks (hardcoded in `server/core/jobs.py`)
- **Container image** — configured in the server config file

### RunContext (shared container utilities)

A shared Python module (`containers/shared/`) is copied into every container image at build time. It provides `RunContext`:

```python
@dataclass
class RunContext:
    job_id: str
    work_dir: Path                    # scratch directory for this job
    artifacts: dict[str, Artifact]    # resolved input artifacts (paths, not refs)
    emit: Callable[[Event], None]     # emits via stdout (JSON lines) + HTTP POST
    log: Logger                       # structured logger that auto-emits log events
    is_cancelled: Callable[[], bool]  # check for cancellation signal

    def save_artifact(self, name: str, path: Path) -> Artifact:
        """Upload artifact to configured shared storage (S3, NFS, HuggingFace)
        and emit an artifact event. Storage credentials come from env vars."""
```

### Event Transport

Containers communicate events back via two channels simultaneously:
- **Stdout** (structured JSON lines) — always works, even if the control plane is unreachable. The compute backend captures stdout and relays to the control plane.
- **HTTP POST** to `AMORTIZED_EVENTS_URL` — real-time delivery when the control plane is reachable.

The shared module handles both transparently. Runner scripts just call `context.emit(event)`.

### Artifact Transfer

Containers push artifacts to shared storage from inside the container. `context.save_artifact("model", local_path)` uploads to the configured storage backend (S3, NFS, HuggingFace — credentials via env vars) and emits an artifact event with the remote location. The control plane never touches the data — it only records the reference.

### Job Types and Container Images

Amortized has 4 known job types. Extensibility lives in the dispatch layers (Training Hub, Amortized Synth), not in Amortized. No plugin system needed at the Amortized level — when someone adds a new training backend, they add it to Training Hub. When someone adds a new synthesis flow, they add it to Amortized Synth. Amortized never changes.

```
Amortized Control Plane (4 job types, hardcoded)
  │
  ├── training   → container → Training Hub (dispatch layer)
  │                              ├── instructlab-training
  │                              ├── Unsloth
  │                              ├── verl
  │                              ├── OpenPipe ART
  │                              └── new backends added HERE, not in Amortized
  │
  ├── synth      → container → Amortized Synth (built-in synthesis engine)
  │                              ├── Dataset planning + attribute generation
  │                              ├── Multi-turn conversation synthesis (agentic loop)
  │                              ├── Attribute transformation
  │                              └── LiteLLM for inference
  │
  ├── inference  → container → vLLM (batch generation on remote GPU)
  │
  └── eval       → container → LLM-as-judge (calls frontier model APIs)
```

Each job type has:
- A JSON Schema (for API/agent validation)
- A validation function (semantic pre-flight checks)
- A container image (with the ML runtime)
- A runner script (reads config, calls the library, emits events)

These are hardcoded in the control plane — no entry points, no plugin discovery, no plugin protocol. Four job types, four container images, done. This is not Oumi's hardcoded-map problem because the extensibility lives one layer down in Training Hub and Amortized Synth, which have their own registry/plugin systems.

### Config Schemas

Each job type has a JSON Schema file (`containers/{type}/schema.json`). These schemas:
- Validate user configs before execution
- Document what each job type accepts (agents read schemas via `GET /api/v1/job-types/{type}/schema`)
- Drive the MCP tool parameter definitions

## Compute Backends

### Capability-Based Interface

Backends declare capabilities instead of implementing no-ops or raising `NotImplementedError`:

```python
class Capability(str, Enum):
    GPU = "gpu"
    MULTI_NODE = "multi_node"
    LOG_STREAM = "log_stream"
    STOP = "stop"
    RESUME = "resume"
```

The API exposes capabilities so clients and agents know what's possible:

```json
GET /api/v1/compute
[
  {"name": "slurm-cluster", "capabilities": ["GPU", "MULTI_NODE", "LOG_STREAM"], "gpus_available": 24},
  {"name": "gcp-sky", "capabilities": ["GPU", "MULTI_NODE", "STOP", "RESUME", "LOG_STREAM"]},
  {"name": "k8s-prod", "capabilities": ["GPU", "MULTI_NODE", "LOG_STREAM", "STOP"], "gpus_available": 16}
]
```

### JobSpec (control plane → backend)

```python
@dataclass
class JobSpec:
    job_id: str
    command: list[str]          # the actual command to run
    env: dict[str, str]         # environment variables
    resources: Resources        # GPUs, CPUs, memory
    work_dir: Path              # directory to sync/mount
    image: str                  # container image (always required — all jobs run in containers)
    timeout: int | None         # max runtime in seconds
```

The control plane translates user config into a JobSpec. The container image + a generated command become the job to run. Clean separation — containers don't know about Slurm, backends don't know about training. Each backend maps JobSpec to its native container runtime: Singularity/Apptainer for Slurm, K8s pods for Kubernetes, SkyPilot tasks for cloud.

### Slurm: One Implementation, Profiles for Variation

```python
class SlurmBackend:
    def __init__(self, ssh_host: str, profile: SlurmProfile = DEFAULT): ...

@dataclass
class SlurmProfile:
    module_loads: list[str] = field(default_factory=list)
    remote_base_dir: str = "~/jobs"
    partition: str | None = None
    account: str | None = None
    scheduler: str = "sbatch"         # "sbatch" or "qsub" for PBS
    gpu_resource_format: str = "gpu:{type}:{count}"

# Pre-built profiles
FRONTIER = SlurmProfile(module_loads=["module load rocm/6.0"], ...)
PERLMUTTER = SlurmProfile(module_loads=["module load cudatoolkit"], ...)
POLARIS = SlurmProfile(scheduler="qsub", ...)
```

One implementation. Adding a new HPC center = defining a new `SlurmProfile`, not copy-pasting 300 lines.

### Cloud Lifecycle (Separate from Jobs)

Provisioning and cluster management is an admin API, not part of job submission:

```
POST   /api/v1/admin/clusters              # provision
DELETE /api/v1/admin/clusters/{name}       # tear down (confirmation required)
POST   /api/v1/admin/clusters/{name}/stop  # pause
```

SkyPilot auto-provisions on job submit for most users. The admin API exists for cost management.

## Config System

### Three Layers, Cleanly Separated

```
┌─────────────────────────────────────┐
│  Job Envelope (control plane owns)  │
│  type, compute, metadata            │
├─────────────────────────────────────┤
│  Job Config (container runner owns) │
│  validated by job type JSON Schema  │
│  opaque to the control plane        │
├─────────────────────────────────────┤
│  Compute Spec (backend owns)        │
│  gpus, gpu_type, nodes, memory      │
└─────────────────────────────────────┘
```

No 200-field dataclass hierarchies. No 190-line `to_hf()` translation method. No 59% pass-through fields. The control plane has one config — `JobRequest` with ~6 fields. Everything else belongs to container runners.

### Recipes: Composable Overlays

```yaml
# recipes/base/lora-sft.yaml
type: training
config:
  algorithm: lora_sft
  num_epochs: 3
  learning_rate: 2e-5
  lora_r: 16
  lora_alpha: 32
  gradient_checkpointing: true
compute:
  gpus: 1

# recipes/llama3/8b-lora-sft.yaml
extends: base/lora-sft
config:
  model_path: meta-llama/Llama-3.1-8B
  model_max_length: 8192
compute:
  gpu_type: A100
```

`extends` is a simple dict merge — child overrides parent. Users customize the same way:

```python
job = client.submit_recipe("llama3/8b-lora-sft", overrides={
    "config.data_path": "my-org/my-data",
    "config.num_epochs": 5,
    "compute.gpus": 4,
})
```

### Validation: Three Levels

1. **Schema validation** — JSON Schema for the job type. Catches typos, wrong types, missing fields. Instant.
2. **Pre-flight validation** — per-job-type validation function in the control plane. Model exists? Dataset accessible? GPU compatible? Fast but may hit network.
3. **Runtime validation** — errors during container execution. CUDA OOM, library version issues. Become FAILED events.

## Data Model

### The Control Plane Doesn't Define ML Data Types

No framework-imposed `Conversation` type with 9 parallel serialization paths. The control plane deals in artifacts — typed references to data, not the data itself.

Conversation-format data lives inside container runners. Each runner uses whatever types its underlying library needs. No doomed unification attempt.

### Artifact Types

Artifact types are **open strings**, not a closed enum. The control plane does not enumerate or switch on them. Container runners define whatever types they produce. Common conventions:

```
model, dataset, checkpoint, eval_results, config, endpoint, report, logs
```

New artifact types can be introduced without framework changes.

### Artifact Storage

Containers push artifacts to shared storage from inside the container via `context.save_artifact()`. The control plane only records the reference — it never touches the data.

```yaml
storage:
  default: s3
  backends:
    s3: {bucket: ml-artifacts, prefix: amortized/}
    huggingface: {namespace: my-org}
    nfs: {mount_path: /shared/ml-artifacts}
```

## Data Flow

The control plane doesn't move data — it passes references to containers, which resolve them at runtime. Artifacts flow between jobs via shared storage.

### Data IN: Getting Data to Remote Containers

Three paths, all result in a reference the container resolves at runtime:

**1. HuggingFace Hub reference (most common)**

The container pulls the dataset from HF Hub at runtime. The user just provides a dataset name.

```python
job = client.submit(type="training", config={
    "model_path": "meta-llama/Llama-3.1-8B",       # pulled from HF at runtime
    "data_path": "my-org/instructions",              # pulled from HF at runtime
}, compute={"backend": "slurm", "gpus": 4})
```

The container needs HF credentials for private datasets — passed via `HF_TOKEN` env var from the control plane config.

**2. Remote storage path (S3/NFS)**

The data is already in storage accessible from the compute backend. The container pulls or mounts it directly.

```python
job = client.submit(type="training", config={
    "data_path": "s3://my-bucket/data/train.jsonl",
}, compute={"backend": "kubernetes", "gpus": 4})
```

**3. Local file upload (via API)**

The user has a file on their laptop. The control plane accepts the upload and stores it in configured shared storage.

```python
# Small files (<1GB) — proxy through control plane
dataset = client.upload("instructions.jsonl", type="dataset")
# SDK calls: POST /api/v1/artifacts (multipart file upload, type="dataset")
# Control plane writes to configured shared storage
# Returns artifact with location, e.g. "s3://bucket/artifacts/art_abc/instructions.jsonl"

# Large files (>1GB) — pre-signed URL, upload directly to storage
dataset = client.upload("big_dataset.jsonl", type="dataset")
# SDK calls: GET /api/v1/artifacts/upload-url (gets pre-signed S3 PUT URL)
# SDK uploads directly to S3, bypassing control plane
# Registers artifact after upload completes

# The SDK picks the right path based on file size — user doesn't think about it.

# Then submit the job with the uploaded dataset
job = client.submit(type="training", config={
    "data_path": dataset.location,
}, compute={"backend": "slurm", "gpus": 4})
```

```bash
# CLI equivalent
amortized upload --type dataset instructions.jsonl
# → Uploaded: s3://bucket/artifacts/art_abc/instructions.jsonl

amortized submit training --recipe llama3/8b-lora-sft \
    --set config.data_path=s3://bucket/artifacts/art_abc/instructions.jsonl
```

A dataset is just an artifact with `type="dataset"` and no `producer_job`. The same artifacts API handles uploads, downloads, listing, and references — no separate datasets API needed.

### Data OUT: Getting Artifacts Back

Containers push artifacts to shared storage via `context.save_artifact()`. The control plane records the reference. Users retrieve artifacts via the API.

```python
# After job completes
job = client.get_job("job_abc")
model = job.artifacts["model"]

# Option 1: get the storage location (S3 path, HF repo, etc.)
model.location     # "s3://bucket/artifacts/job_abc/model/"

# Option 2: download to local machine
model.download("./my-model/")
# SDK calls: GET /api/v1/artifacts/{id}/download
# Control plane returns pre-signed download URL
# SDK downloads from S3 directly

# Option 3: use as input to another job (artifact reference)
eval_job = client.submit(type="eval", config={
    "model": model.ref,    # "artifact:job_abc/model"
})
# Control plane resolves the reference to the S3 path before passing to container
```

### Artifact Download API

```
GET /api/v1/artifacts/{id}               # metadata + storage location
GET /api/v1/artifacts/{id}/download      # pre-signed download URL (redirect or URL in response)
```

### End-to-End Data Flow

```
User's machine                Control Plane              Shared Storage           GPU Container
──────────────                ─────────────              ──────────────           ─────────────

1. Upload dataset
   POST /artifacts ─────────► stores file ──────────────► s3://artifacts/art_abc/
   ◄── artifact ref ◄────────

2. Submit training job
   POST /jobs ───────────────► validates config
                               resolves data_path
                               dispatches to compute ──────────────────────────► starts container
                                                                                 pulls data from S3
                                                                                 pulls model from HF
                                                                                 runs training...
                                                                                 saves model ──────► s3://artifacts/job_abc/model/
                               ◄── artifact event ◄─── ◄── tracking event ◄─────

3. Submit eval job
   POST /jobs ───────────────► resolves artifact:job_abc/model
                               to s3://artifacts/job_abc/model/
                               dispatches to compute ──────────────────────────► starts container
                                                                                 pulls model from S3
                                                                                 runs evaluation...
                                                                                 saves results ────► s3://artifacts/job_def/eval/
                               ◄── artifact event ◄───

4. Download results
   GET /artifacts/{id}/download
   ◄── pre-signed URL ◄──────
   downloads from S3 ◄─────────────────────────────────── s3://artifacts/job_def/eval/
```

### What the Control Plane Needs for Data Flow

- **Shared storage credentials** in server config (same credentials passed to containers as env vars)
- **Upload endpoint** that writes to shared storage and returns artifact references
- **Pre-signed URL generation** for both upload (large files) and download (artifact retrieval)
- **Artifact reference resolution** — translates `artifact:job-xxx/model` to `s3://bucket/artifacts/job-xxx/model/` before passing config to containers

## Deployment Modes

### Local Mode (Default)

```bash
pip install amortized
amortized up                    # starts FastAPI on localhost:9400, SQLite in ~/.amortized/
amortized submit train --recipe llama3/8b-lora-sft --set config.data_path=my-data
amortized logs job_abc --follow
amortized list jobs --status running
```

The CLI is a REST client with nice formatting — not a separate code path. The control plane runs locally (no GPU needed) — all containers run on remote GPU instances via the configured compute backends.

### Server Mode (Teams)

Same binary, different config:

```yaml
server:
  host: 0.0.0.0
  port: 9400
auth:
  provider: oidc
  issuer: https://auth.company.com
storage:
  default: s3
  backends:
    s3: {bucket: ml-artifacts}
compute:
  backends:
    slurm: {ssh_host: cluster.company.com, profile: internal}
    skyPilot: {enabled: true}
mlflow:
  tracking_uri: https://mlflow.company.com
```

Team mode adds: auth (API key or OIDC), multi-user job filtering, shared compute credentials. Still SQLite in v1 — PostgreSQL swap available via repository pattern if scale demands it.

### MCP Connection

```bash
amortized mcp  # starts MCP stdio transport, proxies to API server
```

Same auto-discovery as the SDK. Works with local or remote servers.

### State Persistence

```
SQLite: ~/.amortized/amortized.db

Schema:
  jobs          → id, type, config, status, compute, timestamps, user, tracking_url
  events        → job_id, type, timestamp, data (append-only)
  artifacts     → id, type, name, location, producer_job, metadata
```

SQLite for v1. Repository pattern (`server/db/repository.py`) abstracts all SQL behind async methods, enabling a PostgreSQL swap later if scale demands it. No ORM — raw SQL keeps it simple and transparent.

## Project Layout

Monorepo with multiple independently-publishable packages. This matches the pattern used by Prefect (20 packages from one repo), Hatchet (consolidated from multi-repo), Determined (polyglot monorepo), and Ray. ClearML is the only comparable project using multi-repo, and that's for licensing reasons (SSPL server).

```
amortized/
├── server/                             # "amortized" on PyPI — the control plane
│   ├── pyproject.toml
│   └── src/amortized/
│       ├── core/                       # domain logic, no HTTP imports
│       │   ├── jobs.py                 # job state machine (server-side enforcement)
│       │   ├── job_types.py            # job type registry (4 types, schemas, validation)
│       │   ├── compute.py              # compute backend protocol + capabilities
│       │   ├── artifacts.py            # artifact tracking + storage references
│       │   ├── recipes.py              # recipe loading + extends composition
│       │   └── events.py              # event types + persistence
│       ├── api/                        # FastAPI routes (thin, calls core/)
│       │   ├── app.py                  # create_app(), lifespan, middleware
│       │   ├── jobs.py
│       │   ├── job_types.py
│       │   ├── artifacts.py            # upload, download, list, pre-signed URLs
│       │   ├── compute.py
│       │   ├── recipes.py
│       │   └── admin.py               # cluster lifecycle
│       ├── agent/                      # AI chat agent (OpenAI function-calling)
│       │   ├── chat.py
│       │   └── tools.py               # auto-generated from API
│       ├── mcp/                        # auto-generated MCP from OpenAPI
│       │   └── server.py
│       ├── db/                         # persistence layer
│       │   ├── schema.sql             # table definitions
│       │   └── repository.py          # async repository over aiosqlite (raw SQL)
│       ├── backends/                   # compute backend implementations
│       │   ├── slurm.py               # SlurmBackend + SlurmProfile
│       │   ├── skypilot.py            # SkyPilotBackend
│       │   └── kubernetes.py          # KubernetesBackend
│       ├── cli/                        # Typer CLI (thin REST client)
│       │   └── main.py
│       └── sdk/                        # Python SDK
│           └── client.py
│
├── containers/                         # 4 container images (NOT pip packages)
│   ├── shared/                         # shared Python module copied into each image
│   │   ├── context.py                  # config reading, RunContext
│   │   ├── events.py                   # event emission (stdout JSON lines + HTTP POST)
│   │   └── artifacts.py                # artifact upload to shared storage
│   ├── training/                       # training container
│   │   ├── Dockerfile                  # CUDA + training-hub + all backends
│   │   ├── runner.py                   # reads config → calls training_hub → emits events
│   │   └── schema.json                # JSON Schema for training config
│   ├── synth/                          # synthesis container
│   │   ├── Dockerfile                  # amortized-synth + litellm
│   │   ├── runner.py                   # reads config → calls amortized_synth → emits events
│   │   └── schema.json
│   ├── inference/                      # inference container
│   │   ├── Dockerfile                  # CUDA + vLLM
│   │   ├── runner.py                   # reads config → runs vLLM batch inference → emits events
│   │   └── schema.json
│   └── eval/                           # evaluation container
│       ├── Dockerfile                  # lightweight (just openai/litellm)
│       ├── runner.py                   # reads config → calls LLM-as-judge → emits events
│       └── schema.json
│
├── studio/                             # Next.js UI (dashboard + agent chat)
│   ├── package.json
│   └── src/
│       ├── app/                        # pages (jobs, flows, settings)
│       ├── components/                 # chat-panel, job forms, metrics charts
│       └── lib/                        # API client, chat store
│
├── recipes/                            # composable YAML configs
│   ├── base/                           # base recipes (lora-sft, grpo, sdg)
│   └── llama3/                         # model-specific (extends base)
│
├── docker/                             # docker-compose for dev/prod
│   ├── docker-compose.yaml
│   └── docker-compose.dev.yaml
│
├── docs/                               # specs, competitive analysis
└── tests/                              # integration/e2e tests spanning packages
```

### Packages

One pip package, one npm package, four container images:

| Component | Type | What it is | Dependencies |
|-----------|------|-----------|-------------|
| `server/` | PyPI: `amortized` | Control plane (API + CLI + SDK + compute backends) | FastAPI, aiosqlite, Pydantic, Typer, httpx, AsyncSSH |
| `studio/` | npm: `@amortized/studio` | Dashboard + agent chat UI | Next.js, React |
| `containers/training/` | Container image | Training runner | training-hub, CUDA |
| `containers/synth/` | Container image | Synthesis runner | amortized-synth, litellm |
| `containers/inference/` | Container image | Batch inference runner | vLLM, CUDA |
| `containers/eval/` | Container image | LLM-as-judge runner | openai/litellm |

External dispatch layers (not maintained in this repo):

| Library | Role | Extensibility |
|---------|------|---------------|
| Training Hub | Algorithm dispatch for training | New backends (Tinker, SkyRL) added here |
| Amortized Synth | Built-in synthesis engine (inspired by Oumi's architecture) | New synthesis strategies added here |

### CI Strategy (selective, path-filtered)

```yaml
# .github/workflows/
server.yml:        # triggers on server/ changes
  runs-on: ubuntu-latest            # standard runner, no GPU
  steps: pytest, ruff, pyright      # 2-3 minutes

studio.yml:        # triggers on studio/ changes
  runs-on: ubuntu-latest
  steps: npm test, tsc, eslint      # 1-2 minutes

container-training.yml:  # triggers on containers/training/ changes
  runs-on: [self-hosted, gpu]        # GPU runner
  steps: pytest, docker build        # 15-30 minutes

container-inference.yml: # triggers on containers/inference/ changes
  runs-on: [self-hosted, gpu]
  steps: pytest, docker build

recipes.yml:       # triggers on recipes/ changes
  runs-on: ubuntu-latest
  steps: validate YAML schemas      # seconds

integration.yml:   # triggers on any significant change, runs nightly
  runs-on: [self-hosted, gpu]
  steps: full e2e tests             # 30-60 minutes
```

**Why monorepo**: API contract changes (server ↔ SDK ↔ studio ↔ containers) are atomic — one PR, one review, one merge. Every comparable open-source project (Prefect, Hatchet, Determined, Ray) uses monorepo. Multi-repo exists only for licensing (ClearML's SSPL server) or K8s operators in different languages (Ray's KubeRay).

**Why selective CI**: A recipe YAML fix doesn't trigger a 30-minute GPU build. Server changes don't trigger container Docker builds. Each path has CI proportional to its complexity.

### Key Structural Decisions

- `server/core/` has no HTTP imports — pure domain logic, fully testable without a running server
- `server/api/` is a thin routing layer — calls core functions, handles serialization
- `server/backends/` contains compute implementations — sub-modules, not separate packages
- `containers/` are NOT pip packages — Docker images with runner scripts and shared utilities
- CLI and SDK are sub-modules of the server package — three facades (REST, SDK, CLI) over the same backend
- Studio lives in-repo — API contract coupling makes separate repos painful
- No plugin system in Amortized — 4 hardcoded job types. Extensibility lives in Training Hub and Amortized Synth
- One pip package (`amortized`) — the control plane. Everything else is container images

## Tech Stack

Informed by competitor research (Fireworks, Together, TML/Tinker, CoreWeave) and Python ML platform conventions (Prefect, Determined, MLflow).

### Server (control plane)

| Component | Technology | Why |
|-----------|-----------|-----|
| **API framework** | FastAPI | Industry standard for ML platforms. Prefect uses it. MLflow migrated from Flask to FastAPI. Auto-generates OpenAPI spec. |
| **Database** | SQLite + aiosqlite (v1) | Start simple. Repository pattern allows PostgreSQL swap later if scale demands it. No ORM — raw async SQL via aiosqlite. |
| **Data validation** | Pydantic v2 | Comes with FastAPI. Used by Fireworks, Together, TML SDKs. |
| **CLI** | Typer | Same author as FastAPI. Type-hint-driven, Rich integration. |
| **SDK HTTP client** | httpx | Async-native. Used by Fireworks SDK, TML/Tinker SDK. |
| **MCP generation** | fastapi-mcp | Auto-generates MCP server from FastAPI routes. Zero-config, ASGI transport. |
| **SSH (Slurm)** | AsyncSSH | Async-native, 15x faster than Paramiko for multi-host. |
| **Config format** | YAML (recipes) / JSON (API, container handoff) | YAML for humans, JSON internally. |

### Containers

| Container | Key dependencies |
|-----------|-----------------|
| training | training-hub, torch, CUDA |
| synth | amortized-synth (built-in), litellm |
| inference | vllm, torch, CUDA |
| eval | litellm (for LLM-as-judge API calls) |

### Studio (frontend)

| Component | Technology | Why |
|-----------|-----------|-----|
| **Framework** | Next.js 15 + React 19 | Used by Together, TML. SSR + API routes. |
| **Language** | TypeScript | Type safety across API contract. |
| **Styling** | Tailwind CSS | Standard for ML dashboards. |
| **Charts** | Recharts | Lightweight, React-native. |
| **Streaming** | SSE (Server-Sent Events) | For live job events and agent chat. |

### Dev Tooling

| Tool | Purpose |
|------|---------|
| **uv** | Package management (used by TML/Tinker, fast) |
| **Ruff** | Linting + formatting (used by TML, replaces black + isort + flake8) |
| **pyright** | Type checking |
| **pytest** | Testing |
| **Docker + BuildKit** | Container image builds (dev) |
| **Kaniko** | Container builds in CI/K8s (unprivileged) |

### What We Don't Need

| Tool | Why not |
|------|---------|
| **Celery / ARQ / Redis** | No in-process job queue. Jobs dispatch to remote compute backends. Background work (status polling, heartbeats) is asyncio tasks. |
| **SQLAlchemy / ORM** | Raw async SQL over aiosqlite is sufficient for v1. Repository pattern enables PostgreSQL swap later without an ORM layer. |
| **gRPC / protobuf** | REST + JSON is sufficient for a control plane. gRPC adds complexity without benefit at our scale. |
| **Stainless SDK gen** | Acquired by Anthropic, shutting down. Hand-written SDK with httpx is fine for one language. |
| **OmegaConf** | Only needed inside Amortized Synth (keeps Oumi's config pattern). Control plane uses Pydantic. |

## Observability

### Experiment Tracking: MLflow (via Training Hub)

Training Hub reports to MLflow natively. No custom integration needed — the training container inherits this behavior.

```
Control plane config:
  mlflow:
    tracking_uri: https://mlflow.company.com
    tracking_token: ${MLFLOW_TRACKING_TOKEN}
```

The control plane passes MLflow env vars to containers:
- `MLFLOW_TRACKING_URI` — where to log experiments
- `MLFLOW_TRACKING_TOKEN` — auth token
- `MLFLOW_EXPERIMENT_NAME` — auto-set to job ID or user-specified name

After the training container starts, Training Hub logs metrics to MLflow. The container runner emits a one-time `tracking` event with the MLflow run URL. The control plane stores it as job metadata:

```json
GET /api/v1/jobs/{id}
{
  "id": "job_abc",
  "type": "training",
  "status": "running",
  "tracking_url": "https://mlflow.company.com/#/experiments/1/runs/abc123",
  ...
}
```

### Job Type Tracking Coverage

| Job type | Experiment tracker | What gets tracked |
|----------|-------------------|-------------------|
| training | MLflow (via Training Hub) | Loss, learning rate, gradient norms, epochs, checkpoints |
| synth | MLflow (optional) | Generation counts, token usage, quality metrics |
| inference | None | Results are artifacts (output dataset) |
| eval | None | Results are artifacts (scores, judge outputs) |

### Control Plane Observability

For monitoring the control plane itself (not ML experiments):
- **Structured logging** — JSON logs via Python stdlib logging
- **Health endpoint** — `GET /api/v1/health` for load balancers
- **Metrics** — optional Prometheus endpoint for job throughput, API latency, compute backend status (future work, not v1)

## Design Decisions vs Oumi

| Concern | Oumi | Amortized |
|---------|------|-----------|
| **Architecture** | Python library with CLI | API-first job orchestration control plane |
| **Extensibility** | Asymmetric: registry for datasets, hardcoded maps for engines/trainers (4 files to add an engine) | Extensibility lives in dispatch layers (Training Hub, Amortized Synth), not in Amortized. New training backends added to Training Hub. New synthesis flows added to Amortized Synth. Amortized has 4 known job types. |
| **Config** | 200+ field dataclass hierarchy, 59% HF pass-through, 190-line to_hf() | Container runner owns its JSON Schema, control plane has ~6 fields. Training params owned by Training Hub, synth config inspired by Oumi. |
| **Type awareness** | Control plane knows about training, inference, eval as distinct concepts with different code paths | Control plane is type-agnostic. Train, eval, infer, serve, synth — all just "jobs." Container runners define the types. |
| **Data type** | Conversation with 9 serialization paths, leaky to_dict() fixup | Artifacts (typed references). Container runners own their wire formats. |
| **Metrics** | Built-in WandB/MLflow/TensorBoard integration in training loop | Metrics go to W&B/MLflow directly from containers. Control plane captures tracking URL only. |
| **Compute** | Launcher class (6/8 methods are pass-throughs), 4 copy-pasted Slurm clusters | Capability-based backends, one Slurm impl with profiles |
| **Agent interface** | Bolted-on MCP server (958 lines, own job registry) | Auto-generated MCP from OpenAPI, same backend |
| **State** | Stateless launcher + ad-hoc JSON files | SQLite/PostgreSQL with append-only event log |
| **Recipes** | Standalone YAMLs with 50-60% boilerplate | Composable overlays with `extends` |
| **Dependencies** | Eager import of 102 files on registry init | Control plane has zero ML deps. ML libraries live in container images. |
| **Deployment** | CLI only | Hybrid: local CLI + team server, same binary |
