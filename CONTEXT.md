# Amortized — Domain Glossary

## Core Concepts

- **Job** — the universal unit of work. 4 hardcoded types: training, synth, inference, eval. Each has a container image, JSON Schema, and validation function in the control plane.
- **Container runner** — each job type has a container image with a runner script that calls the underlying library (Training Hub, Amortized Synth, vLLM, or judge). A shared Python module (`containers/shared/`) provides common utilities (config reading, event emission, artifact upload). These are NOT pip packages — just Docker images.
- **No plugin system** — Amortized has 4 hardcoded job types (training, synth, inference, eval). Extensibility lives in the dispatch layers (Training Hub, Amortized Synth), not in Amortized. This is intentional — when someone adds a new training backend, they add it to Training Hub, not Amortized.
- **Compute Backend** — where jobs physically execute. Launches containers. Declares capabilities (GPU, multi-node, log streaming, etc.).
- **Artifact** — a typed reference to data that flows between jobs. Not the data itself — a pointer with metadata (type, location, producer job). Artifact types are open strings (not enums).
- **Event** — a structured lifecycle message yielded by a running container. Types: state_change, artifact, error, progress, heartbeat. Persisted to an append-only log. Events are about job lifecycle, NOT training metrics — detailed metrics go to MLflow directly from containers.
- **Recipe** — a composable YAML config overlay. Recipes can extend other recipes via `extends` with simple dict merge.
- **Artifact transfer** — containers push artifacts to shared storage (S3, NFS, HuggingFace). The shared module provides `context.save_artifact(name, path)` which uploads and emits the artifact event. The control plane never touches the data — it only records the reference.

## Execution Model

- **Containers always** — all jobs run in containers on remote GPU instances. Slurm uses Singularity/Apptainer, K8s uses pods, SkyPilot provisions cloud VMs. There is no local compute backend — the control plane runs locally, containers run remotely.
- **Dev mode** — for container developers iterating on code: `amortized submit --dev` runs the container with source code mounted as a volume on the remote instance. No image rebuild needed.
- **Event transport** — containers communicate back via both stdout (structured JSON lines, always works) and HTTP POST to `AMORTIZED_EVENTS_URL` (real-time when available). The shared module handles both transparently.
- **Multi-node training** — compute backends handle multi-container orchestration natively (K8s PyTorchJob, Slurm `sbatch --nodes`, SkyPilot multi-node). The container is identical — it reads rank/world_size from standard env vars. The `Resources` spec has a `nodes` field.
- **Heartbeat** — shared module emits automatic heartbeat events every 60s. If the control plane sees no events past a configurable timeout (default 5min), it proactively checks the backend. On unclean container death, the job moves to FAILED with structured error (exit code, signal). Checkpoint artifacts emitted before death are preserved for resume.
- **Experiment tracking** — detailed metrics go to MLflow directly from containers (Training Hub reports to MLflow natively). The control plane only captures a one-time `tracking` event with the MLflow run URL, stored as job metadata.
- **Container handoff** — job config is a JSON file mounted at `/amortized/config.json`. Infrastructure concerns (job ID, event URL, storage credentials, tracking keys) are env vars. The shared module bootstraps `RunContext` from both via `RunContext.from_environment()`.
- **Compute credentials** — control plane uses existing credential mechanisms on disk (SSH keys, `~/.aws/credentials`, `~/.kube/config`). Standard tooling sets up credentials as normal. No built-in secrets manager in v1.

## Dispatch Layers

- **Training Hub** — external library (Red Hat AI Innovation Team) that serves as the algorithm dispatch layer for training. Maps (algorithm, backend) pairs to training engines (instructlab-training, Unsloth, verl, ART). New training backends (Tinker, SkyRL) are added to Training Hub, not to Amortized. The training container is a thin wrapper that calls Training Hub.
- **Amortized Synth** — built-in synthesis engine, inspired by Oumi's architecture. Reimplemented from scratch. Handles dataset planning, attribute generation, multi-turn conversation synthesis (agentic tool-call loops), and attribute transformation. Uses LiteLLM for inference.
- **Why not SDG Hub for synth?** — SDG Hub is a DataFrame block-chain pipeline framework (good for simple generate/filter/transform). Amortized Synth is a purpose-built conversation synthesis engine with multi-round agentic loops, batched turn-by-turn generation, and straggler handling. Fundamentally different execution models.
