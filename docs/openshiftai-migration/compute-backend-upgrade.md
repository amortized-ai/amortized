# Compute Backend Upgrade Plan

Deep comparison of Granite.Build's environment abstraction vs amortized's compute backends, with a concrete upgrade plan.

## Current State — Amortized

### ComputeBackend Protocol (5 methods)
```python
class ComputeBackend(Protocol):
    name: str
    def capabilities(self) -> set[Capability]: ...
    async def submit(self, spec: JobSpec) -> BackendHandle: ...
    async def status(self, handle: BackendHandle) -> BackendStatus: ...
    async def cancel(self, handle: BackendHandle) -> None: ...
    def logs(self, handle: BackendHandle) -> AsyncIterator[str]: ...
```

### Backends: Only 2 exist
- **Local** — subprocess via `Popen`, stdout/stderr to files
- **SSH** — asyncssh to remote node, runs podman/docker containers

### Key Limitations
1. No lifecycle hooks beyond submit/status/cancel/logs
2. No cleanup method in protocol (SSH has `cleanup_secrets` but it's non-protocol)
3. No retry/resilience
4. No provisioning abstraction (SSH assumes node exists)
5. Worker hardcodes the entire lifecycle (script generation, SFTP download, artifact registration)
6. K8s, SkyPilot, Slurm backends don't exist despite being planned
7. Single event channel: HTTP callback (`AMORTIZED_EVENTS_URL`) + poll
8. Artifact transport: SFTP only (won't work for cloud backends)

---

## What Granite.Build Does Differently

### Environment Lifecycle (5 phases, async-coordinated)

```
setup(setup_id)              ← Create env-level resources (K8s secrets, shared workdir)
  │                            Blocked by: nothing
  │                            Signals: __setup_done_events[setup_id]
  │
  └─► launch(launch_id)     ← Submit workload (container, pod, cluster)
        │                      Blocked by: setup done, teardown NOT started
        │                      Must call: _release_monitors(launch_id) when ready
        │                      Signals: __launch_done_events[launch_id]
        │
        ├─► monitor(launch_id)  ← Watch logs/status
        │                         Blocked by: _release_monitors() or launch failure
        │                         Signals: __launch_stopped_events[launch_id]
        │
        └─► cleanup(launch_id)  ← Kill/remove workload
                                   Blocked by: launch done
                                   Signals: __cleanup_done_events[launch_id]

teardown(setup_id)           ← Delete env-level resources
                               Blocked by: ALL launches done, ALL cleanups done
```

Key coordination: `asyncio.Event` per phase per ID. Monitors don't start until launch signals ready. Cleanup waits for launch. Teardown waits for everything.

### Suffix-Based Dispatch
Methods named `{verb}_{suffix}` are auto-discovered at `__init__`. The suffix comes from step.yaml config:
- `launch_docker`, `launch_helm`, `launch_nohup`, `launch_runpod`, `launch_skypilot`
- `monitor_docker_log`, `monitor_sidecar_monitor`, `monitor_log_monitor`
- `pullasset_filestore`, `pullasset_hfstore`, `pullasset_cosstore`

### Event Queue
Single `asyncio.Queue` carries typed `BuildEvent` objects with:
- `BuildEventType` enum (STATUS, ARTIFACT, MESSAGE, METRICS, WORKLOAD_STATUS, TERMINATE)
- `EntityRunMetadata` (build_id, target_name, step, etc.)
- Typed payloads per event type

### Log-Based Event Extraction
`EventLogLineParserConfig` defines regex patterns to parse stdout lines into structured events. Training frameworks emit lines like `step=100 loss=0.45` and the regex parser extracts metrics events without modifying the training code.

### Retry Framework
`RetryHandler` wraps the monitor event queue:
```
Monitor → wrapper_queue → RetryHandler → downstream_queue
```
On failure event: evaluate strategies → extract nodes to avoid → backoff delay → `environment.retry_workload()` → tear down old, launch new with anti-affinity.

Built-in strategies: NCCL error, pod eviction, provision failure, file not found, Aspera failure, any-failure catchall.

### Asset Store Abstraction
Pluggable `pullasset_*` / `pushasset_*` per store type:
- `filestore` — local filesystem / sync
- `hfstore` — HuggingFace Hub (download/upload)
- `cosstore` — S3/COS (rclone-based)
- `envstore` — shared filesystem (no-op)
- `lhstore` — IBM Lakehouse

### Per-Backend Implementation Size
| Backend | Lines | Complexity |
|---------|-------|-----------|
| Bash | ~290 | Simple subprocess |
| Docker | ~450 | Container lifecycle + volume mounts + GPU |
| RunPod | ~250 | API-based pod provisioning |
| SkyPilot | ~1430 | Multi-cloud + retry + concurrency control |
| SkyPilot Managed | ~350 | Delegates monitoring to SkyPilot controller |
| K8s | ~2732 | Helm + AppWrapper + 4 monitor types + retry |
| Base class | ~1391 | Lifecycle coordination + events + retry framework |

---

## Upgrade Plan for Amortized

### Phase 1: Enrich the Protocol

Expand `ComputeBackend` from 5 methods to ~10, adding lifecycle hooks and optional retry/artifact methods:

```python
class ComputeBackend(Protocol):
    name: str

    def capabilities(self) -> set[Capability]: ...

    # Lifecycle (required)
    async def submit(self, spec: JobSpec) -> BackendHandle: ...
    async def status(self, handle: BackendHandle) -> BackendStatus: ...
    async def cancel(self, handle: BackendHandle) -> None: ...
    async def cleanup(self, handle: BackendHandle) -> None: ...
    def logs(self, handle: BackendHandle) -> AsyncIterator[str]: ...

    # Lifecycle (optional — have default no-op implementations)
    async def setup(self, config: BackendConfig) -> None: ...
    async def teardown(self) -> None: ...

    # Retry (optional — backends that support it override)
    def retry_strategies(self) -> list[RetryStrategy]: ...
    async def retry(self, handle: BackendHandle, failure: FailureInfo) -> BackendHandle: ...

    # Artifacts (optional — backends with non-SFTP transport override)
    async def fetch_outputs(self, handle: BackendHandle, local_dir: str) -> None: ...
    async def push_inputs(self, handle: BackendHandle, local_dir: str) -> None: ...
```

### Phase 2: Extract Worker Logic into Protocol Methods

Currently the worker hardcodes: script generation, SFTP download, secret management, artifact registration. Move these into the backend protocol:

| Currently in Worker | Move to Backend |
|---|---|
| `_fetch_remote_outputs()` (SFTP) | `fetch_outputs()` |
| Secret creation via `podman secret create` | `setup()` |
| Secret cleanup via `podman secret rm` | `cleanup()` |
| Container runtime selection (podman vs docker) | Backend config |
| Script generation (`_trl_trainer_script`) | Keep in worker (backend-agnostic) |

### Phase 3: Add New Backends

Priority order (ignoring LSF and SLURM per user request):

#### 3a. Docker Backend (local GPU development)
Based on Granite.Build's Docker env (~450 lines). For users with local GPUs who don't want SSH.

```python
class DockerBackend:
    # Uses docker Python SDK (or podman-compatible API)
    # GPU: docker.types.DeviceRequest(count=N, capabilities=[["gpu"]])
    # Volumes: bind-mount work dir
    # Logs: container.logs(stream=True, follow=True) with thread-queue bridge
    # Cleanup: container.stop() + container.remove()
```

Key patterns from Granite.Build:
- Docker/Podman auto-detection (try `docker.from_env()`, fallback to podman socket)
- Pull policy support (always, if-not-present, never)
- Thread-safe log streaming via queue bridge (avoid async generator issues)

#### 3b. RunPod Backend (cloud GPU on-demand)
Based on Granite.Build's RunPod env (~250 lines). Simplest cloud backend.

```python
class RunPodBackend:
    # Uses runpod Python SDK
    # Provision: runpod.create_pod(name, image, gpu_type, gpu_count, env)
    # Status: runpod.get_pod(pod_id) → check desiredStatus
    # Cleanup: runpod.terminate_pod(pod_id)
    # GPU map: A100-80GB, H100-80GB, L40S, RTX-4090, etc.
    # Artifacts: via S3 (RunPod doesn't have built-in storage)
```

Key patterns from Granite.Build:
- GPU type normalization (short names → RunPod native IDs)
- Exponential backoff polling for pod readiness (5s → 30s, 600s timeout)
- No spot support (use SECURE cloud_type)

#### 3c. SkyPilot Backend (multi-cloud K8s, AWS, GCP)
Based on Granite.Build's SkyPilot env (~1430 lines, but we'd start simpler). This is the most impactful — one backend covers K8s + AWS + GCP.

```python
class SkyPilotBackend:
    # Uses sky Python SDK
    # Provision: sky.launch(task, cluster_name, idle_minutes_to_autostop)
    # Status: sky.job_status(cluster_name, job_ids)
    # Cleanup: sky.down(cluster_name, purge=True)
    # Task spec: sky.Task(name, setup, run, envs) + sky.Resources(accelerators, ...)
    # Multi-cloud: config.default_cloud (kubernetes, aws, gcp)
    # Artifacts: HF Hub or S3 via sky.Storage mounts
```

Key patterns from Granite.Build:
- Concurrency semaphore (prevent SSH MaxAuthTries failures during fan-out)
- Provision retry with tenacity (transient resource errors only)
- Spot/preemption recovery: tear down old cluster → create fresh with unique name
- SkyPilot Managed mode: delegate monitoring to SkyPilot controller (server doesn't need to stay alive)

### Phase 4: Retry Framework

Add a simple retry system inspired by Granite.Build but without the enterprise complexity:

```python
class RetryStrategy(Protocol):
    def should_retry(self, failure: FailureInfo) -> bool: ...
    def backoff_seconds(self, attempt: int) -> float: ...

class ContainerCrashRetry(RetryStrategy):
    """Retry on non-zero exit code (OOM, segfault)"""

class ProvisionRetry(RetryStrategy):
    """Retry on resource unavailable (cloud backends)"""

class PreemptionRetry(RetryStrategy):
    """Retry on spot/preemptible instance eviction"""
```

The worker's heartbeat monitor (300s timeout → mark failed) becomes a strategy rather than the only option.

### Phase 5: Artifact Transport Abstraction

Replace hardcoded SFTP with pluggable transport:

```python
class ArtifactTransport(Protocol):
    async def push(self, local_path: str, remote_uri: str) -> None: ...
    async def pull(self, remote_uri: str, local_path: str) -> None: ...

class SFTPTransport(ArtifactTransport): ...      # Current (SSH backend)
class S3Transport(ArtifactTransport): ...         # Cloud backends
class HFHubTransport(ArtifactTransport): ...      # Model artifacts
class LocalTransport(ArtifactTransport): ...      # Local/Docker backend
```

Each backend declares its transport. The worker uses `backend.fetch_outputs()` instead of hardcoded SFTP.

---

## What NOT to Adopt from Granite.Build

1. **Suffix-based dispatch** — auto-discovering methods by name prefix is clever but hard to follow. A simple class hierarchy is clearer for amortized's scale.

2. **Helm charts + AppWrapper + Kueue** — enterprise K8s complexity. Use SkyPilot as the K8s abstraction instead of building a 2700-line direct K8s integration.

3. **RabbitMQ sidecar** — enterprise messaging. HTTP event ingest is simpler and sufficient.

4. **Convention-based step configuration** — Granite.Build's step.yaml declares which launcher/monitor suffixes to use. Amortized's approach (backend type in config) is simpler.

5. **Thread-local environment caching** — premature optimization. One backend instance per backend config is fine.

6. **Multi-step DAG orchestration** — Granite.Build chains steps within a build. Amortized submits individual jobs. Pipeline orchestration is a separate concern (future `pipeline.yaml` feature).

---

## Implementation Priority

| # | Task | Effort | Impact |
|---|---|---|---|
| 1 | Add `cleanup()` to protocol + SSH implementation | Small | Fixes secret leak |
| 2 | Add `fetch_outputs()` to protocol, move SFTP logic from worker | Small | Cleaner separation |
| 3 | Docker backend | Medium | Local GPU dev without SSH |
| 4 | RunPod backend | Medium | Cloud GPU on-demand |
| 5 | SkyPilot backend | Large | Multi-cloud (K8s, AWS, GCP) |
| 6 | Retry framework | Medium | Production resilience |
| 7 | Artifact transport abstraction | Medium | Required for cloud backends |

Items 1-2 are cleanup. Items 3-5 are new backends. Items 6-7 are infrastructure for production use.

---

## References

- Granite.Build environments: `src/gbserver/environment/` (7 backends, ~6500 lines total)
- Granite.Build base class: `src/gbserver/environment/environment.py` (~1391 lines)
- Granite.Build retry: `src/gbserver/resilience/retry_handler.py`
- Granite.Build events: `src/gbserver/types/buildevent.py`
- Amortized backends: `src/amortized/backends/` (2 backends, ~500 lines total)
- Amortized worker: `src/amortized/worker.py`
