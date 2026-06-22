> **SUPERSEDED** — Implementation is complete. See `prd-openshift.md` and `deployment-guide.md`. This doc is kept for historical reference.

# ROSA Implementation Plan — Deploying Amortized on OpenShift

Based on deep scans of all 3 repos + ROSA cluster inventory.

## Current State Summary

### Amortized Server
- **DB**: Raw SQLite via aiosqlite (3 tables: jobs, events, artifacts). WAL mode.
- **Scheduler**: In-process asyncio task. In-memory `_running` dict — lost on restart.
- **Backends**: SSH (fully implemented), K8s (stub — NotImplementedError), SkyPilot (stub), LocalStub (test).
- **Config**: File-based `~/.amortized/config.yaml`.
- **Port**: 9400 (default).
- **Container dispatch**: Builds JobSpec → SSH to remote host → docker/podman run. SCP for file sync.
- **Events**: Containers POST to `AMORTIZED_EVENTS_URL`. Server stores in SQLite, streams via SSE.
- **Auth**: None.
- **MCP**: fastapi-mcp auto-generates tools from OpenAPI spec.

### asynth (SDG)
- **Runtime**: Pure Python, no torch. ~100MB installed.
- **Container**: `python:3.11-slim + pip install asynth`. No entrypoint, no config mount.
- **Config**: `SynthesisConfig.from_dict()` works. YAML loadable.
- **API keys**: litellm reads from env vars (OPENAI_API_KEY, etc.).
- **GPU**: Not needed.

### Studio (Frontend)
- **Build**: Vite 8 + React 19. Output to `dist/`.
- **API client**: Relative paths (`/api/v1/...`). `VITE_API_URL` build-time only.
- **WebSocket**: Auto-derived from `window.location`. Path `/api/v1/ws`.
- **Dockerfile**: Does not exist.
- **Production**: Needs nginx + SPA fallback + `/api/` reverse proxy.

### ROSA Cluster
- **GPUs**: 4x NVIDIA L40S (48GB each) on g6e.4xlarge nodes.
- **RHOAI 3.3.1**: KServe, Model Registry, Training Operator, KubeRay all running.
- **Running**: Model Registry is already deployed as `default-modelregistry` (Kubeflow Hub).
- **Missing**: MLflow (not deployed), Kueue (not installed), S3 (not configured).
- **Storage**: gp3-csi (EBS) as default StorageClass.

---

## Minimum Viable Deployment (Phase 1)

Goal: Get amortized running on ROSA, submit a training job to an L40S GPU, see results in Studio.

### What we build vs defer

| Component | Phase 1 (MVP) | Deferred |
|---|---|---|
| **Server DB** | SQLite with PVC (single-replica is fine for testing) | PostgreSQL migration |
| **Storage** | PVC (EBS volume) for artifacts | S3/MLflow artifact store |
| **Compute backend** | K8s Jobs (implement KubernetesBackend) | Kueue, SkyPilot |
| **Model serving** | Raw vLLM pod (existing pattern) | KServe InferenceService |
| **Experiment tracking** | `report_to: none` initially | MLflow integration |
| **Auth** | Bearer token (existing) | OpenShift OAuth |
| **Studio** | nginx + reverse proxy | OAuth proxy sidecar |
| **Config** | ConfigMap + Secret | Auto-discovery from RHOAI |

### 1.1 Studio Production Dockerfile

Create `Dockerfile` and `nginx.conf.template` in the studio repo:

```dockerfile
# studio/Dockerfile
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 8080
```

```nginx
# studio/nginx.conf.template
server {
    listen 8080;

    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://${BACKEND_HOST}:${BACKEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

`nginxinc/nginx-unprivileged` runs as non-root (required by OpenShift's default SCC).

**Env vars** (runtime, set in Deployment):
- `BACKEND_HOST`: amortized server service name (e.g., `amortized-server`)
- `BACKEND_PORT`: `9400`

### 1.2 Amortized Server Dockerfile

Create `Dockerfile` in amortized repo root:

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install server
COPY server/ server/
COPY recipes/ recipes/
COPY examples/ examples/
COPY containers/ containers/
RUN pip install --no-cache-dir ./server

# Non-root user (OpenShift compatibility)
RUN useradd -m -u 1001 amortized
USER 1001

# Data directory (mount PVC here)
RUN mkdir -p /data
VOLUME /data

ENV AMORTIZED_DB_PATH=/data/amortized.db
ENV AMORTIZED_STORAGE_PATH=/data/artifacts
ENV AMORTIZED_RECIPES_PATH=/app/recipes
ENV AMORTIZED_EXAMPLES_PATH=/app/examples

EXPOSE 9400
CMD ["uvicorn", "amortized.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "9400"]
```

### 1.3 KubernetesBackend Implementation

The critical code change. Replace the stub with a working implementation.

```python
# backends/kubernetes.py
from kubernetes_asyncio import client, config, watch
from kubernetes_asyncio.client import ApiException

class KubernetesBackend:
    name: str

    def __init__(self, name: str, namespace: str = "amortized-jobs",
                 image_registry: str = "ghcr.io/amortized-ai"):
        self.name = name
        self.namespace = namespace
        self.image_registry = image_registry
        self._batch_api = None
        self._core_api = None

    async def _ensure_client(self):
        if self._batch_api is None:
            config.load_incluster_config()  # uses ServiceAccount token
            self._batch_api = client.BatchV1Api()
            self._core_api = client.CoreV1Api()

    def capabilities(self) -> set:
        return {Capability.GPU, Capability.LOG_STREAM, Capability.STOP}

    async def submit(self, spec: JobSpec) -> BackendHandle:
        await self._ensure_client()

        job_name = f"amortized-{spec.job_id[:8]}"

        # Map job type to container image
        image = f"{self.image_registry}/{spec.image}"

        # Build K8s Job manifest
        k8s_job = client.V1Job(
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={"app": "amortized", "job-id": spec.job_id},
            ),
            spec=client.V1JobSpec(
                backoff_limit=0,  # no K8s-level retry
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": "amortized", "job-id": spec.job_id},
                    ),
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        containers=[
                            client.V1Container(
                                name="worker",
                                image=image,
                                command=spec.command,
                                env=[client.V1EnvVar(name=k, value=str(v))
                                     for k, v in spec.env.items()],
                                resources=client.V1ResourceRequirements(
                                    limits={"nvidia.com/gpu": str(spec.resources.gpus)}
                                    if spec.resources.gpus else {},
                                    requests={"nvidia.com/gpu": str(spec.resources.gpus)}
                                    if spec.resources.gpus else {},
                                ),
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name="work",
                                        mount_path="/amortized/work",
                                    ),
                                ],
                            ),
                        ],
                        volumes=[
                            client.V1Volume(
                                name="work",
                                empty_dir=client.V1EmptyDirVolumeSource(),
                            ),
                        ],
                        # Schedule on GPU nodes
                        node_selector={"nvidia.com/gpu.present": "true"},
                    ),
                ),
            ),
        )

        await self._batch_api.create_namespaced_job(
            namespace=self.namespace, body=k8s_job
        )

        return BackendHandle(
            backend_name=self.name,
            handle_id=job_name,
            metadata={"namespace": self.namespace},
        )

    async def status(self, handle: BackendHandle) -> BackendStatus:
        await self._ensure_client()
        try:
            job = await self._batch_api.read_namespaced_job(
                name=handle.handle_id,
                namespace=handle.metadata["namespace"],
            )
            if job.status.succeeded:
                return BackendStatus(state="succeeded")
            if job.status.failed:
                return BackendStatus(state="failed", error="Job failed")
            return BackendStatus(state="running")
        except ApiException as e:
            if e.status == 404:
                return BackendStatus(state="failed", error="Job not found")
            raise

    async def cancel(self, handle: BackendHandle) -> None:
        await self._ensure_client()
        await self._batch_api.delete_namespaced_job(
            name=handle.handle_id,
            namespace=handle.metadata["namespace"],
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    async def logs(self, handle: BackendHandle) -> AsyncIterator[str]:
        await self._ensure_client()
        # Find the pod for this job
        pods = await self._core_api.list_namespaced_pod(
            namespace=handle.metadata["namespace"],
            label_selector=f"job-name={handle.handle_id}",
        )
        if not pods.items:
            return
        pod_name = pods.items[0].metadata.name

        # Stream logs
        w = watch.Watch()
        async for line in w.stream(
            self._core_api.read_namespaced_pod_log,
            name=pod_name,
            namespace=handle.metadata["namespace"],
            follow=True,
        ):
            yield line

    async def cleanup(self, handle: BackendHandle) -> None:
        try:
            await self.cancel(handle)
        except ApiException:
            pass
```

### 1.4 Config via Environment Variables

Replace `~/.amortized/config.yaml` dependency with env vars for K8s deployment:

```python
# In config or app startup
COMPUTE_BACKEND = os.environ.get("AMORTIZED_COMPUTE_BACKEND", "local")
COMPUTE_NAMESPACE = os.environ.get("AMORTIZED_COMPUTE_NAMESPACE", "amortized-jobs")
IMAGE_REGISTRY = os.environ.get("AMORTIZED_IMAGE_REGISTRY", "ghcr.io/amortized-ai")
```

The config.yaml path still works for local dev. On K8s, env vars from ConfigMap take precedence.

### 1.5 K8s Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: amortized
---
apiVersion: v1
kind: Namespace
metadata:
  name: amortized-jobs

---
# k8s/server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: amortized-server
  namespace: amortized
spec:
  replicas: 1
  selector:
    matchLabels:
      app: amortized-server
  template:
    metadata:
      labels:
        app: amortized-server
    spec:
      serviceAccountName: amortized-server
      containers:
        - name: server
          image: ghcr.io/amortized-ai/amortized:latest
          ports:
            - containerPort: 9400
          env:
            - name: AMORTIZED_COMPUTE_BACKEND
              value: kubernetes
            - name: AMORTIZED_COMPUTE_NAMESPACE
              value: amortized-jobs
            - name: AMORTIZED_IMAGE_REGISTRY
              value: ghcr.io/amortized-ai
            - name: AMORTIZED_DB_PATH
              value: /data/amortized.db
            - name: AMORTIZED_STORAGE_PATH
              value: /data/artifacts
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: amortized-secrets
                  key: openai-api-key
          volumeMounts:
            - name: data
              mountPath: /data
          livenessProbe:
            httpGet:
              path: /api/v1/health
              port: 9400
            initialDelaySeconds: 5
          readinessProbe:
            httpGet:
              path: /api/v1/health
              port: 9400
            initialDelaySeconds: 5
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: amortized-data

---
# k8s/server-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: amortized-server
  namespace: amortized
spec:
  selector:
    app: amortized-server
  ports:
    - port: 9400
      targetPort: 9400

---
# k8s/studio-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: amortized-studio
  namespace: amortized
spec:
  replicas: 1
  selector:
    matchLabels:
      app: amortized-studio
  template:
    metadata:
      labels:
        app: amortized-studio
    spec:
      containers:
        - name: studio
          image: ghcr.io/amortized-ai/studio:latest
          ports:
            - containerPort: 8080
          env:
            - name: BACKEND_HOST
              value: amortized-server
            - name: BACKEND_PORT
              value: "9400"

---
# k8s/studio-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: amortized-studio
  namespace: amortized
spec:
  selector:
    app: amortized-studio
  ports:
    - port: 8080
      targetPort: 8080

---
# k8s/studio-route.yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: amortized-studio
  namespace: amortized
spec:
  to:
    kind: Service
    name: amortized-studio
  port:
    targetPort: 8080
  tls:
    termination: edge

---
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: amortized-data
  namespace: amortized
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: gp3-csi
  resources:
    requests:
      storage: 50Gi

---
# k8s/serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: amortized-server
  namespace: amortized

---
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: amortized-job-manager
  namespace: amortized-jobs
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: amortized-job-manager
  namespace: amortized-jobs
subjects:
  - kind: ServiceAccount
    name: amortized-server
    namespace: amortized
roleRef:
  kind: Role
  name: amortized-job-manager
  apiGroup: rbac.authorization.k8s.io

---
# k8s/secrets.yaml (apply separately, not in git)
apiVersion: v1
kind: Secret
metadata:
  name: amortized-secrets
  namespace: amortized
type: Opaque
stringData:
  openai-api-key: "sk-..."
```

### 1.6 Event Ingest from K8s Jobs

Training/SDG containers need to POST events back to the server. In K8s, the server is reachable via its Service DNS:

```
AMORTIZED_EVENTS_URL=http://amortized-server.amortized.svc:9400/api/v1/events/ingest
```

Set this as an env var in the K8s Job spec created by the KubernetesBackend.

### 1.7 Container Images — Build and Push

```bash
# Build and push all images to ghcr.io
# Server
docker build -t ghcr.io/amortized-ai/amortized:latest -f Dockerfile .
docker push ghcr.io/amortized-ai/amortized:latest

# Studio
cd studio
docker build -t ghcr.io/amortized-ai/studio:latest .
docker push ghcr.io/amortized-ai/studio:latest

# Training container (already exists)

# SDG container
cd containers/synth
docker build -t ghcr.io/amortized-ai/asynth:latest .
docker push ghcr.io/amortized-ai/asynth:latest
```

---

## Deployment Steps on ROSA

```bash
# 1. Create namespaces
oc apply -f k8s/namespace.yaml

# 2. Create secrets (interactively, not from git)
oc create secret generic amortized-secrets \
  --from-literal=openai-api-key=sk-... \
  -n amortized

# 3. Create PVC
oc apply -f k8s/pvc.yaml

# 4. Create ServiceAccount + RBAC
oc apply -f k8s/serviceaccount.yaml
oc apply -f k8s/rbac.yaml

# 5. Deploy server
oc apply -f k8s/server-deployment.yaml
oc apply -f k8s/server-service.yaml

# 6. Deploy studio
oc apply -f k8s/studio-deployment.yaml
oc apply -f k8s/studio-service.yaml
oc apply -f k8s/studio-route.yaml

# 7. Verify
oc get pods -n amortized
oc get route -n amortized

# 8. Open Studio
# https://amortized-studio-amortized.apps.rosa.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com
```

---

## Code Changes Required

### amortized repo

| File | Change | Effort |
|---|---|---|
| `Dockerfile` (new) | Server production Dockerfile | Small |
| `backends/kubernetes.py` | Implement KubernetesBackend (replace stub) | Medium |
| `core/scheduler.py` | Add `AMORTIZED_EVENTS_URL` env var to K8s Job specs | Small |
| `api/app.py` | Support K8s backend type in `_load_backends()` | Small |
| `k8s/` (new directory) | All K8s manifests (deployments, services, routes, RBAC, PVC) | Small |

### studio repo

| File | Change | Effort |
|---|---|---|
| `Dockerfile` (new) | Multi-stage build: node → nginx | Small |
| `nginx.conf.template` (new) | SPA routing + /api/ reverse proxy | Small |

### asynth repo

| File | Change | Effort |
|---|---|---|
| `Dockerfile` | Add entrypoint script that loads config YAML and runs synthesize() | Small |

### Total effort: ~3-5 days of implementation

---

## Test Plan on ROSA

### Test 1: Server + Studio deployment
```bash
oc apply -f k8s/
# Verify: Studio loads in browser, /api/v1/health returns 200
```

### Test 2: Submit SDG job
```bash
# Via Studio chat or CLI
amortized submit sdg --recipe examples/ticket-classifier/synth -x
# Verify: K8s Job created in amortized-jobs namespace
# Verify: SDG pod runs on any node (no GPU needed)
# Verify: Output JSONL registered as artifact
```

### Test 3: Submit training job
```bash
amortized submit training --recipe examples/ticket-classifier/train \
  --set config.data_path=artifact:JOB_ID/generated_data.jsonl -x
# Verify: K8s Job created with GPU request
# Verify: Pod scheduled on L40S node
# Verify: Training metrics streamed via events
# Verify: Model artifact registered
```

### Test 4: Full pipeline
```
Studio chat: "Build me a ticket classifier"
→ Agent proposes SDG → Confirm
→ SDG runs as K8s Job (CPU)
→ Agent reviews data quality
→ Agent proposes training → Confirm
→ Training runs as K8s Job (L40S GPU)
→ Agent shows results
→ Deploy via KServe (Phase 2)
```

---

## Phase 2 (After MVP Works)

| # | Work | Depends on |
|---|---|---|
| 1 | Model Registry integration — register models via `model_registry` Python client (Kubeflow Hub) | MVP working |
| 2 | MLflow experiment tracking — deploy MLflow, set `report_to: mlflow` | MVP working |
| 3 | KServe integration for model serving — target RawDeployment mode. Add annotation `serving.knative.dev/progress-deadline: 30m`. | MVP working |
| 4 | S3 artifact store (AWS S3 bucket) | MLflow |
| 5 | OpenShift OAuth proxy for Studio | MVP working |
| 6 | Persistent job handles (survive server restart) | MVP working |
