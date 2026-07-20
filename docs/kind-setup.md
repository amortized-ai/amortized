# Kind Cluster Setup for GPU Development

Deploy the full amortized stack on a shared GPU machine using kind (Kubernetes in Docker). One command to go from zero to a running cluster with 8x H100 GPUs, MinIO, MLflow, Studio, and OpenCode.

## Prerequisites

The following must be installed on the GPU node:

- **Docker** with nvidia as the default runtime (`/etc/docker/daemon.json` → `"default-runtime": "nvidia"`)
- **kind** (v0.20+)
- **kubectl**
- **nvidia-smi** (NVIDIA drivers installed on the host)
- **jq** (for OpenCode credential copying)

Verify with:

```bash
docker info | grep "Default Runtime"   # should show: nvidia
kind version
kubectl version --client
nvidia-smi -L                          # should list GPUs
```

### inotify limits

If the machine runs multiple kind clusters, bump the inotify limits to prevent kubelet crashes:

```bash
sudo sysctl -w fs.inotify.max_user_watches=1048576
sudo sysctl -w fs.inotify.max_user_instances=8192
```

To make this persistent across reboots:

```bash
echo "fs.inotify.max_user_watches=1048576" | sudo tee -a /etc/sysctl.conf
echo "fs.inotify.max_user_instances=8192" | sudo tee -a /etc/sysctl.conf
```

## Quick Start

```bash
git clone https://github.com/amortized-ai/amortized && cd amortized
make up GHCR_USER=<github-user> GHCR_TOKEN=<github-pat>
```

This creates a kind cluster, sets up GPU passthrough, pulls GHCR images, and deploys the full prod stack. First run takes ~20 minutes (pulls ~16GB training image).

The `GHCR_TOKEN` needs the `read:packages` scope. It's used to pull private images and create in-cluster pull secrets.

## Architecture

### Cluster layout

```
kind cluster: amortized
├── control-plane node (scheduling, API server)
└── worker node (GPU workloads, 8x H100)

Namespaces:
├── amortized           ← server, studio, opencode, claude-code, minio, mlflow
├── amortized-jobs      ← training/SDG/eval K8s Jobs (4 GPU quota)
├── amortized-dev       ← dev server, studio, opencode, claude-code
└── amortized-dev-jobs  ← dev training jobs (4 GPU quota)
```

### Component map

| Component | Namespace | Image | Port | NodePort |
|-----------|-----------|-------|------|----------|
| Server (API) | amortized | ghcr.io/amortized-ai/amortized:latest | 8000 | 31081 |
| Studio (UI) | amortized | ghcr.io/amortized-ai/studio:latest | 8080 | 31080 |
| OpenCode (AI assistant) | amortized | ghcr.io/anomalyco/opencode | 4096 | — |
| Claude Code | amortized | ghcr.io/amortized-ai/claude-code-agent | 4096 | — |
| MinIO (S3) | amortized | quay.io/minio/minio | 9000 | — |
| MLflow | amortized | ghcr.io/mlflow/mlflow | 5000 | 31082 |
| Dev Server | amortized-dev | amortized-server:kind-\<sha\> | 8000 | 31091 |
| Dev Studio | amortized-dev | amortized-studio:kind-\<sha\> | 8080 | 31090 |
| Dev OpenCode | amortized-dev | ghcr.io/anomalyco/opencode | 4096 | — |
| Dev Claude Code | amortized-dev | ghcr.io/amortized-ai/claude-code-agent | 4096 | — |

**Prod** uses GHCR images directly (pulled from the registry). **Dev** builds from local source (for PR testing).

### GPU isolation

Each jobs namespace gets a ResourceQuota of 4 GPUs. The 8 physical GPUs are split:

- `amortized-jobs`: 4 GPU limit (prod training)
- `amortized-dev-jobs`: 4 GPU limit (dev/PR testing)

Jobs that request more GPUs than available will pend (not error). There is no preemption — first come, first served.

### Dev namespace

The dev namespace (`amortized-dev`) is designed for PR testing. It:

- Runs its own server and studio instances
- Has its own jobs namespace with isolated RBAC
- **Shares** MinIO and MLflow with prod (cross-namespace DNS)
- Has its own OpenCode and Claude Code deployments (pointing at dev server)

## Makefile Reference

```
make help            Show all targets
make up              Full setup: cluster + GPU + deploy prod
make deploy          Deploy prod stack (pulls GHCR images)
make deploy-dev      Build from source + deploy dev stack
make test-server     Build server from current branch + deploy to dev
make test-studio     Build studio from current branch + deploy to dev
make build           Build all images locally (server + studio + pull deps)
make build-server    Build server image only
make build-studio    Build studio image only
make status          Show pods, GPUs, access URLs
make down            Delete dev namespaces (keep prod)
make destroy         Delete entire kind cluster
```

### Overridable variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GHCR_USER` | *(required for prod)* | GitHub username for pulling private images |
| `GHCR_TOKEN` | *(required for prod)* | GitHub PAT with `read:packages` scope |
| `STUDIO_DIR` | `../studio` | Path to studio repo checkout |
| `CLUSTER_NAME` | `amortized` | Kind cluster name |
| `IMAGE_TAG` | `kind-<git-sha>` | Tag for locally built images (dev only) |
| `SERVER_IMAGE` | `ghcr.io/amortized-ai/amortized:latest` | Server image for prod |
| `STUDIO_IMAGE` | `ghcr.io/amortized-ai/studio:latest` | Studio image for prod |
| `GOOGLE_ADC_PATH` | `~/.config/gcloud/application_default_credentials.json` | Path to GCP ADC credentials |
| `VERTEX_PROJECT` | `redhat-ai-analysis` | GCP project for Vertex AI |
| `VERTEX_LOCATION` | `us-central1` | Vertex AI region |

Example: `make deploy-dev STUDIO_DIR=~/my-studio-fork GHCR_USER=user GHCR_TOKEN=token`

## Developer Workflows

### First-time setup

```bash
ssh user@169.62.17.147
git clone https://github.com/amortized-ai/amortized && cd amortized
make up GHCR_USER=<github-user> GHCR_TOKEN=<github-pat>
```

The first run takes ~20 minutes due to the training image pull (~16GB). Subsequent runs are faster since images are cached.

`make up` handles everything: cluster creation, GPU setup, pulling third-party images, pulling GHCR images, creating secrets, and deploying.

### Test a server PR

```bash
git checkout feature-branch
make test-server
# Dev API available at http://localhost:31091
```

This rebuilds the server image from the current branch, loads it into kind, and deploys to the dev namespace.

### Test a studio PR

```bash
cd ../studio && git checkout feature-branch && cd ../amortized
make test-studio
# Dev Studio available at http://localhost:31090
```

### Redeploy after config changes

```bash
make deploy GHCR_USER=<user> GHCR_TOKEN=<token>       # redeploy prod (pulls latest GHCR images)
make deploy-dev GHCR_USER=<user> GHCR_TOKEN=<token>   # redeploy dev (builds from local source)
```

### Check status

```bash
make status
```

Shows all pods, GPU allocation, and access URLs.

### Clean up

```bash
make down      # remove dev namespace only, prod stays
make destroy   # delete the entire kind cluster
```

## Accessing Services

### From the GPU node

Services are directly accessible via NodePort:

```bash
curl http://localhost:31081/api/v1/health    # API
curl http://localhost:31080                   # Studio
curl http://localhost:31082                   # MLflow
```

### From your laptop (SSH tunnel)

Only port 22 is externally accessible on the GPU node. Use SSH tunneling:

```bash
ssh -L 31080:localhost:31080 \
    -L 31081:localhost:31081 \
    -L 31082:localhost:31082 \
    -L 31090:localhost:31090 \
    -L 31091:localhost:31091 \
    user@169.62.17.147
```

Then open in your browser:

| Service | URL |
|---------|-----|
| Prod Studio | http://localhost:31080 |
| Prod API | http://localhost:31081 |
| MLflow | http://localhost:31082 |
| Dev Studio | http://localhost:31090 |
| Dev API | http://localhost:31091 |

## How GPU Passthrough Works

Getting GPUs into a kind cluster requires three layers:

1. **Host Docker runtime**: nvidia must be the default runtime in `/etc/docker/daemon.json`. This makes all Docker containers (including kind nodes) GPU-aware.

2. **Kind worker mount**: The kind config mounts `/dev/null:/var/run/nvidia-container-devices/all` into the worker node. This tells the nvidia container runtime to inject all GPU devices into the container.

3. **Containerd configuration inside the worker**: The `make gpu` target installs `nvidia-container-toolkit` inside the kind worker node and configures containerd to use the nvidia runtime as the default. This is required because containerd (which runs inside the kind node) doesn't inherit Docker's nvidia runtime config.

4. **NVIDIA device plugin**: A DaemonSet that registers `nvidia.com/gpu` resources with the kubelet, making GPUs schedulable by Kubernetes.

After `make gpu`, verify GPUs are visible:

```bash
kubectl --context kind-amortized get nodes \
  -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
```

## Troubleshooting

### Worker node NotReady / kubelet crash

**Symptom**: `kubectl get nodes` shows worker as NotReady, kubelet logs show `inotify_init: too many open files`.

**Fix**: Bump inotify limits (see Prerequisites section), then restart the worker:

```bash
docker restart amortized-worker
```

### GPUs not showing up

**Symptom**: `nvidia.com/gpu` is blank in node allocatable resources.

Check in order:

1. **nvidia-smi inside the worker**:
   ```bash
   docker exec amortized-worker nvidia-smi -L
   ```
   If this fails, the nvidia-container-devices mount is missing. Recreate the cluster.

2. **Device plugin logs**:
   ```bash
   kubectl --context kind-amortized -n kube-system logs -l name=nvidia-device-plugin-ds
   ```
   If you see `Incompatible strategy detected auto`, the containerd nvidia runtime isn't configured. Run `make gpu` again.

3. **Device plugin running but no devices**: Restart the device plugin after containerd is reconfigured:
   ```bash
   kubectl --context kind-amortized -n kube-system delete pod -l name=nvidia-device-plugin-ds
   ```

### ImagePullBackOff

**Symptom**: Pods stuck in ImagePullBackOff.

Kind clusters have no registry — images must be pre-loaded. The image tag in the deployment must match what was loaded:

```bash
# Check what tags are loaded
docker exec amortized-worker crictl images | grep amortized

# Check what tag the deployment expects
kubectl --context kind-amortized -n amortized get deploy amortized-server \
  -o jsonpath='{.spec.template.spec.containers[0].image}'
```

If they don't match, rebuild and reload:

```bash
make build load
kubectl --context kind-amortized -n amortized rollout restart deployment/amortized-server
```

### Pods stuck in CreateContainerConfigError

**Symptom**: MinIO or MLflow pods show `CreateContainerConfigError`.

Usually caused by `runAsNonRoot: true` in the pod security context — these images run as root. The Makefile patches this automatically during `make deploy`, but if you applied manifests manually:

```bash
kubectl --context kind-amortized -n amortized patch deployment minio --type json \
  -p '[{"op":"remove","path":"/spec/template/spec/securityContext/runAsNonRoot"}]'
```

### OpenCode not starting

**Symptom**: OpenCode pod in CrashLoopBackOff or CreateContainerConfigError.

The OpenCode deployment needs two secrets:

- `opencode-gcp` — GCP Application Default Credentials (created automatically from `GOOGLE_ADC_PATH`)
- `opencode-llm` — Vertex AI project/location (created automatically from `VERTEX_PROJECT`/`VERTEX_LOCATION`)

These are created automatically during `make deploy`. If they're missing:

```bash
kubectl --context kind-amortized -n amortized get secret opencode-gcp opencode-llm
```

To recreate manually:

```bash
kubectl --context kind-amortized -n amortized create secret generic opencode-gcp \
  --from-file=credentials.json=$HOME/.config/gcloud/application_default_credentials.json
kubectl --context kind-amortized -n amortized create secret generic opencode-llm \
  --from-literal=google-cloud-project=redhat-ai-analysis \
  --from-literal=vertex-location=us-central1
```

### Studio crash: "host not found in upstream"

**Symptom**: Studio pod crashes with `host not found in upstream "opencode"`.

The nginx config resolves upstream hostnames at startup. If OpenCode isn't deployed, set a dummy host:

```bash
kubectl --context kind-amortized -n amortized set env deployment/amortized-studio OPENCODE_HOST=127.0.0.1
```

### Port conflicts with existing cluster

The `amortized` cluster uses ports 31080-31091. If another cluster uses the same ports:

```bash
ss -tlnp | grep -E '310[89]'
```

To use different ports, edit `k8s/kind/kind-config.yaml` and the corresponding NodePort service files.

## Differences from OpenShift Deployment

The kind setup differs from production OpenShift in several ways:

| Aspect | OpenShift | Kind |
|--------|-----------|------|
| Ingress | Routes (`route.openshift.io`) | NodePort services |
| Security | SCCs enforce non-root | `runAsNonRoot` patched out |
| Image pull | Registry (GHCR) | GHCR pull secret + `kind load` (prod pulls from GHCR, dev builds locally) |
| GPU access | Native device plugin | Manual nvidia-toolkit setup |
| Storage | Dynamic provisioner | local-path (hostPath) |
| OpenCode creds | Secrets created by admin | Auto-created from local ADC + Vertex config |

Route manifests (`*-route.yaml`) are skipped during kind deployment.

### Private image pulls (GHCR)

Images under `ghcr.io/amortized-ai/` are private. GHCR pull secrets are created automatically during `make deploy` and `make deploy-dev` when `GHCR_USER` and `GHCR_TOKEN` are provided. To create them standalone:

```bash
make ghcr-pull-secret GHCR_USER=<github-username> GHCR_TOKEN=<github-pat>
```

The PAT needs the `read:packages` scope.
