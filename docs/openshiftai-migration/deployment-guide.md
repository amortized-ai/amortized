# Deploying Amortized on OpenShift

Tested on ROSA (Red Hat OpenShift Service on AWS) 4.20.21 with NVIDIA L40S GPUs.

## Prerequisites

- OpenShift 4.14+ cluster with `oc` CLI access
- NVIDIA GPU Operator installed (`nvidia.com/gpu` resources available)
- Permission to create projects (`oc new-project`)
- Docker (for building images)
- Access to push images to ghcr.io or another registry

## Step 1: Create Namespaces

```bash
oc new-project amortized        # control plane: server, studio, MLflow, MinIO
oc new-project amortized-jobs   # compute: training, SDG, eval Jobs + serve Deployments
```

## Step 2: Deploy Infrastructure (MinIO + MLflow)

```bash
# Switch to amortized namespace
oc project amortized

# Apply secrets (S3/MinIO credentials for both namespaces)
oc apply -f k8s/dev/s3-secret.yaml
oc apply -f k8s/dev/s3-secret-jobs.yaml

# Deploy MinIO (S3-compatible object store)
oc apply -f k8s/dev/minio-pvc.yaml
oc apply -f k8s/dev/minio-deployment.yaml
oc apply -f k8s/dev/minio-service.yaml

# Deploy MLflow (experiment tracking + artifact store)
oc apply -f k8s/dev/mlflow-pvc.yaml
oc apply -f k8s/dev/mlflow-deployment.yaml
oc apply -f k8s/dev/mlflow-service.yaml
oc apply -f k8s/dev/mlflow-route.yaml

# Wait for pods to be ready
oc get pods -n amortized -w
# Both minio and mlflow should show 1/1 Running
```

## Step 3: Create MinIO Bucket

```bash
# Wait for MinIO to be ready
oc wait --for=condition=available deploy/minio -n amortized --timeout=120s

# Create the amortized bucket
oc exec deploy/minio -n amortized -- mc alias set local http://localhost:9000 minioadmin minioadmin
oc exec deploy/minio -n amortized -- mc mb local/amortized
```

## Step 4: Verify MLflow → MinIO Connectivity

```bash
# Test that MLflow can write artifacts to MinIO
oc exec deploy/mlflow -n amortized -- python3 -c "
import mlflow, tempfile
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('test/connectivity')
with mlflow.start_run(run_name='connectivity-test'):
    mlflow.log_param('test', 'hello')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('test artifact')
        f.flush()
        mlflow.log_artifact(f.name, 'test')
print('SUCCESS')
"

# Verify artifact landed in MinIO
oc exec deploy/minio -n amortized -- mc ls local/amortized/mlflow/ --recursive
# Should show the test artifact file
```

## Step 5: Access MLflow UI

```bash
# Get the MLflow Route URL
oc get route mlflow -n amortized -o jsonpath='{.spec.host}'
# Open in browser: https://<mlflow-route-url>
# You should see the test experiment and run
```

## Step 6: Deploy RBAC + Config

```bash
# ServiceAccount for the amortized server
oc apply -f k8s/base/serviceaccount.yaml

# RBAC — grants amortized-server permission to create Jobs, Deployments,
# Services, Secrets, ConfigMaps in the amortized-jobs namespace
oc apply -f k8s/base/rbac.yaml

# ConfigMap — server configuration (compute backend, MLflow URI, etc.)
oc apply -f k8s/base/configmap.yaml
```

## Step 7: Build and Push Server Image

```bash
# Build the amortized server image
docker build -t ghcr.io/amortized-ai/amortized:latest .

# Login to ghcr.io (needs a GitHub PAT with packages:write scope)
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Push
docker push ghcr.io/amortized-ai/amortized:latest
```

Alternatively, use OpenShift's internal registry:
```bash
# Login to the internal registry
oc registry login

# Tag and push
REGISTRY=$(oc registry info)
docker tag ghcr.io/amortized-ai/amortized:latest $REGISTRY/amortized/amortized:latest
docker push $REGISTRY/amortized/amortized:latest

# Update server-deployment.yaml to use the internal registry image
```

## Step 8: Deploy Server

```bash
oc apply -f k8s/base/server-pvc.yaml
oc apply -f k8s/base/server-deployment.yaml
oc apply -f k8s/base/server-service.yaml

# Wait for server to be ready
oc wait --for=condition=available deploy/amortized-server -n amortized --timeout=120s

# Verify health
oc exec deploy/amortized-server -n amortized -- curl -s http://localhost:8000/api/v1/health
# Should return: {"status": "ok"} or similar
```

## Step 9: Deploy Studio

Studio source lives in a separate repo. Build and push similarly:
```bash
# From the studio repo
docker build -t ghcr.io/amortized-ai/studio:latest .
docker push ghcr.io/amortized-ai/studio:latest

# Deploy
oc apply -f k8s/base/studio-deployment.yaml
oc apply -f k8s/base/studio-service.yaml
oc apply -f k8s/base/studio-route.yaml

# Get the Studio URL
oc get route amortized-studio -n amortized -o jsonpath='{.spec.host}'
# Open in browser: https://<studio-route-url>
```

## Step 10: Add LLM API Keys

Data scientists add their API keys through Studio's Settings UI, or via the API:

```bash
STUDIO_URL=$(oc get route amortized-studio -n amortized -o jsonpath='{.spec.host}')

curl -X POST https://$STUDIO_URL/api/v1/settings/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "openai", "provider": "openai", "key_value": "sk-..."}'
```

Keys are stored encrypted in amortized's database and injected into job pods as per-job K8s Secrets at dispatch time.

## Step 11: Test a Job

```bash
# Submit an SDG job (CPU, no GPU needed)
amortized --url https://$STUDIO_URL submit examples/ticket-classifier/synth.yaml -x

# Watch the K8s Job
oc get jobs -n amortized-jobs -w

# Check logs
oc logs job/amortized-<job-id> -n amortized-jobs -f

# After completion, check MLflow for artifacts
# Open MLflow UI → amortized/sdg experiment → latest run → artifacts
```

## Verification Checklist

| Check | Command | Expected |
|---|---|---|
| Namespaces exist | `oc get projects \| grep amortized` | 2 projects |
| MinIO running | `oc get pods -n amortized -l component=minio` | 1/1 Running |
| MLflow running | `oc get pods -n amortized -l component=mlflow` | 1/1 Running |
| MLflow UI accessible | `curl https://<mlflow-route>/health` | OK |
| MinIO bucket exists | `oc exec deploy/minio -- mc ls local/amortized/` | Lists mlflow/ |
| Server running | `oc get pods -n amortized -l component=server` | 1/1 Running |
| Health endpoint | `curl https://<studio-route>/api/v1/health` | 200 OK |
| Studio loads | Open `https://<studio-route>` in browser | React app loads |
| RBAC works | `oc auth can-i create jobs -n amortized-jobs --as system:serviceaccount:amortized:amortized-server` | yes |

## Troubleshooting

**Pod stuck in Pending:**
```bash
oc describe pod <pod-name> -n amortized
# Check Events section for: insufficient GPU, PVC not bound, image pull errors
```

**MLflow can't write to MinIO:**
```bash
# Verify MLFLOW_S3_ENDPOINT_URL is set
oc exec deploy/mlflow -n amortized -- env | grep MLFLOW_S3
# Should show: MLFLOW_S3_ENDPOINT_URL=http://minio.amortized.svc.cluster.local:9000
```

**Job pod can't reach MLflow:**
```bash
# Test cross-namespace DNS from a job pod
oc run test-dns --rm -i --restart=Never --image=busybox -n amortized-jobs -- \
  nslookup mlflow.amortized.svc.cluster.local
```

**Image pull errors:**
```bash
# Check if the image is accessible
oc get events -n amortized --field-selector reason=Failed | grep -i pull
# For private registries, create an image pull secret:
oc create secret docker-registry ghcr-pull \
  --docker-server=ghcr.io \
  --docker-username=USERNAME \
  --docker-password=$GITHUB_TOKEN \
  -n amortized
oc secrets link default ghcr-pull --for=pull
```

## Architecture

```
Namespace: amortized
  ├── MinIO (S3 object store) — port 9000, PVC 200Gi
  ├── MLflow (tracking + artifacts) — port 5000, Route for UI
  ├── amortized-server (FastAPI) — port 8000, PVC 10Gi
  └── amortized-studio (nginx) — port 8080, Route for UI

Namespace: amortized-jobs
  ├── Training K8s Jobs (GPU)
  ├── SDG K8s Jobs (CPU)
  ├── Eval K8s Jobs (CPU)
  └── Serve Deployments + Services (GPU, long-running)

Storage:
  MinIO bucket "amortized" → MLflow artifact store
  Server PVC → SQLite (job queue, chat history)
  MLflow PVC → SQLite (experiment metadata)
```

## Cleanup

```bash
# Remove everything
oc delete project amortized
oc delete project amortized-jobs
```
