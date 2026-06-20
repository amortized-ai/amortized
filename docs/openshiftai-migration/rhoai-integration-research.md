# Amortized on OpenShift AI — Deep Research Report

Research conducted 2026-06-19. 105 agents, 23 sources fetched, 114 claims extracted, 25 adversarially verified (18 confirmed, 7 killed).

## Executive Summary

Amortized can eliminate most of its custom infrastructure by integrating with OpenShift AI 3.3.1 services already running or readily available on the ROSA cluster. The architectural shift: **amortized stops generating configs and SSHing them to GPU nodes**, and instead **generates Kubernetes CRs and API calls**, letting RHOAI operators handle scheduling, scaling, and lifecycle.

| Current (standalone) | Target (on RHOAI) | RHOAI Service |
|---|---|---|
| SSH to GPU node + podman run vLLM | Create InferenceService CR | KServe (GA) |
| SQLite artifact table + local filesystem | Register via REST API | Model Registry (GA) |
| `report_to: none`, custom metrics.jsonl | `report_to: mlflow` | MLflow (Tech Preview → GA in 3.4) |
| SSH + podman run TRL container | Create PyTorchJob/TrainJob CR | Training Operator (GA) / Trainer v2 (TP) |
| No GPU scheduling | Add Kueue label to jobs | Kueue (not installed, easy to add) |
| Bearer token auth | OAuth proxy sidecar | OpenShift OAuth |
| Local filesystem artifacts | S3 bucket | AWS S3 / MinIO |

---

## 1. Model Registry — Use It, Don't Rebuild It

**Status**: GA, already running as `default-modelregistry` on the cluster.

### What It Is

The RHOAI Model Registry is **Kubeflow-based** (recently renamed Kubeflow Hub), NOT MLflow Model Registry. This is a critical distinction — your existing docs reference MLflow Model Registry, but the service running on the cluster is a different system.

**Key facts** (verified 3-0):
- REST API at `v1alpha3` with endpoints under `/api/model_registry/v1alpha3/`
- OpenAPI contract-first design (model-registry.yaml)
- MySQL 5.x+ backend (8.x recommended). PostgreSQL support arrives in RHOAI 3.4
- Stores **model metadata only**, not artifacts themselves. Artifacts live in S3/PVC/OCI
- Python client: `model_registry` package

### How Amortized Integrates

```python
from model_registry import ModelRegistry

registry = ModelRegistry(
    server_address="https://default-modelregistry-rhoai-model-registries.svc:8080",
    author="amortized",
)

# After training completes — register the model
registered_model = registry.register_model(
    name=f"{project}/{model_name}",
    uri=f"s3://{bucket}/models/{job_id}/",
    model_format_name="vllm",
    model_format_version="1",
    description=f"Task model trained by amortized job {job_id}",
)

# Deploy via KServe with lineage labels
isvc_labels = {
    "modelregistry/registered-model-id": registered_model.id,
    "modelregistry/model-version-id": registered_model.versions[0].id,
}
```

### What This Eliminates

- Amortized's SQLite `artifacts` table for model tracking
- Custom `artifact:<uuid>` reference system
- Any plan to build a custom model registry
- MLflow Model Registry integration (Model Registry replaces this role)

### Confidence: HIGH (3-0 unanimous)

Sources: [RHOAI 3.3 Managing Model Registries](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/pdf/managing_model_registries/Red_Hat_OpenShift_AI_Self-Managed-3.3-Managing_model_registries-en-US.pdf), [Kubeflow Hub REST API](https://www.kubeflow.org/docs/components/hub/reference/rest-api/)

---

## 2. KServe — Model Serving via InferenceService CRs

**Status**: GA at version 0.15, vLLM CUDA v0.13.0. Already running on the cluster with 2 active InferenceServices.

### Architecture

Model serving on RHOAI uses two CRDs:
- **ServingRuntime** — defines the serving container environment (vLLM image, resources, pod templates)
- **InferenceService** — references a ServingRuntime and specifies the model to serve

Pre-built ServingRuntimes exist for NVIDIA GPU, AMD GPU, Intel Gaudi, CPU, and IBM Spyre (Tech Preview). The platform auto-selects runtimes based on hardware profile.

### Deployment Modes

Two modes exist, but this is shifting:
- **RawDeployment** ("Standard") — standard K8s Deployments. **Target this.**
- **Serverless** ("Advanced") — Knative + Istio. **Deprecated in RHOAI 3.3.**
- **RHOAI 3.4** introduces `LLMInferenceService` / `LLMInferenceServiceConfig` as new CRDs specifically for LLM workloads.

### Model Storage Options (verified 3-0)

| Method | URI Format | Best For |
|---|---|---|
| **S3** | `s3://bucket/path/` | Training checkpoints, large models |
| **PVC** | `pvc://claim-name/path/` | Models already on persistent storage |
| **OCI ModelCar** | `oci://quay.io/org/model:tag` | Versioned deployment, no S3 dependency |
| **HTTP/HF** | `https://...` or HuggingFace repo | Public models |

### OCI ModelCar (verified 3-0)

ModelCar packages models as OCI container images with files in `/models`. The ModelCar sidecar uses shared process namespaces and `/proc` symlinks to expose models at `/mnt/models` without data copying. Introduced in RHOAI 2.14, so 3.3.1 fully supports it.

For amortized's task models, ModelCar is attractive:
1. After training completes, build OCI image: `FROM busybox` + `COPY model/ /models/`
2. Push to Quay/registry
3. Reference in InferenceService: `storageUri: oci://registry/model:tag`

**Caveat**: Practical size limit ~15-20GB. Transitional technology — Kubernetes 1.31+ native OCI image volumes will replace it.

### How Amortized Integrates

```python
from kubernetes import client

isvc = {
    "apiVersion": "serving.kserve.io/v1beta1",
    "kind": "InferenceService",
    "metadata": {
        "name": model_name,
        "namespace": compute_namespace,
        "labels": {
            "modelregistry/registered-model-id": model_id,
            "modelregistry/model-version-id": version_id,
        },
        "annotations": {
            # Critical for LLMs — default 10min timeout kills model downloads
            "serving.knative.dev/progress-deadline": "30m",
        },
    },
    "spec": {
        "predictor": {
            "model": {
                "modelFormat": {"name": "vllm"},
                "storageUri": f"s3://{bucket}/models/{model_path}",
                "resources": {
                    "requests": {"nvidia.com/gpu": "1"},
                    "limits": {"nvidia.com/gpu": "1"},
                },
            }
        }
    }
}
custom_api.create_namespaced_custom_object(
    group="serving.kserve.io", version="v1beta1",
    namespace=compute_namespace, plural="inferenceservices",
    body=isvc,
)
```

### What This Eliminates

- Amortized's `serve` job type container orchestration
- vLLM container image management
- Custom vLLM YAML config generation (`_serve_config_yaml()`)
- Entrypoint CMD construction for vLLM
- The gotcha about vLLM entrypoint being `["vllm", "serve"]`

### Confidence: HIGH (3-0)

Sources: [RHOAI Supported Configs](https://access.redhat.com/articles/rhoai-supported-configs-3.x), [Deploy LLM Inference](https://developers.redhat.com/articles/2025/11/03/deploy-llm-inference-service-openshift-ai), [ModelCar on RHOAI](https://developers.redhat.com/articles/2025/01/30/build-and-deploy-modelcar-container-openshift-ai)

---

## 3. Training — Kubeflow Training Operator / Trainer v2

**Status**: Training Operator v1 (PyTorchJob) is GA and running. Trainer v2 (TrainJob) is **Technology Preview** in RHOAI 3.3.

### Training Operator v1 (PyTorchJob) — Use Now

PyTorchJob is the safe choice today. It provides:
- Multi-node distributed training (Master/Worker replica pattern)
- Automatic pod scheduling with GPU resource requests
- Built-in gang scheduling support
- Status tracking via K8s CRD conditions

For amortized's single-node LoRA SFT workloads, PyTorchJob with 1 replica works but is heavier than a plain `batch/v1 Job`. The benefit comes when scaling to multi-node.

### Trainer v2 (TrainJob) — Coming Soon

Trainer v2 introduces a significantly simpler API (verified 3-0):

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: amortized-train-abc123
spec:
  runtimeRef:
    name: torch-distributed  # references ClusterTrainingRuntime
  trainer:
    numNodes: 1
    resourcesPerNode:
      requests:
        nvidia.com/gpu: "1"
    command: ["python", "-m", "torch.distributed.run", "train.py"]
    env:
      - name: AMORTIZED_CONFIG_JSON
        value: '...'
```

Key improvements over PyTorchJob:
- `numNodes` instead of separate Master/Worker replicas
- `resourcesPerNode` for uniform allocation
- `runtimeRef` references reusable ClusterTrainingRuntime resources

**Critical caveat**: The exact ClusterTrainingRuntimes shipped with RHOAI 3.3 could NOT be verified (0-3 refuted). Amortized may need to create custom TrainingRuntime resources with the TRL/LoRA stack.

### Recommendation

**Phase 1**: Use plain `batch/v1 Job` (simplest, no dependency on Training Operator specifics)
**Phase 2**: Migrate to PyTorchJob when multi-node is needed
**Phase 3**: Adopt TrainJob v2 when it reaches GA (likely RHOAI 3.4-3.5)

### Confidence: MEDIUM (3-0 on API, but Tech Preview status and refuted specifics about bundled runtimes)

Source: [RHOAI 3.3 Trainer v2](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/html/working_with_distributed_workloads/running-kubeflow-trainerv2_distributed-workloads)

---

## 4. Kueue — GPU Scheduling

**Status**: NOT installed on the cluster. CRDs don't exist. Easy to add.

### What It Provides

GPU quota management, fair-sharing between teams, and preemption — without amortized building custom scheduling logic.

### Integration Is Lightweight (verified 2-1)

From amortized's perspective, Kueue integration is a single label on job metadata:

```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: amortized-queue
```

Kueue intercepts the Job and manages it through its queue system.

### Prerequisites (cluster-admin, one-time)

```bash
# Install Kueue operator (or enable via RHOAI dashboard)
# Then create queue infrastructure:
kubectl apply -f - <<EOF
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: l40s-gpu
spec:
  nodeLabels:
    nvidia.com/gpu.product: NVIDIA-L40S
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  resourceGroups:
    - coveredResources: ["nvidia.com/gpu"]
      flavors:
        - name: l40s-gpu
          resources:
            - name: nvidia.com/gpu
              nominalQuota: 4
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: amortized-queue
  namespace: amortized-jobs
spec:
  clusterQueue: gpu-cluster-queue
EOF
```

### Kueue + Training Operator

Kueue integrates with both PyTorchJob and Trainer v2 TrainJob. For PyTorchJobs, enable `kubeflow.org/pytorchjob` in Kueue's `integrations.frameworks` config.

### Recommendation

Install Kueue in Phase 2. For Phase 1, plain K8s Jobs with `nodeSelector` for GPU nodes is sufficient. Kueue becomes important when:
- Multiple users compete for GPUs
- Fair-sharing between teams is needed
- Preemption of lower-priority jobs is desired

### Confidence: MEDIUM (2-1, understates one-time cluster config)

Source: [Kueue PyTorchJobs](https://kueue.sigs.k8s.io/docs/tasks/run/kubeflow/pytorchjobs/), [GPU Utilization with Kueue on RHOAI](https://developers.redhat.com/articles/2025/05/22/improve-gpu-utilization-kueue-openshift-ai)

---

## 5. MLflow — Experiment Tracking

**Status**: AMBIGUOUS. The claim that MLflow is Tech Preview in RHOAI 3.3 was refuted (1-2). Not deployed on the cluster.

### What We Know

- MLflow is NOT currently deployed on the user's cluster
- Model Registry (Kubeflow Hub) IS deployed and serves model versioning needs
- MLflow would complement Model Registry (experiment tracking) rather than replace it
- The `mlflow-operator` exists in the OpenDataHub ecosystem

### Where MLflow Fits

MLflow handles what Model Registry doesn't: **experiment tracking** (loss curves, hyperparams, run comparison).

| Concern | Model Registry (Kubeflow Hub) | MLflow |
|---|---|---|
| Model versioning | Yes | Yes (but don't use — MR is canonical) |
| Artifact storage | Metadata only (points to S3/OCI) | Full artifact store |
| Experiment tracking | No | Yes |
| Run comparison | No | Yes |
| Hyperparameter logging | No | Yes |
| Loss curves | No | Yes |

### Recommendation

**Phase 1**: Skip MLflow. Use Model Registry for model tracking. Training metrics go to amortized's existing event stream + Studio UI.

**Phase 2**: Deploy MLflow for experiment tracking. Set `report_to: mlflow` in TRL configs. This gives data scientists MLflow's run comparison UI — don't rebuild it in Studio.

**Phase 3**: If MLflow becomes GA in RHOAI 3.4+, switch to the RHOAI-managed instance.

### Important Correction to Existing Docs

Your `openshift-ai-integration.md` and `mlflow-integration-plan.md` conflate MLflow Model Registry with RHOAI Model Registry. They are different systems:

- **RHOAI Model Registry** = Kubeflow Hub. REST API at v1alpha3. Already running.
- **MLflow Model Registry** = Part of MLflow. NOT deployed. NOT the same thing.

For model versioning and deployment lineage, use the Kubeflow Model Registry. MLflow is only for experiment tracking.

### Confidence: MEDIUM (MLflow status unverified, but Model Registry facts are high-confidence)

---

## 6. Storage — S3, PVC, OCI

### Current Cluster State

- **EBS** (gp3-csi) is the default StorageClass — PVCs work
- **S3** is NOT configured — need to create an AWS S3 bucket
- **OCI** (ModelCar) is supported for model serving

### Recommended Storage Architecture

| Data Type | Storage | Why |
|---|---|---|
| Training data (JSONL) | S3 | Shared across jobs, survives pod restarts |
| Model checkpoints | S3 | Resume from checkpoint on retry |
| Trained model weights | S3 → registered in Model Registry | Source of truth for deployment |
| Model for serving | S3 or OCI ModelCar | KServe reads from either |
| SDG output | S3 | Input to training jobs |
| Eval results | S3 | Archival |
| Job queue + conversations | SQLite on PVC | Amortized control plane state |

### S3 Setup

```bash
# Create S3 bucket (AWS CLI)
aws s3 mb s3://amortized-artifacts --region us-east-2

# Create K8s secret with S3 credentials
oc create secret generic amortized-s3 \
  --from-literal=AWS_ACCESS_KEY_ID=... \
  --from-literal=AWS_SECRET_ACCESS_KEY=... \
  --from-literal=AWS_S3_BUCKET=amortized-artifacts \
  --from-literal=AWS_S3_ENDPOINT=https://s3.us-east-2.amazonaws.com \
  --from-literal=AWS_DEFAULT_REGION=us-east-2 \
  -n amortized
```

---

## 7. OpenShift OAuth

### How It Works

The `oauth-proxy` sidecar container handles SSO authentication:

1. User hits Studio Route → oauth-proxy intercepts
2. Redirects to OpenShift OAuth login (corporate SSO via LDAP/SAML)
3. User authenticates → oauth-proxy sets session cookie
4. Subsequent requests pass through to Studio/API

### Integration Pattern

```yaml
# Sidecar on the Studio deployment
containers:
  - name: oauth-proxy
    image: registry.redhat.io/openshift4/ose-oauth-proxy:latest
    args:
      - --https-address=:8443
      - --provider=openshift
      - --openshift-service-account=amortized-studio
      - --upstream=http://localhost:8080
      - --cookie-secret=<random-32-bytes>
      - --tls-cert=/etc/tls/private/tls.crt
      - --tls-key=/etc/tls/private/tls.key
    ports:
      - containerPort: 8443
```

### Recommendation

Phase 2. Bearer token auth is fine for initial deployment and testing. OAuth proxy adds SSO when the platform is shared with data scientists.

Source: [OpenShift OAuth Proxy](https://github.com/openshift/oauth-proxy)

---

## 8. Data Science Projects

### What They Are

RHOAI Data Science Projects (DSPs) are OpenShift namespaces with labels that the RHOAI dashboard recognizes. They provide:
- Namespace isolation
- S3 connection secrets (labeled `opendatahub.io/managed=true`)
- Workspace for notebooks, pipelines, model servers

### Should Amortized Map to a DSP?

**No, not initially.** Amortized should be its own namespace (`amortized` for control plane, `amortized-jobs` for compute). Reasons:
- Amortized has its own dashboard (Studio), doesn't need the RHOAI dashboard
- DSP connection secrets follow a specific format amortized doesn't need to conform to
- Mapping to DSPs adds coupling without value for Phase 1

**Phase 3** (multi-tenant): Each amortized "project" could map to a DSP namespace for isolation.

Source: [RHOAI Using Connections](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.0/html/working_on_projects/using-connections_projects)

---

## 9. Other RHOAI Components

| Component | Status on Cluster | Useful for Amortized? |
|---|---|---|
| **KubeRay** | Running | Maybe — for distributed inference or GRPO training with Ray. Not Phase 1. |
| **TrustyAI** | Running | No — AI fairness/bias monitoring is orthogonal to task model building. |
| **Feast** | Running | No — feature store for tabular ML, not relevant for LLM fine-tuning. |
| **Llama Stack** | Running | No — inference API standard. Amortized uses vLLM via KServe. |
| **Notebook Controller** | Running | No — amortized replaces the notebook workflow with agent-guided chat. |

---

## 10. Competitive Analysis

### InstructLab on OpenShift

InstructLab (`ilab-on-ocp`) follows a similar pattern to what amortized should do:
- Uses PyTorchJob for distributed training
- Integrates with RHOAI Model Registry
- Deploys models via KServe InferenceService
- Runs on the same RHOAI infrastructure

This validates amortized's integration approach — it's the established pattern for ML platforms on RHOAI.

Source: [ilab-on-ocp](https://github.com/opendatahub-io/ilab-on-ocp)

---

## What Was Refuted

These claims were adversarially tested and killed. Do NOT rely on them:

| Claim | Vote | Why Killed |
|---|---|---|
| RHOAI 3.3 ships 5 specific ClusterTrainingRuntimes | 0-3 | Could not verify from docs. Check cluster directly. |
| Trainer v2 is GA with version 2.1.0 | 0-3 | It's Technology Preview, not GA. |
| Bundled PyTorch 2.10.0, DeepSpeed 0.18.9 versions | 0-3 | Specific versions unverifiable. |
| Target Serverless mode for KServe | 0-3 | Serverless deprecated in favor of RawDeployment. |
| vLLM ServingRuntime means no image management | 1-2 | May still need custom ServingRuntime configs. |
| MLflow is Tech Preview in 3.3, GA in 3.4 | 1-2 | Could not fully verify. |
| ModelCar requires explicit ConfigMap patch | 1-2 | May work out-of-the-box on RHOAI 3.3.1. |

---

## Open Questions (Require Cluster Verification)

1. **What ClusterTrainingRuntimes actually ship with RHOAI 3.3.1?** Run `oc get clustertrainingruntimes` on the cluster to see what's available vs what needs to be created.

2. **What is MLflow's exact status?** Check `oc get csv -A | grep mlflow` and the RHOAI dashboard.

3. **Does ModelCar work without ConfigMap patching?** Try deploying with `oci://` URI and see if it works out-of-the-box.

4. **RawDeployment vs Serverless** — Which mode is configured on the cluster? Check the KServe controller config.

---

## Corrections to Existing Documentation

| Document | Issue | Correction |
|---|---|---|
| `openshift-ai-integration.md` | References "MLflow Model Registry" for model versioning | Should use Kubeflow Model Registry (already on cluster). MLflow is only for experiment tracking. |
| `openshift-ai-integration.md` | Helm chart includes PostgreSQL subchart | Not needed — amortized keeps SQLite for job queue; Model Registry has its own MySQL. |
| `openshift-deployment-plan.md` | Phase 0 includes PostgreSQL migration as "blocks everything" | Not needed for Phase 1 — SQLite with PVC is fine for single-replica. |
| `mlflow-integration-plan.md` | Phase 4 uses `mlflow.register_model()` | Should use Kubeflow Model Registry's Python client instead. MLflow only for experiment tracking. |
| `prd-openshift-deployment.md` | "No MLflow in Phase 1" is correct but reasoning is wrong | Correct decision, wrong reason. Skip MLflow because Model Registry handles model versioning. MLflow is only needed later for experiment tracking. |
| All docs | Assume Serverless/Knative for KServe | Target RawDeployment — Serverless is deprecated in RHOAI 3.3. |
