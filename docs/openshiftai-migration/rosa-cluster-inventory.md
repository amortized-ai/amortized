# ROSA Cluster Inventory

Cluster scanned on 2026-06-18.

## Cluster Details

| Field | Value |
|---|---|
| **Platform** | ROSA (Red Hat OpenShift Service on AWS) |
| **OpenShift Version** | 4.20.21 |
| **RHOAI Version** | 3.3.1 |
| **Region** | us-east-2 |
| **API Server** | https://api.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com:443 |
| **Dashboard** | https://data-science-gateway.apps.rosa.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com/ |
| **User** | ssudalai@redhat.com |

## GPU Nodes

4 worker nodes, each with 1x NVIDIA L40S GPU (48GB VRAM):

| Node | Instance Type | GPU | VRAM |
|---|---|---|---|
| ip-10-0-1-104 | g6e.4xlarge | NVIDIA L40S | 48GB |
| ip-10-0-1-135 | g6e.4xlarge | NVIDIA L40S | 48GB |
| ip-10-0-1-72 | g6e.4xlarge | NVIDIA L40S | 48GB |
| ip-10-0-1-80 | g6e.4xlarge | NVIDIA L40S | 48GB |

Total: 4 GPUs, 192GB aggregate VRAM. L40S supports BF16, FP8, INT8.

Training capacity:
- Qwen3-0.6B LoRA: fits easily on 1 GPU
- Qwen3-4B LoRA: fits on 1 GPU (~20GB)
- Qwen 2.5 7B QLoRA: fits on 1 GPU (~16GB with 4-bit)
- Llama 3.1 8B LoRA: fits on 1 GPU (~24GB)
- Llama 3.1 8B full SFT: needs 2+ GPUs or QLoRA

## RHOAI Components — What's Running

| Component | Status | Namespace | Notes |
|---|---|---|---|
| **RHOAI Operator** | Running (3 replicas) | redhat-ods-operator | v3.3.1 |
| **RHOAI Dashboard** | Running (2 replicas) | redhat-ods-applications | 5 containers per pod |
| **KServe** | Running | redhat-ods-applications | Controller deployed, 2 InferenceServices active |
| **Model Registry** | Running | rhoai-model-registries | `default-modelregistry`, operator deployed. Kubeflow Hub (v1alpha3 REST API). NOT MLflow Model Registry. |
| **Kubeflow Training Operator** | Running | redhat-ods-applications | Ready for PyTorchJobs, no jobs submitted yet |
| **KubeRay** | Running | redhat-ods-applications | Ray operator for distributed workloads |
| **Feast** | Running | redhat-ods-applications | Feature store operator |
| **TrustyAI** | Running | redhat-ods-applications | AI trustworthiness/fairness |
| **Llama Stack** | Running | redhat-ods-applications | Llama Stack K8s operator |
| **Notebook Controller** | Running | redhat-ods-applications | For Jupyter notebooks |
| **ODH Model Controller** | Running | redhat-ods-applications | Model lifecycle management |

## RHOAI Components — What's NOT Running

| Component | Status | Impact for Amortized |
|---|---|---|
| **MLflow** | Not deployed | MLflow is for experiment tracking only (loss curves, hyperparams). Model versioning is handled by Model Registry (already running). MLflow deployment is Phase 2. |
| **Kueue** | Not installed (CRDs don't exist) | GPU scheduling will use direct K8s Jobs with nodeSelector instead. Can install Kueue separately if needed. |
| **Data Science Pipelines** | Not configured (no DSPApplication) | Not needed — amortized orchestrates its own workflow. |
| **S3 Object Storage** | No bucket configured in DS projects | Need to create an S3 bucket (AWS) or deploy MinIO for artifact storage. |

## Existing Workloads

### InferenceServices (KServe)

| Name | Namespace | Model | Runtime | Storage | GPU | Status |
|---|---|---|---|---|---|---|
| qwen35-35b-a3b | llama-serving | vLLM | vllm-runtime | PVC (llama-model-storage) | 1x | Ready |
| redhataigemma-2-9b-it | test-shiv | vLLM | custom | OCI (registry.redhat.io modelcar) | 0 (CPU only) | Not Ready |

### ServingRuntimes

| Namespace | Name | Format | Container |
|---|---|---|---|
| llama-serving | vllm-runtime | vllm | kserve-container |
| test-shiv | redhataigemma-2-9b-it | vLLM | kserve-container |

### Data Science Projects

| Namespace | Age | Contents |
|---|---|---|
| test-shiv | 12 days | Gemma 2 9B InferenceService, 1 secret |
| test-kai | 85 days | Unknown |
| yi-workbench | 69 days | Unknown |

## Storage

| StorageClass | Provisioner | Reclaim | Binding | Default |
|---|---|---|---|---|
| gp2-csi | ebs.csi.aws.com | Delete | WaitForFirstConsumer | No |
| gp3-csi | ebs.csi.aws.com | Delete | WaitForFirstConsumer | **Yes** |

EBS volumes only. No S3/MinIO configured on-cluster.

## What's Ready for Amortized Testing

### Ready now
- KServe for model serving (InferenceService CR)
- Model Registry for model versioning (Kubeflow Hub, `default-modelregistry`)
- Kubeflow Training Operator for PyTorchJobs
- 4x L40S GPUs (48GB each)
- EBS storage (gp3-csi) for PVCs

### Needs quick setup
- MLflow tracking server (deploy as pod + route)
- S3 bucket or MinIO (for MLflow artifact store + training data)
- `amortized` namespace/project

### Not needed initially
- Kueue (direct K8s Jobs work without it)
- Data Science Pipelines (amortized orchestrates)
- Feast (not relevant for task models)
