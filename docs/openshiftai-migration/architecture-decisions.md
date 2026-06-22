# Architecture Decisions — Amortized on OpenShift

Grilled and agreed 2026-06-22. These decisions guide all implementation.

## What Amortized Is

Amortized is a **thin orchestration layer** for building task models on OpenShift.

It does NOT: store artifacts (MLflow does), track experiments (MLflow does), manage S3 (platform engineer does), build container images (CI does), schedule GPUs (K8s does), generate Python scripts (TRL/asynth CLIs accept config).

It DOES: translate user intent → tool-native YAML configs, create K8s Jobs/Deployments, track job lifecycle, store mlflow_run_id per job, guide users via agent chat, expose MCP for AI agents.

## Decisions

### AD-1: Plug into infrastructure, don't bundle it

Amortized connects to existing S3 and MLflow. It does not deploy them.

```
k8s/base/    ← amortized only (server, studio, RBAC, configmap)
k8s/dev/     ← MinIO + MLflow for development/testing
```

Platform engineer provides `AMORTIZED_S3_ENDPOINT` and `AMORTIZED_MLFLOW_TRACKING_URI`. Amortized doesn't care what's behind them.

### AD-2: Single code path, no branching

The worker generates config YAML for every backend — K8s, SSH, local. No `is_k8s` branches. No generated Python scripts (`_trl_trainer_script()`, `_eval_script()`). Every job type uses its tool's native CLI with a YAML config:

- Training: `trl {algo} --config config.yaml`
- SDG: `asynth synthesize --config config.yaml`
- Eval: `asynth judge --config config.yaml`
- Serve: `vllm serve --config config.yaml`

The backend protocol handles the differences (how config is delivered, how containers are launched).

### AD-3: MLflow is the artifact store

All artifacts flow through MLflow. Amortized does not manage artifacts itself.

- SDG: `mlflow.log_artifact(data.jsonl)` + `mlflow.log_input()` for lineage
- Training: `report_to: mlflow` + `HF_MLFLOW_LOG_ARTIFACTS=true` (automatic via TRL)
- Eval: `mlflow.log_metrics()` + `mlflow.log_artifact(results)`
- Serve: reads model from MLflow artifact URI (S3 path)

Amortized stores `mlflow_run_id` per job. Everything else — artifact storage, versioning, lineage — is MLflow's responsibility.

**Eliminated from amortized:** `artifacts` table, `artifact:<uuid>` references, `register_artifacts_for_job()`, `_register_s3_artifacts()`, `core/artifacts.py` pattern matching.

### AD-4: Two namespaces

`amortized` (control plane) and `amortized-jobs` (GPU compute). The isolation is worth the cross-namespace complexity. GPU jobs shouldn't share RBAC with the control plane.

### AD-5: TRL 1.5.0 with thin custom image

Pin to `huggingface/trl:1.5.0` (latest working — 1.5.1+ has `kernels` bug). Add `mlflow` + `boto3` in a thin Dockerfile layer. Rebuild when TRL fixes the bug.

Eventually migrate to Kubeflow Trainer SDK / training-hub.

### AD-6: Polling for now, push-based later

Worker polls K8s Job status every 2s. Migrate to push-based callbacks (eval-hub sidecar pattern) when scaling requires it.

### AD-7: No serve monitor on K8s

K8s Deployments are self-healing. The worker doesn't spawn a background monitor for serve jobs on K8s. Status queries go directly to the K8s API.

### AD-8: Reuse K8s API client

Create the `kubernetes_asyncio.ApiClient` once and reuse it across all operations. Don't create a new client per call.

## Work Items

### Architectural (do first)

1. **Split manifests** — `k8s/base/` + `k8s/dev/`
2. **Unify worker code path** — config-only for all backends, remove `is_k8s` branches and all generated Python script functions
3. **MLflow artifact integration** — asynth#19 for SDG/eval, fix TRL `report_to: mlflow` version compatibility
4. **Remove amortized artifact tracking** — drop artifact table usage, add `mlflow_run_id` to job records

### Tactical (do second)

5. **Fix TRL image** — pin mlflow version compatible with TRL 1.5.0's transformers
6. **Training output to S3** — `aws s3 sync` chained with TRL CLI (interim until MLflow `report_to` works)
7. **Reuse K8s API client** — create once in `__init__`
8. **Skip serve monitor on K8s** — don't spawn `_monitor_serve_job()` for K8s backend

## Container Images

| Job Type | Image | Custom? | Why |
|---|---|---|---|
| Training | `ghcr.io/amortized-ai/trl:1.5.0` | Thin layer over stock | Adds `mlflow` + `boto3` |
| SDG | `ghcr.io/amortized-ai/asynth:latest` | Our project | asynth + mlflow + s3fs |
| Eval | `ghcr.io/amortized-ai/asynth:latest` | Our project | Same as SDG |
| Serve | `docker.io/vllm/vllm-openai` | Stock | No customization |

## Artifact Flow

```
SDG Run → mlflow.log_artifact(data.jsonl) → MLflow → S3
  │
  └─ mlflow.log_input(dataset, context="training_data")
                │
Training Run → mlflow.log_input(sdg_dataset) → lineage link
  │
  └─ report_to: mlflow → auto-logs model + metrics → MLflow → S3
                │
Eval Run → mlflow.log_input(model, context="eval") → lineage link
  │
  └─ mlflow.log_metrics({accuracy, f1}) → MLflow
                │
Serve → reads model from MLflow artifact URI (s3://...) → vLLM
```

Full traceability in MLflow UI. Amortized never tracks artifacts itself.
