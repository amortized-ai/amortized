> **SUPERSEDED** — MLflow is now fully integrated. See `architecture-decisions.md` (AD-3: MLflow is the artifact store). This doc is kept for historical reference.

# MLflow Integration Plan

Deep research across 105 agents + codebase scan of amortized and asynth repos.

> **Important: MLflow vs. Model Registry on RHOAI**
>
> On Red Hat OpenShift AI, MLflow is used **only for experiment tracking** (params, metrics, artifacts, run comparison). Model versioning and the model registry are handled by a **separate system**: Kubeflow Model Registry (now Kubeflow Hub), which is already running on the ROSA cluster as `default-modelregistry`. Do not conflate the two — they serve different purposes.

## Why MLflow

Red Hat uses MLflow via OpenShift AI. Amortized will eventually be a Red Hat product. MLflow is the standard for experiment tracking in the Python ML ecosystem.

On RHOAI specifically, MLflow's role is **experiment tracking**: logging params, metrics, artifacts, and enabling run comparison. Model versioning and deployment lineage are handled by the separate **Kubeflow Model Registry** (Kubeflow Hub), not MLflow's built-in model registry.

## Current State — Amortized Has No MLflow Integration

- Zero references to `mlflow` in either amortized or asynth codebases
- `report_to: none` hardcoded in every training template
- Custom artifact table in SQLite (jobs → artifacts FK)
- Metrics stored as `training_metrics.jsonl` files, read on-the-fly
- No experiment grouping, no run comparison, no model registry
- `depends_on` field on jobs declared but never stored or enforced

## MLflow Architecture (from research)

**Two-store architecture:**
- **Backend store** — metadata (params, metrics, tags) in SQLAlchemy DB (SQLite default, PostgreSQL for production)
- **Artifact store** — large files (model weights, datasets) in pluggable storage (local, S3, GCS, Azure Blob, MinIO)

**Hierarchy:** Experiment → Run → Artifacts
- Experiment groups related runs (e.g., "ticket-classifier-training")
- Run = one training execution with params, metrics, artifacts
- URI scheme: `runs:/<run_id>/<path>`, `models:/<name>/<version>`

**Key integration point — HuggingFace MLflowCallback:**
- Built-in `transformers.integrations.MLflowCallback` implementing `TrainerCallback`
- Configured via env vars: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_RUN_ID`
- Auto-logs: training loss, learning rate, epoch, eval metrics, model checkpoints
- `HF_MLFLOW_LOG_ARTIFACTS=true` copies checkpoints to remote artifact store
- Activated by setting `report_to: mlflow` in TrainingArguments — this is what TRL passes through

**TRL integration note:** TRL doesn't have native MLflow support documented — but TRL trainers inherit from HuggingFace Trainer, so `report_to: mlflow` works through the inherited MLflowCallback. Verified against HuggingFace docs.

## Integration Plan

### Phase 1: MLflow Tracking Server + `report_to: mlflow`

**Deploy MLflow alongside amortized:**

```yaml
# docker-compose addition or separate container
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  command: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0
  ports:
    - "5000:5000"
```

For production: PostgreSQL backend + S3/MinIO artifact store.

**Inject MLflow env vars into training containers:**

```python
# worker.py — when building JobSpec for training jobs
env = {
    "MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri,  # e.g., "http://mlflow:5000"
    "MLFLOW_EXPERIMENT_NAME": f"amortized/{job.project or 'default'}/{job.id[:8]}",
    "HF_MLFLOW_LOG_ARTIFACTS": "true",  # copy checkpoints to MLflow artifact store
}
```

**Change default `report_to` from `none` to `mlflow`:**
- Update all training templates: `report_to: mlflow`
- Update the training script generators to set `report_to: mlflow` by default
- Keep `report_to: none` as a user override option

**Config:**
```python
# config.py
mlflow_tracking_uri: str = Field(
    default="",
    description="MLflow tracking server URI. Empty = MLflow disabled."
)
```

When `mlflow_tracking_uri` is empty, default `report_to` stays `none`. When set, default becomes `mlflow`.

### Phase 2: Log SDG Runs to MLflow

asynth doesn't log to MLflow, but amortized's SDG runner can:

```python
# sdg_runner.py — after synthesis completes
import mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(f"amortized/{project}/sdg")

with mlflow.start_run(run_name=f"sdg-{job_id[:8]}"):
    # Log generation params
    mlflow.log_params({
        "model": config.model,
        "num_samples": config.num_samples,
        "temperature": config.temperature,
    })
    # Log quality metrics
    if quality_report:
        mlflow.log_metrics({
            "samples_passed": quality_report.passed,
            "samples_failed": quality_report.failed,
            "pass_rate": quality_report.passed / quality_report.total,
        })
    # Log output as artifact
    mlflow.log_artifact(output_path, "generated_data")
    # Log as dataset for lineage
    dataset = mlflow.data.from_pandas(df, source=output_path, name=f"sdg-{job_id[:8]}")
    mlflow.log_input(dataset, context="training_data")
```

### Phase 3: Data Lineage via `mlflow.log_input()`

When a training job uses an SDG output as its dataset:

```python
# training runner — before training starts
import mlflow

with mlflow.start_run():
    # Log the input dataset for lineage
    dataset = mlflow.data.from_pandas(
        pd.read_json(data_path, lines=True),
        source=data_path,
        name=artifact_name,
    )
    mlflow.log_input(dataset, context="training")
    # Training proceeds — MLflowCallback handles the rest
```

This creates a lineage link: SDG run (produced dataset) → Training run (consumed dataset, produced model).

### Phase 4: Model Registry (Kubeflow Hub, NOT MLflow)

After training succeeds, register the model in **Kubeflow Model Registry** (already deployed on the ROSA cluster as `default-modelregistry`). Do not use `mlflow.register_model()` — on RHOAI, model versioning is handled by Model Registry, not MLflow.

```python
# worker.py — after training job succeeds
from model_registry import ModelRegistry

registry = ModelRegistry(
    server_address="https://default-modelregistry-rhoai-model-registries.svc:8080",
    author="amortized",
)

registered_model = registry.register_model(
    name=f"{project}/{model_name}",
    uri=f"s3://{bucket}/models/{job_id}/",
    model_format_name="vllm",
    model_format_version="1",
    description=f"Task model trained by amortized job {job_id}",
)
```

For KServe InferenceService lineage, attach registry labels to the serving manifest:

```python
isvc_labels = {
    "modelregistry/registered-model-id": registered_model.id,
    "modelregistry/model-version-id": registered_model.versions[0].id,
}
```

This gives: versioned model registry via Kubeflow Hub, KServe deployment lineage, and integration with the RHOAI model serving stack.

### Phase 5: Bridge Amortized Artifacts to MLflow

Keep amortized's existing artifact table for fast API access, but sync model metadata to **Model Registry** (Kubeflow Hub) and experiment data to **MLflow**:

```python
# After registering an artifact in amortized's DB, sync to the appropriate backend
async def register_artifact(job_id, artifact_type, path, ...):
    # Existing: save to amortized DB
    artifact = Artifact(job_id=job_id, type=artifact_type, path=path)
    db.save(artifact)
    
    # New: sync model artifacts to Model Registry (if configured)
    if artifact_type == "model" and settings.model_registry_address:
        registry = ModelRegistry(server_address=settings.model_registry_address, author="amortized")
        registry.register_model(name=f"{project}/{model_name}", uri=path, ...)
    
    # New: sync experiment data to MLflow (if configured)
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        with mlflow.start_run(run_id=job.mlflow_run_id):
            mlflow.log_artifact(path, artifact_type)
```

Amortized's DB remains the fast read path (Studio API queries). Model Registry is the model versioning backend (Kubeflow Hub, KServe lineage). MLflow is the experiment tracking backend (run comparison, metrics, Red Hat ecosystem integration).

### Phase 6: Studio UI — Link to MLflow

Add "Open in MLflow" and "Open in Model Registry" buttons on:
- Job detail page → "Open in MLflow" links to the MLflow experiment run
- Model detail page → "Open in Model Registry" links to the Kubeflow Hub model version
- Dataset detail page → "Open in MLflow" links to the MLflow dataset

URL patterns:
- MLflow: `{MLFLOW_TRACKING_URI}/#/experiments/{exp_id}/runs/{run_id}`
- Model Registry: `{MODEL_REGISTRY_UI}/model-registry/{model_id}/versions/{version_id}`

Don't duplicate these UIs — link to them. MLflow shows experiment runs and metric charts. Model Registry shows model versions and deployment lineage.

## Deployment Patterns

### Self-hosted (development)
```
amortized server (port 8000) + MLflow server (port 5000) + SQLite for both
```

### Self-hosted (production)
```
amortized server + MLflow server + PostgreSQL (shared or separate) + S3/MinIO for artifacts
```

### Red Hat OpenShift AI
```
amortized deployed on OpenShift + Model Registry from RHOAI operator (Kubeflow Hub) + MLflow for experiment tracking (optional, may be RHOAI-managed in 3.4+) + S3-compatible object storage
```

## What NOT to Do

1. **Don't replace amortized's artifact system** — keep it for fast API access. MLflow is the lineage backend, not the primary artifact store.
2. **Don't build custom experiment comparison UI** — link to MLflow's UI instead.
3. **Don't make MLflow required** — it should be optional. When `mlflow_tracking_uri` is empty, everything works without it.
4. **Don't use OpenLineage** — MLflow's `log_input()` + model registry provides sufficient lineage. OpenLineage adds complexity with no additional value for this use case.
5. **Don't use MLflow Model Registry for model versioning** — use Kubeflow Model Registry (already deployed as `default-modelregistry`). MLflow and Model Registry serve different purposes on RHOAI: MLflow tracks experiments, Model Registry versions and serves models.

## Sources

- MLflow architecture: https://mlflow.org/docs/latest/self-hosting/architecture/overview/
- MLflow artifact store: https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/
- HuggingFace MLflowCallback: https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.MLflowCallback
- MLflow Datasets API: https://mlflow.org/docs/latest/ml/tracking/data-api/
- MLflow Model Registry: https://mlflow.org/docs/latest/model-registry/
