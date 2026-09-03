# amortized (core install)

Helm chart for the Amortized core stack: a single `helm install` that stands up
the control plane in one namespace on Kubernetes or OpenShift. It wraps the
manifests in `k8s/base/` and `k8s/services/`.

This is the Phase-1 **core install**: single-tenant / per-namespace. Multi-tenant
provisioning, the OpenShift dashboard tile, and OpenShell are separate follow-ups
and are intentionally not included here.

## What it installs

| Component | Kind | Notes |
|---|---|---|
| Namespaces | Namespace | `namespace` + `jobsNamespace` (toggle with `createNamespaces`) |
| Server | Deployment, Service, PVC, ServiceAccount | the amortized API/control plane |
| Jobs RBAC | Role, RoleBinding | `amortized-job-manager` in `jobsNamespace` |
| Studio | Deployment, Service, (Route) | the web UI; optional OpenShift Route |
| OpenCode agent | Deployment, Service, ConfigMap | runs "Morty"; persona + skills mounted from ConfigMaps |
| Morty persona/skills | ConfigMap | `morty-config` + `morty-skills`, built via `.Files.Glob` |
| Config | ConfigMap | `amortized-config` server env |
| S3 creds | Secret | `amortized-s3` (in both namespaces) |
| Model creds | Secret | templated from `model.*` |
| Data stores (bundled) | StatefulSet/Deployment/Service/PVC/Job | PostgreSQL, MinIO (+ bucket init), MLflow |

## Quick start

Bundled data stores (dev / self-contained):

```bash
helm install amortized deploy/helm/amortized
```

External data stores (production brings its own MLflow / S3 / PostgreSQL):

```bash
helm install amortized deploy/helm/amortized \
  --set dataStores.bundled=false \
  --set database.url='postgresql://user:pass@db.example.com:5432/amortized' \
  --set mlflow.trackingUri='https://mlflow.example.com' \
  --set mlflow.gatewayUrl='https://mlflow.example.com/gateway/mlflow/v1' \
  --set s3.endpoint='https://s3.us-east-1.amazonaws.com' \
  --set s3.bucket='my-bucket' \
  --set s3.accessKey='...' --set s3.secretKey='...'
```

Install into a pre-existing namespace instead of having the chart create it:

```bash
helm install amortized deploy/helm/amortized \
  -n my-ns --set namespace=my-ns --set jobsNamespace=my-ns-jobs \
  --set createNamespaces=false
```

## Key values

| Key | Default | Description |
|---|---|---|
| `namespace` | `amortized` | app namespace |
| `jobsNamespace` | `amortized-jobs` | namespace training/SDG jobs run in |
| `createNamespaces` | `true` | render the two Namespace objects |
| `global.storageClass` | `""` | applied to all PVCs; empty = cluster default |
| `images.{server,studio,opencode}` | see values | `{repository, tag, pullPolicy}` |
| `server.persistence.size` | `200Gi` | server data PVC |
| `server.doclingUrl` | `""` | optional `AMORTIZED_DOCLING_URL` |
| `studio.host` | `""` | Route host (empty = auto) |
| `studio.route.enabled` | `false` | create an OpenShift Route for Studio |
| `dataStores.bundled` | `true` | deploy PostgreSQL + MinIO + MLflow in-namespace |
| `model.provider` | `vertex` | `vertex` or `openai` |
| `model.opencodeModel` | `google-vertex-anthropic/claude-opus-4-6@default` | model string in `opencode.json` |
| `mlflow.trackingUri` / `mlflow.gatewayUrl` | `""` | external MLflow (when not bundled) |
| `s3.bucket` / `s3.endpoint` / `s3.accessKey` / `s3.secretKey` | see values | object storage |
| `database.url` | `""` | external PostgreSQL (when not bundled) |

See `values.yaml` for the full list.

## Model provider

`model.provider` selects how the OpenCode agent authenticates to the LLM:

- **`vertex`** (default, mirrors the base): the agent uses Google Vertex AI via
  Application Default Credentials. It reads the `opencode-llm` Secret
  (`google-cloud-project`, `vertex-location`) and mounts the `opencode-gcp`
  Secret (`credentials.json`). These are expected to already exist in the
  namespace. You may optionally have the chart template them by setting
  `model.vertex.project` + `model.vertex.location` and/or
  `model.vertex.credentials`.
- **`openai`**: set `model.openai.apiKey` (stored as `OPENAI_API_KEY` in a
  Secret) or point `model.openai.existingSecret` at an existing Secret that holds
  `OPENAI_API_KEY`, and set `model.opencodeModel` to an OpenAI-compatible model.

## OpenShift SCC note (bundled data stores)

The bundled data-store images (PostgreSQL, MinIO, MLflow) run as **root**
(`runAsNonRoot: false`), matching the base manifests. On OpenShift the default
`restricted-v2` SCC forbids this, so the bundled stores will not start there.
On OpenShift either:

- set `dataStores.bundled=false` and point at external MLflow / S3 / PostgreSQL
  (recommended for production), or
- grant an SCC that permits running as root (e.g. `anyuid`) to the namespace's
  default ServiceAccount.

The application pods (server, studio, opencode) keep the base's non-root
`securityContext` and are compatible with `restricted-v2`.

## Morty persona & skills

The `morty-config` and `morty-skills` ConfigMaps are built from the files under
`files/` via `.Files.Glob`, reproducing the base `configMapGenerator` behavior —
including the base's `__` -> `/` path flattening for skills (the opencode init
container reverses it to rebuild the skills tree). The content under `files/` is
generated from the repo's `agents/` directory (the same source the `make prompt`
target uses for the kustomize build); this chart does not alter persona/skill
content.

## Not included (follow-ups)

- Multi-tenant / per-user provisioning (studio-gateway, provisioner)
- OpenShift dashboard tile (`OdhApplication`)
- OpenShell / agent-sandbox
