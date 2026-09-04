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

Bundled data stores (dev / self-contained, e.g. kind):

```bash
helm install amortized deploy/helm/amortized -f deploy/helm/amortized/values-kind.yaml
```

(`values-kind.yaml` relaxes `runAsNonRoot` for vanilla clusters — see the SCC note
below. On OpenShift, drop that flag and use external data stores.)

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
| `model.opencodeModel` | `google-vertex-anthropic/claude-opus-4-8@default` | model string in `opencode.json` |
| `security.runAsNonRoot` | `true` | app pods runAsNonRoot; set `false` on vanilla/kind (see `values-kind.yaml`) |
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

## SDG teacher model

`model.*` selects **Morty's** own chat model. The **SDG teacher** (the model that
generates synthetic data in dispatched SDG jobs) is configured separately via
`teacherKeys`, so the two can differ (e.g. Morty on Vertex, teacher on OpenAI).

The keys in `teacherKeys` are loaded into the amortized-server env — so
`list_models` surfaces the provider and SDG jobs get a `model_providers.yaml` — and
the `teacherKeys.forward` names are forwarded into SDG job pods.

```bash
# reference an existing secret holding OPENAI_API_KEY
helm install amortized deploy/helm/amortized \
  --set teacherKeys.existingSecret=amortized-llm-keys
```

For one OpenAI key to power both Morty and the teacher, point `model.openai` and
`teacherKeys` at the same secret. **Requires** a server image with direct-provider
support (amortized #427+); on an older image the teacher path is inert.

## OpenShift / `restricted-v2`

The bundled data stores (PostgreSQL, MinIO, MLflow) **run on OpenShift
`restricted-v2` with no `anyuid` and no cluster-admin** — the SCC injects a
non-root uid + fsGroup and the images run under it (verified: all three come up as
the injected uid on a `restricted-v2` namespace).

The bundled-store images default to **non-Docker-Hub registries**
(`mirror.gcr.io/library/*`, `public.ecr.aws/...`) so they pull on OpenShift
clusters that have no Docker Hub pull secret (anonymous Docker Hub pulls are
rate-limited). Override `postgres.image` / `minioInit.*` / `opencode.skillsInitImage`
if you mirror images elsewhere.

For production you'll still typically want **external** MLflow / S3 / PostgreSQL
(durability, scale, backups) via `dataStores.bundled=false` — but the bundled
stack is a valid self-contained install on `restricted-v2`.

**Bundled credentials are dev-only defaults** (`minio.rootUser`/`rootPassword` =
`minioadmin`, `postgres.password` = `amortized`). For anything beyond local dev,
override them (`s3.accessKey`/`s3.secretKey`, `postgres.password`) or use external
stores.

The application pods (server, studio, opencode) run with pod-level
`runAsNonRoot` (value `security.runAsNonRoot`, default `true`). On OpenShift the
SCC injects a non-root uid, so even the root-based images (the server's
`wait-for-db` init uses the postgres image; the opencode image runs as root) are
admitted under `restricted-v2`.

On a **vanilla cluster with no SCC (e.g. kind)** those root images are rejected
under `runAsNonRoot: true` (`container has runAsNonRoot and image will run as
root`). Set `security.runAsNonRoot=false` (or install with `-f values-kind.yaml`)
so they run as root there.

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
