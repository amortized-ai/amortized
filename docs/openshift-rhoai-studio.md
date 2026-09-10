# Amortized Studio on OpenShift AI (RHOAI community plugin)

This documents the deployment of Amortized Studio on OpenShift and its
incorporation into the Red Hat OpenShift AI (RHOAI) dashboard as a community
plugin. It is **experimental / in-progress** — see "Status" at the bottom.

## Goal

Deliver the studio inside the RHOAI dashboard the most OpenShift-AI-native way,
with a **shared UI** and **per-user isolated backends**.

## Architecture (hybrid: shared UI + per-user backends)

```
Browser
  │  (RHOAI dashboard iframe → studio Route)
  ▼
Route ──▶ oauth-proxy (OpenShift SSO) ──▶ studio-gateway
                                            ├─ /api,/agent,/mcp ─▶ amortized-server.amortized-u-<user>.svc:8000   (provisioned on demand)
                                            ├─ /mlflow          ─▶ shared MLflow upstream (env MLFLOW_UPSTREAM)
                                            └─ /*               ─▶ amortized-studio-static (shared SPA nginx)
```

Why hybrid: the amortized backend has no built-in multi-tenancy — the jobs API
scopes only by `AMORTIZED_COMPUTE_NAMESPACE` (one value per server) and
MLflow/S3/opencode are global. Isolation therefore comes from running a separate
backend stack per user, not from per-user filtering in code. A shared server
would show every user everyone else's jobs/data/chats. MLflow is the intended
exception (shared, RHOAI-native, wired via `MLFLOW_UPSTREAM`).

## Components in this repo

| Path | What |
|------|------|
| `rhoai-plugin/` | The RHOAI dashboard Module-Federation plugin (seeded from the `hello-world` community template). Exposes `remoteEntry.js`; its `StudioEmbed` renders the studio in a full-bleed iframe. Nav entry: *Community plugins → Amortized Studio*. |
| `studio-gateway/` | Node/Express identity-aware gateway. Behind oauth-proxy; provisions a per-user backend stack on demand via the K8s API (`src/manifests.js`) and routes API traffic to it (`src/server.js`). |
| `k8s/overlays/openshift-hybrid/` | Shared tier: `studio-static` (shared SPA) + `gateway` (with oauth-proxy, provisioner ClusterRole, reencrypt Route). This is the hybrid deployment. |
| `k8s/overlays/openshift/` | Phase-1 single-tenant reference (server + studio + Postgres + Route in one namespace). Superseded by the hybrid overlay — **do not apply both** (Route name `amortized-studio` collides). |

## Deploy (on a cluster where you are cluster-admin)

```bash
# 1. Build images in-cluster (no local docker needed)
oc new-build --name=studio-gateway --binary --strategy=docker -n <shared-ns>
oc patch bc/studio-gateway -n <shared-ns> --type=json \
  -p '[{"op":"add","path":"/spec/strategy/dockerStrategy/dockerfilePath","value":"Containerfile"}]'
oc start-build studio-gateway --from-dir=studio-gateway --follow -n <shared-ns>

oc new-build --name=amortized-studio-plugin --binary --strategy=docker -n <plugin-ns>
oc patch bc/amortized-studio-plugin -n <plugin-ns> --type=json \
  -p '[{"op":"add","path":"/spec/strategy/dockerStrategy/dockerfilePath","value":"Containerfile"}]'
oc start-build amortized-studio-plugin --from-dir=rhoai-plugin --follow -n <plugin-ns>

# 2. Deploy the shared tier (studio-static + gateway + oauth-proxy + Route)
oc -n <shared-ns> create secret generic amortized-studio-gateway-oauth \
  --from-literal=session_secret="$(openssl rand -base64 32)"
oc apply -k k8s/overlays/openshift-hybrid

# 3. Deploy the plugin (Helm, frontend only) + register with the dashboard
helm template amortized-studio rhoai-plugin/chart \
  --set image.repository=<internal-registry>/<plugin-ns>/amortized-studio-plugin \
  --set image.tag=latest --set bff.enabled=false | oc apply -f -
# append an entry to the federation-config configmap in redhat-ods-applications,
# then: oc rollout restart deployment/rhods-dashboard -n redhat-ods-applications
```

Per-user backend stacks (`amortized-u-<user>`) are created automatically by the
gateway on first authenticated request — no manual step.

## OpenShift gotchas captured here

- Docker Hub images rate-limit on cluster nodes → use quay/ghcr/registry.redhat.io
  (Postgres = `quay.io/sclorg/postgresql-16-c9s`, arbitrary-UID friendly; do **not**
  grant `anyuid` — it breaks fsGroup volume perms under restricted-v2).
- The server has no auto-migration; provisioning runs `alembic upgrade head` as an
  init container before the server starts (its lifespan queries the `jobs` table).
- `@kubernetes/client-node` v1.x server-side-apply patch → HTTP 415; use `create()`
  with 409-tolerance.
- RBAC escalation prevention: the gateway ClusterRole must hold `pods`/`pods/log`
  and have `escalate`/`bind` to create the per-user job-manager Role.

## Status

Working end-to-end on the experimental ROSA cluster: plugin renders in the RHOAI
nav; oauth-proxy enforces OpenShift SSO; the gateway provisions and routes to
isolated per-user backends. Open items: authenticated-iframe visual confirmation
(third-party-cookie behavior with `SameSite=None`), a provisioning splash for the
~60–90s first-load, wiring `MLFLOW_UPSTREAM` to the shared MLflow, per-user
opencode/Morty, and a runtime-configurable studio embed URL.
