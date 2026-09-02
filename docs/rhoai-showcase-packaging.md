# Amortized-on-RHOAI Showcase — Packaging / PR Ledger

> Living record of every change made during the prove-out and where it must land:
> which **repo PR** (amortized app / amortized-deploy) and which **cluster-side** step
> (one-time SSA action, permission grant, build, secret, gateway op). Populated as the
> prove-out proceeds in ns `amortized-showcase`. Companion to
> `docs/rhoai-showcase-integration-plan.md` and memory `amortized-showcase-proveout`.

Legend: [DONE] applied+verified in the showcase · [WIP] in progress · [TODO] not started

---

## A. `amortized` app repo (code + manifests)

### Code
- [DONE] `src/amortized/config.py` — MLflow auth knobs: `mlflow_tracking_token_file`,
  `mlflow_workspace`, `mlflow_ca_bundle`, `mlflow_tracking_insecure_tls`.
- [DONE] `src/amortized/core/mlflow_client.py` — `_MLflowAuth`: per-request bearer token
  (re-read from token file for rotation) + `X-MLFLOW-WORKSPACE` header + CA/verify; applied
  to every request via `_client()`. Enables the RHOAI enterprise (kubernetes-auth) MLflow.
- [DONE] `src/amortized/api/agent.py` (+ `config.py`) — agent-proxy mTLS to the OpenShell
  gateway. New knobs: `agent_upstream_client_cert`, `agent_upstream_client_key`,
  `agent_upstream_ca_bundle`, `agent_upstream_insecure_tls`. `startup()` builds the shared
  httpx client via `_upstream_client_kwargs()`. GOTCHA (httpx 0.28.1): passing `cert=` +
  `verify=<ca path>` separately does NOT present the client cert (server aborts TLS 1.3 with
  CERTIFICATE_REQUIRED); the fix builds ONE `ssl.SSLContext` loading both the CA and the client
  cert, passed as `verify=`. Host routing needs NO code — URL host + a pod hostAlias (manifests).
  Backward-compatible: empty knobs = plain HTTP (default opencode Service). lint+mypy clean.

- [DONE] `src/amortized/backends/kubernetes.py` — OpenShift-compat bug fix (found during browser
  test): the dispatched-job `work` volume was a **hostPath** (`/var/local-path-provisioner/...`, a
  kind-cluster artifact), which the OpenShift restricted SCC forbids → EVERY dispatched job
  (document/SDG/training) was rejected pre-schedule (`FailedCreate: hostPath volumes are not allowed`).
  Changed to `emptyDir`. General app-repo fix, not showcase-specific. (Dataset upload was unaffected —
  it writes to MLflow in-process, no K8s job.)
- [DONE] `src/amortized/worker.py` + NEW `src/amortized/_mlflow_job_sitecustomize.py` — **job-pod
  MLflow auth (was BLOCKER #1)**. Dispatched job pods log to the enterprise MLflow with a *vanilla*
  `mlflow` client (the `mlflow` CLI in pre/post commands + TRL `report_to: mlflow` auto-logging), which
  failed `CERTIFICATE_VERIFY_FAILED` (no service-CA trust) and would 400 (`Workspace context is
  required`) — the enterprise MLflow needs a bearer token **and** an `X-MLFLOW-WORKSPACE` header, and
  mlflow 3.15.2 (the client in the job images) has **no** built-in way to send a custom header (zero
  `request_auth_provider` entry points; `MLFLOW_TRACKING_AUTH=kubernetes-namespaced` does NOT exist —
  the earlier memory assumption was wrong). Fix without rebuilding the 3 job images: `_inject_job_mlflow_auth()`
  ships a `sitecustomize.py` (auto-imported via a prepended `PYTHONPATH=/amortized` pre-command) that
  registers an MLflow `RequestHeaderProvider` adding `Authorization: Bearer <token>` (re-read per
  request → handles projected-token rotation) + `X-MLFLOW-WORKSPACE`; TLS trust is core mlflow
  (`MLFLOW_TRACKING_SERVER_CERT_PATH` → the service CA delivered as a config file, or
  `MLFLOW_TRACKING_INSECURE_TLS`). Gated on `mlflow_tracking_token_file` set (enterprise MLflow only) →
  self-hosted MLflow path unchanged. Job pods run as SA `default`, so `default` must be bound to
  mlflow view+edit (see cluster-side below). VERIFIED end-to-end: a real `POST /api/v1/documents/convert`
  → upload job pod did `mlflow artifacts download` + 3× `log-artifact(s)` cleanly; job `succeeded`,
  document FINISHED + visible in the amz-esivaram MLflow workspace. lint+mypy clean.
- [DONE] `src/amortized/api/models.py` + `agent/skills/sdg/task-distillation/reference-payload.json` —
  **`list_models` drop-in models (was BLOCKER #2, stopgap)**. This MLflow distribution (server 3.14.x)
  has NO AI Gateway (`/api/3.0/.../gateway/endpoints/list` → 501), so `list_models` returned `[]` and
  Morty couldn't populate SDG `model_configs`. `list_models` now falls back (when the gateway is
  unavailable/empty) to a **direct-provider catalog** built from data-designer's own builtin providers
  (`openai`/`nvidia`/`openrouter`): a provider's models appear only once its API-key env var is set on
  the server (`_provider_key_available`), and every model is one the SDG job can actually call. Reference
  payload no longer hardwires `provider: "gateway"` — it takes `model` AND `provider` from `list_models`.
  Verified locally: `OPENAI_API_KEY` set → returns `gpt-4.1`+`gpt-5` (provider `openai`); unset → `[]`.
  Stopgap "for the time being" until a proper MLflow-gateway alternative. lint+mypy clean.
  Cluster: `AMORTIZED_FORWARD_ENV=["OPENAI_API_KEY"]` (forwards the key to job pods) + an optional
  `OPENAI_API_KEY` secretKeyRef (secret `amortized-llm-keys`) on the amz-esivaram server. Key dropped in
  out-of-band (user-created secret; not in git). PACKAGING TODO: fold both into `k8s/overlays/rhoai/`
  + the gateway per-user provisioner.

### Manifests
- [DONE] `k8s/overlays/rhoai/` — new self-contained overlay for the showcase (single ns:
  server+postgres+jobs; sclorg postgres; `migrate` init; reuse central MinIO; enterprise
  MLflow env + service-CA mount; view+edit RBoles). NOTE: decide final home — this repo's
  `k8s/overlays/rhoai` (per plan §6) vs the `amortized-deploy` repo. See open question below.

### Build
- [NOTE] `Dockerfile` base `python:3.12-slim` is Docker-Hub rate-limited during in-cluster
  builds. Prove-out used `mirror.gcr.io/library/python:3.12-slim` in the build-context copy
  only (repo Dockerfile untouched). Packaging decision: make base a build ARG, or publish a
  release image so SSAs don't build.

---

## B. `amortized-deploy` repo (overlays / deploy)
- [OPEN] Where the showcase overlay ultimately lives (here as a `users/`-style or `rhoai`
  overlay, vs in the app repo). amortized-deploy currently carries per-user overlays that
  wire a PLAIN `opencode` Deployment; the OpenShell-sandboxed Morty wiring is new.

## B2. Studio-in-dashboard — CONVERGED ON PR #419 (meyceoz), Option 2 (per-user hybrid)
Decision: per-user hybrid (each user gets an isolated `amz-<user>` backend), per-user
OpenShell Morty, Vertex model, take over the single `amortizedStudio` federation entry.
Work on branch `rhoai-integration` (off PR #419's `feat/rhoai-studio-plugin`) = PR source + our B/C.
- [DONE] Fold B+C into per-user provisioning — `studio-gateway/src/manifests.js`: adds MLflow
  (workspace==ns, token-file, CA mount, view+edit RoleBindings) + Morty (per-user gateway mTLS:
  AGENT_UPSTREAM=`https://default--morty-<user>--opencode.openshell.localhost:8080`, mounts
  openshell-client-tls, hostAlias). `provision.js` nsForUser → `amz-<user>` prefix (avoids
  colliding with existing `amortized-u-*`). node --check + render validated.
- [DONE] 3 per-user OpenShell Morty sandboxes (esivaram/xingyu/meyceoz) in ns `openshell`:
  reuse `morty:aipcc`, per-user egress policy (allows `amortized-server.amz-<user>.svc:8000`),
  `USER_NS` sed rewrites opencode.json MCP host, Vertex ADC, `service expose`. esivaram verified
  reachable (opencode healthy + MCP host = amz-esivaram) via gateway mTLS.
- [DONE] Built `studio-gateway` + `amortized-studio-plugin` images in-cluster (UBI Containerfiles)
  → internal registry `amortized-showcase/{studio-gateway,amortized-studio-plugin}:latest`.
- [DONE] Deployed shared tier — new overlay `k8s/overlays/rhoai-hybrid/shared-tier.yaml` (ns
  `amortized-showcase`; DISTINCT cluster-scoped RBAC names `amz-*` so meyceoz's bindings untouched;
  gateway env AMORTIZED_SERVER_IMAGE=internal B/C image, SHARED_MLFLOW_TRACKING_URI=enterprise,
  OPENSHELL_MTLS_DIR=/etc/openshell-mtls + OPENSHELL_GATEWAY_IP=172.30.119.72 + mount
  openshell-client-tls; PLUGIN_UPSTREAM; EMBED_BASE_PATH). oauth cookie secret created out-of-band.
- [DONE] Cross-ns image-puller (`system:image-puller` to `system:serviceaccounts` on amortized-showcase).
- [DONE] Provisioner ClusterRole `amz-studio-gateway-provisioner` includes `bind` on the two
  `mlflow-operator-mlflow-*` ClusterRoles (verified: per-user RoleBindings created on provision).
- [DONE] Plugin deployed + TOOK OVER the `amortizedStudio` federation-config entry → our
  gateway-http svc; restarted rhods-dashboard. Gateway serves remoteEntry.js on the dashboard path (200).
- [DONE/VERIFIED - esivaram path] Simulated the dashboard flow (forward esivaram token → gateway):
  auto-provisioned `amz-esivaram` with full wiring (MLflow workspace==ns, view/edit RB, mTLS secret,
  hostAlias); Morty answered in-character via the per-user sandbox; a real upload job logged a run to
  the amz-esivaram MLflow workspace. xingyu/meyceoz sandboxes pre-created; their backends provision on
  first login (same verified mechanism).
- [PENDING - human] Browser SSO login by the 3 users (iframe render + third-party-cookie SameSite=None
  behavior + ~60-90s provisioning splash are browser-only, per meyceoz's PR open items).
- [DONE] `/mlflow` UI proxy auth-injection — `studio-gateway/src/server.js`: the `/mlflow/**` proxy
  now injects the gateway SA token + per-user `X-MLFLOW-WORKSPACE`=`nsForUser(user)` and trusts the
  service CA (`MLFLOW_CA_FILE`); `MLFLOW_UPSTREAM`=`https://mlflow.redhat-ods-applications.svc:8443`.
  Gateway SA bound cluster-wide to `mlflow-operator-mlflow-{view,edit}`. Verified: with esivaram's
  identity, `/mlflow/api/.../experiments/search` → 200 returning the `amz-esivaram` workspace's
  experiments; no-identity → 400 (no workspace). So "see in MLflow" opens the enterprise MLflow
  scoped per-user. HARDENING TODO: strip any client-supplied `X-MLFLOW-WORKSPACE`/`Authorization` on
  `/mlflow` before injecting (real path is SSO-gated, so not currently exploitable).
- [DONE] Long-turn 504 fix (browser test): delegating Morty turns (orchestrator + SDG subagent +
  Vertex) exceed the dashboard ingress default timeout → the blocking `POST /agent/.../message` 504s
  even though the server completes it (response is fetchable via the UI's `GET .../message` poll).
  Bumped `rhods-dashboard` HTTPRoute `timeouts.request`+`backendRequest` to `300s` (owned by the
  OOMKilled Dashboard operator, so it persists). Shared-route edit (affects all dashboard traffic;
  permissive/low-risk).
- [DONE - code + deployed, needs browser verify] Long-turn 504 robust fix (blocker #3):
  `src/amortized/api/agent.py` — `POST /session/{id}/message` now runs the turn in a background task
  (`_run_turn`) keyed by a `turn_id` and returns `{turn_id,status:"processing"}` immediately; new
  `GET /session/{id}/turn/{turn_id}` returns `{active,result,error,error_status}` (result = the exact old
  POST shape). Per-session `lock` still serializes turns; last MAX_TURNS_TRACKED=8 results retained.
  `studio/src/lib/api-client.ts` — `sendOpenCodeMessage` POSTs then polls `/turn` until done via
  `pollTurn` (short GETs), returns the same shape → `use-chat`/rendering/`proposedAction` (SDG card)
  unchanged. Backward-compatible: if the server returns a full body (no `turn_id`) it's used directly,
  and the turn error_status preserves the stale-session recreate+replay retry. No single request is long
  → dashboard embed proxy no longer 504s. Gateway needs NO change (`/agent/**` glob already covers `/turn`).
  Deployed: server digest 4c38c59 on amz-esivaram; NEW in-cluster BuildConfig `amortized-studio` (builds
  `studio/` via Dockerfile.kind w/ mirror.gcr.io bases) → `amortized-studio-static` repointed off
  `ghcr.io/amortized-ai/studio:latest` to the internal image (served bundle confirmed to contain the
  `/turn` poll). studio-static is SHARED across users; the new client is backward-compatible so this is
  safe. VERIFIED via curl: POST returns in ~1s + `/turn` poll yields the reply. Browser SSO test PENDING
  (can't be done from CLI). PACKAGING: fold studio image build + repoint into the overlay/pipeline;
  currently ghcr `studio:latest` would overwrite on next provision if not addressed.
- Follow-ups: on-demand sandbox creation in the gateway (pre-created 3 now); ghcr publish for the kit;
  2 PMs; mlflow-postgres PVC durability.
- Working branch `rhoai-integration` (off PR #419) holds gateway/manifests changes; NOT committed/pushed.

---

## C. Cluster-side (one-time; SSA action / permission grant / build / secret / gateway op)

- [DONE] Namespace `amortized-showcase` + core deploy via `oc apply -k k8s/overlays/rhoai`.
- [DONE] MLflow RBAC — bind ClusterRoles `mlflow-operator-mlflow-view` + `-edit` to SA
  `amortized-server` (namespaced RoleBindings, in the overlay). Grants SSAR read+write in the
  workspace (== namespace).
- [DONE] MLflow RBAC for JOB pods — job pods run as SA `default`, so also bind `default` to
  `mlflow-operator-mlflow-{view,edit}` in the job namespace (RoleBindings `amortized-job-mlflow-{view,edit}`).
  Required by the blocker-#1 fix (the job's own projected `default` token is the bearer). PACKAGING TODO:
  add these to `k8s/overlays/rhoai/` and to the gateway per-user provisioner (`studio-gateway/src/manifests.js`),
  which currently binds only `amortized-server`. (Least-privilege alternative: a dedicated `amortized-job`
  SA + `AMORTIZED_COMPUTE_SERVICE_ACCOUNT` — deferred; binding `default` in the isolated per-user ns is fine.)
- [DONE] Enterprise MLflow reuse — no S3 creds needed (serveArtifacts). Service CA auto-present
  (`openshift-service-ca.crt`), mounted for TLS verify.
- [DONE] In-cluster image build (BuildConfig `amortized` → internal registry
  `amortized-showcase/amortized:showcase`).
- [DONE] OpenShell Morty reachability — VERIFIED: Morty replies in-character through the
  amortized-server `/agent` proxy → gateway (mTLS) → sandboxed opencode → Vertex.
  1. `openshell service expose morty-aipcc 4096 opencode` (gateway route) — DONE. Needs the
     openshell CLI pointed at the gateway (`oc port-forward svc/openshell 8080` + mTLS certs).
  2. Copy secret `openshell-client-tls` (ca.crt/tls.crt/tls.key) from ns `openshell` into
     `amortized-showcase` (out-of-band; NOT in git) — DONE.
  3. Server pod: mount that secret at `/etc/openshell-mtls` + hostAlias
     `default--morty-aipcc--opencode.openshell.localhost` → gateway ClusterIP (172.30.119.72 —
     cluster-specific, templatize) + env `AMORTIZED_AGENT_UPSTREAM_URL` (the exposed https URL) +
     `_CLIENT_CERT`/`_CLIENT_KEY`/`_CA_BUNDLE`. In the overlay.
  gateway server cert SAN includes `*.openshell.localhost` so verify against ca.crt passes (no insecure).
  Deployed + verified end-to-end (`/agent/health` healthy; Morty answers via `/agent/session`).
- [TODO] Studio + dashboard plugin (Step D). RESEARCH DONE (upstream odh-dashboard @ main,
  corroborated on-cluster): NO supported third-party MF-remote path. `federation-config`
  (key module-federation-config.json) is generated by `dashboard-operator` from a hardcoded Go
  registry (`internal/controller/modules.go`), written via SSA + field-ownership + content-hash
  restart (`module_deploy.go`) → hand-edits revert on reconcile. `spec.modules` only toggles
  registry-known modules. meyceoz's works only because the operator is **OOMKilled** (256Mi limit,
  1021 restarts; deploy owned by DataScienceCluster/default-dsc → memory bump would be reverted AND
  a healthy operator would revert the module). Runtime host IS generic (`frontend/config/
  moduleFederation.js` loads any MODULE_FEDERATION_CONFIG entry; backend auto-proxies) — only the
  operator overwrite blocks a third-party remote.
  Options (ranked): (1) custom operator build with `amortizedStudio` in the registry — supported,
  reconcile-safe, but requires shipping a forked dashboard-operator image + overriding it at the
  platform level (invasive on a SHARED cluster); (2) hand-edit + keep operator from reconciling —
  what meyceoz did; inherently fragile, no granular opt-out (Dashboard CR only Managed/Removed);
  (3) OdhApplication tile — supported-ish, clean, but a tile/link not embedded nav (unverified the
  modular dashboard still renders tiles); (4) upstream RFE for a merge-from-labeled-CM extension —
  correct long-term, no immediate unblock.

### Prereqs / grants an SSA must have (from prove-out)
- cluster-admin (or scoped grants): create ns, RoleBindings to the mlflow-operator ClusterRoles.
- RHOAI with MLflow operator component + dashboard; OpenShell gateway + agent-sandbox installed
  (privileged SCC on `openshell-sandbox`); a block storage class for PVCs.

---

## Browser test findings (2026-08-31) + next-session blockers
WORKS: dashboard→Community plugins→Amortized Studio renders in iframe; SPA + /api (→amz-esivaram) +
/mlflow proxy all route; Morty first turn works (per-user OpenShell sandbox); /mlflow auth-injection done.
Blockers (priority for the chat→SDG→MLflow demo):
1. **[RESOLVED 2026-08-31] Job-pod MLflow TLS/auth** — dispatched job pods now authenticate to the
   enterprise MLflow. Root cause was different from the earlier guess: mlflow 3.15.2 (the client in the
   job images) has NO `MLFLOW_TRACKING_AUTH=kubernetes-namespaced` provider (zero request_auth entry
   points), and there is no core env for a custom header, so the required `X-MLFLOW-WORKSPACE` header
   could not be sent. Fixed by shipping a `sitecustomize.py` `RequestHeaderProvider` (bearer token
   re-read per request + workspace header) via `PYTHONPATH`, plus core-mlflow TLS trust of the service
   CA (delivered as a config file). See `worker.py`/`_mlflow_job_sitecustomize.py` in section A above.
   Cluster-side: SA `default` bound to mlflow view+edit in the job namespace. VERIFIED: real document
   upload job logged download + all artifacts to the amz-esivaram workspace; job succeeded.
2. **[RESOLVED 2026-08-31, stopgap] SDG teacher model / model discovery** — decision: direct provider
   keys (OpenAI), NOT the (absent) MLflow AI Gateway. `list_models` now returns data-designer's builtin
   provider catalog filtered to providers with a configured key; drop in `OPENAI_API_KEY` → OpenAI models
   appear and are forwarded to SDG job pods. See section A (`api/models.py`). "For the time being" until a
   proper gateway alternative. Enabled only after the user creates the `amortized-llm-keys` secret + server
   restart. 2nd fix (live-run): the job IMAGE's data-designer only knows a `gateway` provider (KIND
   cluster's bundled MLflow gateway), so `jobs/sdg.py` writes a `model_providers.yaml` (from
   `core/model_catalog.py`) into the pod + `DATA_DESIGNER_HOME` + a copy pre_command. VERIFIED end-to-end
   with a funded key: SDG generated 3 records via gpt-4.1 → dataset logged to the amz-esivaram MLflow
   workspace (visible via `/api/v1/datasets`). New file `src/amortized/core/model_catalog.py` (shared by
   `api/models.py` + `jobs/sdg.py`). NOTE: OpenAI 429 is ambiguous — data-designer labels no-credits
   (`insufficient_quota`) as `rate_limit`; diagnose via a direct `chat/completions` call.
3. **[UX] Long agent turns 504 in the embed** — dashboard's internal Fastify proxy timeout (HTTPRoute/Envoy
   bump to 300s did NOT fix; downstream cap). Server completes; UI polls recover. Real fix = `api/agent.py`
   async. Workaround: standalone Studio Route (timeout 300s, bypasses the dashboard proxy).
NOTE: the amortized worker is SERIAL (one dispatched job at a time).

## D. Starter-kit packaging kinks (must make seamless)

Non-obvious traps hit during the prove-out — each needs a real fix so an SSA doesn't rediscover them:

1. **Studio SPA is BAKED into the `studio-gateway` image** (`studio-dist`, built with `VITE_BASE_PATH=/amortized-studio-embed`), served directly by the gateway. `amortized-studio-static` (ghcr `studio:latest`) exists but is NOT what the dashboard iframe loads — it's effectively vestigial for the embed. **Updating the Studio frontend requires rebuilding `studio-gateway` from the repo ROOT** (needs both `studio/` and `studio-gateway/`), NOT `studio-static`. Cost this session: updated the wrong component first. Fix for kit: document this clearly, or split the studio into its own served image + have the gateway proxy it (STUDIO_STATIC_UPSTREAM is set but unused by the deployed gateway).
2. **The deployed `studio-gateway` src is UNCOMMITTED** — `server.js`/`manifests.js`/`provision.js` in the running pod DIFFER from `rhoai-integration` branch (cluster-side prove-out edits: federation take-over, per-user provisioning). Rebuilding from the branch would REGRESS them. Had to extract deployed `/opt/app-root/src/src` from the pod to rebuild. Fix for kit: **commit the deployed gateway src to the branch/repo** — it is currently the only source of truth and lives only in the pod.
3. **Job images bake a default `gateway` model provider** → `mlflow.amortized.svc:5000/gateway` (the KIND cluster's bundled MLflow AI Gateway). Absent here → "No provider named 'openai' registered". Worked around by writing `model_providers.yaml` into the pod (`jobs/sdg.py` + `core/model_catalog.py`). Fix for kit: rebuild job images with a sane default provider set, or make the provider file injection first-class.
4. **Per-user wiring is manual per-namespace**: `default`-SA MLflow view/edit RoleBindings, the `amortized-llm-keys` secret, and `AMORTIZED_FORWARD_ENV`. Must move into the gateway per-user provisioner (`manifests.js`) so every `amz-<user>` gets them on provision. Until then only `amz-esivaram` works end-to-end.
5. **Caching**: the gateway serves the Studio `index.html` with no `Cache-Control` → after a redeploy the browser runs the OLD bundle until a hard-refresh (manifested as `e.parts is undefined` = old client hitting the async server). Add `Cache-Control: no-cache` on the SPA HTML so updates are seamless. (Hashed assets stay immutable-cacheable.)
6. **Async agent change spans server + gateway(studio) images** — both must be rebuilt. The Studio client is backward-compatible (falls back to a sync-server body), so rollout order is flexible, but a stale Studio + async server breaks chat (see #5).
7. **In-cluster builds need rate-limit-safe bases**: Docker Hub (`python:3.12-slim`, `node:22-alpine`, `nginxinc/*`) rate-limits the cluster egress IP; use `mirror.gcr.io/...` or UBI (`registry.access.redhat.com`). The gateway/plugin Containerfiles already use UBI.
8. **`oc set image` needs the CONTAINER name, not the deployment name** (e.g. `studio`, `gateway`, `server`) — a wrong name silently no-ops the image update.
9. **Left-behind from this session**: unused BuildConfig+imagestream `amortized-studio` (created while chasing the wrong component) — remove or repurpose.

## Deferred / durability
- [RISK] `mlflow-postgres` (ns `amortized`, xingyu's) is `emptyDir` — needs a PVC before any
  real showcase.
