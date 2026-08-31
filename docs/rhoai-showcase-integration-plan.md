# Amortized on RHOAI — Integration Showcase + Starter Kit (Handoff Plan)

> **For the next session**: this is a self-contained handoff. The goal is to take **three independently-built pieces** and prove them working **together** once on the `pawshift` cluster, then **package** that into a turnkey starter kit an SSA/PM can run on their own RHOAI cluster. Read "Current state (verified)" first, then follow "The prove-out plan".
>
> **Security note**: contains internal infra addresses, NO secrets. Do not paste `oc login` tokens, ADC JSON, or DB/S3 creds into this file. Keep out of public commits.
>
> **Date context**: esivaram OpenShell piece built + verified 2026-08-27; meyceoz + xingyu pieces stock-taken 2026-08-28..30. "Today" when writing = 2026-08-30.

---

## Progress log (2026-08-31 session)

Working in fresh ns **`amortized-showcase`** (single ns: server+postgres+jobs). New overlay **`k8s/overlays/rhoai/`** captures everything (turnkey `oc apply -k`).

- **Step A (amortized core) — DONE & verified.** server + sclorg-postgres healthy; migrations applied; `/api/v1/health`→`db:ok`, `/api/v1/jobs`→200, `/mcp` mounted. Mirrors xingliu's OpenShift-proven shapes (sclorg postgres, `migrate` init container), reuses central MinIO.
- **Step B (enterprise MLflow) — server path DONE & verified end-to-end.** Real `POST /api/v1/datasets/upload` → `upload` job succeeded → run + artifact confirmed in the enterprise MLflow. **Correction to this plan: Step B is NOT config-only.** It required (a) a code change — `core/mlflow_client.py` now sends `Authorization: Bearer` (SA token file, re-read for rotation) + **`X-MLFLOW-WORKSPACE: <ns>`** (this MLflow runs `--enable-workspaces`) + service-CA TLS verify; knobs in `config.py`; (b) RBAC — SSAR splits read/write, so bind **both** `mlflow-operator-mlflow-view` and `-edit` ClusterRoles to the SA; (c) deploy — server image built in-cluster (base via `mirror.gcr.io` to dodge Docker Hub rate limits). GOTCHA: the auth plugin caches SSAR 300s per token hash — use a fresh token when testing RBAC changes.
  - **Deferred:** the training job-pod path (TRL auto-log) uses mlflow 3.15.2's native `MLFLOW_TRACKING_AUTH=kubernetes-namespaced` (env-only) but needs the `kubernetes` pip pkg in the job image (unverified) + the job SA bound to view+edit. SDG/upload MLflow I/O is server-side so it didn't need this.
- **Step C (Morty via OpenShell) — DONE & verified.** amortized-server `/agent` proxy → OpenShell gateway (mTLS, client cert from `openshell-client-tls`) → sandboxed opencode; Morty replies in-character. Required (1) an agent-proxy code change — mTLS via a single `ssl.SSLContext` (httpx won't present a client cert when `cert=` + `verify=<ca path>` are passed separately → TLS1.3 CERTIFICATE_REQUIRED); (2) a pod hostAlias mapping the exposed sandbox hostname → gateway ClusterIP so the Host routes correctly (SANs cover `*.openshell.localhost`). **This was the doc's pending "3f" — NOT previously wired; the earlier proof was via CLI/nsenter (per esivaram).** Reuses the central `morty-aipcc` sandbox in ns `openshell`; `allowUnauthenticatedUsers=true` (per the RHOAI opencode kit) removes API-token auth but transport mTLS still applies.
- **Step D (Studio in RHOAI dashboard) — DONE & verified (esivaram path).** Converged on meyceoz's PR #419 (per-user hybrid: shared UI + per-user isolated backends). Research confirmed NO supported third-party MF-remote path (operator regenerates federation-config from a compiled-in registry; meyceoz's works only because that operator is OOMKilled) — so we TOOK OVER the single `amortizedStudio` federation-config entry → our `amortized-showcase` gateway. Folded our Step B (MLflow) + C (Morty) into the gateway's per-user provisioning (`studio-gateway/src/manifests.js`): each user gets an isolated `amz-<user>` backend (own MLflow workspace==ns + per-user OpenShell Morty sandbox via gateway mTLS). Model=Vertex. Verified end-to-end by simulating the dashboard flow for esivaram: gateway auto-provisioned `amz-esivaram`, Morty replied via the per-user sandbox, an upload job logged to the amz-esivaram MLflow workspace. 3 sandboxes pre-created (esivaram/xingyu/meyceoz); browser SSO login by the 3 users is the remaining human step. Work on branch `rhoai-integration` (off PR #419); details in `docs/rhoai-showcase-packaging.md`.

Details + gotchas in memory `amortized-showcase-proveout`.

---

## 0. The goal (what we're selling)

Get **amortized** running in the **RHOAI ecosystem**, integrated with three RHOAI-native technologies instead of amortized's self-contained defaults:

1. **Enterprise MLflow** (the RHOAI MLflow *operator*) instead of amortized's own bundled MLflow server.
2. **OpenShell** sandbox for the **Morty** (opencode) agent instead of a plain `opencode` Deployment.
3. **Community plugin** surfacing **Studio** inside the **RHOAI dashboard** instead of a bare Route.

End deliverable: **one working showcase** + a **starter kit** so an SSA/PM (non-expert) can stand the whole stack up with a few standard commands + a few permission grants — NOT a 20-step manual implementation.

**Strategic decision (made this session): prove-once, then package.** You cannot write turnkey instructions for an integration that has never worked end-to-end. So the next session proves the 3 pieces together in a clean namespace, capturing every command/manifest as the raw material for the kit. See §5–6.

---

## 1. How we got here (session narrative)

1. **Built the esivaram piece** — OpenShell-sandboxed Morty on `pawshift` end-to-end (agent-sandbox CRDs → gateway → in-cluster image build → sandbox → verified Morty answering via Vertex). Hit and solved a long chain of experimental rough edges (see §7). Standardized on the RHOAI `aipcc` base image (`morty-aipcc`). Full detail: `docs/openshell-morty-cluster-plan.md` + `docs/openshell-step3/` + memory `openshell-morty-pawshift`.
2. **Took stock of the meyceoz piece** — "Amortized Studio as a RHOAI dashboard community plugin." He got it working; we reverse-engineered the mechanism (Module Federation remote + iframe; see §3.3).
3. **Took stock of the xingyu piece** — "enterprise MLflow on RHOAI via the operator." Healthy and running centrally; we mapped its backend/artifacts/auth (see §3.1).
4. **Realized the pieces are 3 isolated slices** in separate projects — none wire all three together. The remaining work is **integration + packaging**, which is what this doc plans.

---

## 2. Access & environment

- **Cluster (ROSA/RHOAI)**: `pawshift`, API `https://api.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com:443`, console/dashboard `https://rh-ai.apps.rosa.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com` (the RHOAI **data-science-gateway**). `oc login --token=<USER PROVIDES>` — token is a secret, **[USER-RUN]** (via `!` prefix or their terminal; shared kubeconfig on the Mac). Session tokens expire — expect to re-login.
- **Cluster-admin**: via membership in the **`pawshift-cluster-admins`** group (bound to `cluster-admin`). `esivaram@`, `meyceoz@`, `xingliu@` are members (added 2026-08-27). IdP is **Keycloak** (groups likely IdP-synced — `pawshift-group-sync-binding`). See memory `pawshift-cluster-access`.
- **Assistant constraint**: the Bash tool hook blocks `sudo` and `rm -rf`. `oc`/`helm`/`podman` work locally; Mac `docker` daemon is down (use `podman`); `helm` installed via brew. `openshell` CLI (v0.0.113) is on the Mac.
- **Vertex** (Morty's model, POC): project `lightwell-devel`, region `global`, model `claude-opus-4-8`, ADC at `~/.config/gcloud/application_default_credentials.json`. Low quota → occasional 429/404.
- **VM prototype** (original Morty proof, docker driver): `ssh -J jump@169.62.18.122 esivaram@10.138.37.252`. Not needed for cluster work.

---

## 3. Current state (VERIFIED) of each piece

### 3.1 Enterprise MLflow (xingyu) — healthy, NOT wired, one durability risk
- **RHOAI operator MLflow**, cluster-scoped `Mlflow` CR `mlflow` (v3.14.0, `Available=True`, migration succeeded). Platform component `default-mlflowoperator` = Ready; CRDs `mlflows.mlflow.opendatahub.io` + `mlflowconfigs.mlflow.kubeflow.org`.
- **Runs centrally in `redhat-ods-applications`**: `deployment/mlflow` 2/2, RHOAI image `registry.redhat.io/rhoai/odh-mlflow-rhel9@sha256:6e71...`, SA `mlflow-sa`, `ca-bundle-watcher` sidecar (NOT oauth-proxy).
- **Backend (metadata)**: Postgres `mlflow-postgres.amortized.svc:5432/mlflow` (`sslmode=disable`). ⚠️ **`mlflow-postgres` is `emptyDir` (ephemeral)** in ns `amortized` — reschedule loses all MLflow metadata. Give it a **PVC** before any real showcase.
- **Artifacts**: `s3://amortized/artifacts` on shared MinIO `minio.amortized.svc:9000` (bucket `amortized`, prefix `artifacts/`; legacy self-hosted mlflow uses same bucket, prefix `mlflow/` — no collision). `serveArtifacts: true` → **clients need NO S3 creds** (server proxies artifact I/O). S3 creds via secret `mlflow-s3-credentials` (in `redhat-ods-applications`).
- **Tracking URL**: in-cluster `https://mlflow.redhat-ods-applications.svc:8443/mlflow` (service-serving cert); external `https://rh-ai.apps.rosa.../mlflow` via the data-science-gateway (Gateway-API `HTTPRoute/mlflow`).
- **Auth**: **Kubernetes-native** (`--app-name=kubernetes-auth`, `MLFLOW_K8S_AUTH_AUTHORIZATION_MODE=self_subject_access_review`). Clients present a **K8s bearer token** (`MLFLOW_TRACKING_TOKEN`); MLflow authorizes via SelfSubjectAccessReview against RBAC. TLS = service-serving CA (or `MLFLOW_TRACKING_INSECURE_TLS=true`).
- **Not wired**: both amortized-servers (shared `amortized` ns + `amortized-u-xingliu`) have empty/legacy `AMORTIZED_MLFLOW_TRACKING_URI`.
- Concept reminder: MLflow always has **two** stores — backend DB (Postgres) for metadata, artifact store (S3/MinIO) for files. Postgres cannot hold artifacts.

### 3.2 OpenShell Morty (esivaram/us) — running + verified, NOT wired to amortized
- ns **`openshell`**: agent-sandbox **v0.5.6** (CRD `sandboxes.agents.x-k8s.io` + controller in `agent-sandbox-system`); OpenShell gateway **Helm 0.0.113** (StatefulSet `openshell`; installed `disableTls=false`, `podSecurityContext.fsGroup=null`, `securityContext.runAsUser=null`, **`server.auth.allowUnauthenticatedUsers=true`**); `privileged` SCC on SA `openshell-sandbox`.
- **Image**: built **in-cluster** (BuildConfig `morty-aipcc`, dockerStrategy) → istag `morty:aipcc`. Base `quay.io/aipcc/agentic-ci/openshell:0.3.27` + node + Morty persona baked at `/workspace` (`docs/openshell-step3/Dockerfile.aipcc`). `opencode.json` includes the amortized MCP block.
- **Sandbox `morty-aipcc`** (CR `default--morty-aipcc`), created via `openshell -g cluster sandbox create --auto-providers --provider google-vertex-ai --env GOOGLE_* --policy policy.aipcc.yaml --from <IMAGE@digest> -- sh -c 'cd /workspace && HOME=/workspace opencode serve --port 4096 --hostname 0.0.0.0'`; ADC uploaded to `/workspace/adc.json`. **Verified**: Morty replies in-character via Vertex Opus 4.8.
- **Reachability**: opencode `:4096` is in a supervisor **sub-netns** (nothing on the pod IP). Probe with `oc exec ... -c agent -- nsenter -t <opencode_pid> -n curl localhost:4096`. External access via **`openshell service expose morty-aipcc 4096 opencode`** → `https://default--morty-aipcc--opencode.openshell.localhost:8080/` (gateway-served).
- **Model**: Vertex (POC). Could swap to in-cluster vLLM / RHOAI model-serving for a fully RHOAI-native showcase (the RHOAI opencode kit uses vLLM).
- Artifacts + full gotchas: `docs/openshell-step3/` (README has verified commands) + memory `openshell-morty-pawshift`.

### 3.3 Studio dashboard plugin (meyceoz) — works, but FRAGILE/UNSUPPORTED
- **Mechanism**: a RHOAI (3.3.0) **Module-Federation remote** named `amortizedStudio`, registered in the dashboard's **`federation-config`** ConfigMap (key `module-federation-config.json`, read via env `MODULE_FEDERATION_CONFIG` by `rhods-dashboard` in `redhat-ods-applications`). The remote renders the **real Studio inside an `<iframe>`**. NOT a ConsolePlugin, NOT an OdhApplication tile.
- ns **`cp-amortized-studio`**: the MF plugin (nginx serving `remoteEntry.js`), image built in-cluster (`amortized-studio-plugin`, Helm chart `amortized-studio-chart-0.4.2`), edge Route. Build env **`STUDIO_URL`** = the real Studio route (iframe src).
- ns **`meyceoz-amortized`**: the **real Studio** — `amortized-studio-gateway` (OpenShift `oauth-proxy` sidecar + gateway) + `amortized-studio-static` (`ghcr.io/amortized-ai/studio:latest`), reencrypt Route with OpenShift SSO.
- ns **`amortized-u-meyceoz`**: backend `amortized-server` (ClusterIP, no route) + `postgres` StatefulSet.
- **User flow**: RHOAI dashboard (SSO) → left-nav "Amortized Studio" → `/amortized-studio` → iframe → real Studio route (own SSO) → Studio SPA + gateway → amortized-server.
- ⚠️ **CRITICAL / biggest turnkey risk**: `federation-config` is normally **regenerated by the `dashboard-operator`** from a registry compiled into the operator binary (only ~8 known modules). **There is no supported CR path for arbitrary third-party plugins** — hand edits are reverted on reconcile. meyceoz's `oc replace` patch survives **only because the `dashboard-operator` is crashlooping (0/1, 307 restarts) and not reconciling.** Reproducible paths: (i) patch `federation-config` + hold off the operator, or (ii) fork/extend the operator's `modules.go` + rebuild. **CLAUDE SPECULATION**: the operator crashloop is likely pre-existing/unrelated — verify. Official refs: `github.com/opendatahub-io/odh-dashboard` (`docs/architecture.md`, `docs/dashboard-operator.md`).

---

## 4. Integration seams (where the pieces plug into amortized)

All three plug into `amortized-server` via existing config knobs (verified in `src/amortized/config.py` + `k8s/`):

| Piece | amortized seam (env) | Wire to | Notes |
|---|---|---|---|
| Enterprise MLflow | `AMORTIZED_MLFLOW_TRACKING_URI` + `MLFLOW_TRACKING_TOKEN` (+ CA or `MLFLOW_TRACKING_INSECURE_TLS`) | operator MLflow URL | **No S3 creds needed** (serveArtifacts). **Job pods** (training/SDG in `amortized-jobs`) also log to MLflow → they need the URI+token too. |
| OpenShell Morty | `AMORTIZED_AGENT_UPSTREAM_URL` (server→opencode `:4096`); `AMORTIZED_AGENT_SERVER_URL` (Studio→server) | `service expose` URL | default `http://opencode:4096`; today unset. |
| Studio plugin | `AMORTIZED_EXTERNAL_URL` + Studio Route → plugin build's `STUDIO_URL` | dashboard federation-config | iframe src = Studio route. |
| Core | `AMORTIZED_DATABASE_URL` | PostgreSQL | server job DB. |
| Storage | `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `S3_ENDPOINT_URL`, `MLFLOW_S3_ENDPOINT_URL` | MinIO | gotcha: MLflow reads `MLFLOW_S3_ENDPOINT_URL`. |

---

## 5. The prove-out plan (NEXT SESSION'S JOB)

**Principle: prove the 3 pieces together ONCE, in a clean isolated namespace, capturing every command/manifest as the packaging source. Sequence by risk (fail fast on the hard parts).**

**Setup:**
- Work in a **fresh namespace `amortized-showcase`** we own (do NOT tangle with xingyu's / meyceoz's projects — keeps the sequence clean & reproducible).
- **Reuse the central, one-per-cluster services**: the RHOAI enterprise MLflow (in `redhat-ods-applications`) and the shared MinIO (in `amortized`). Stand up per-showcase pieces fresh (amortized core, Morty sandbox, Studio+plugin).
- Confirm cluster-admin (re-login if the token expired).

**Step A — amortized core (lowest risk):** deploy `amortized-server` + PostgreSQL into `amortized-showcase` (start from `k8s/overlays/rosa` + `k8s/services/`, or crib xingyu's `amortized-u-xingliu` which already runs server+postgres). Get it healthy with SQLite-free Postgres. Verify `/api/v1` + `/mcp` reachable in-cluster.

**Step B — MLflow swap (config-only, supported):**
1. First **give `mlflow-postgres` a PVC** (durability) — coordinate with xingyu; it's her resource in the shared `amortized` ns. (Or stand up a durable Postgres for the showcase MLflow.)
2. Set on the showcase `amortized-server` (and the job template): `AMORTIZED_MLFLOW_TRACKING_URI` = `https://mlflow.redhat-ods-applications.svc:8443/mlflow`, `MLFLOW_TRACKING_TOKEN` = a K8s SA token with RBAC that passes SSAR, CA trust (or `MLFLOW_TRACKING_INSECURE_TLS=true`). Drop amortized's bundled mlflow.
3. **Verify**: run a trivial SDG or training job → confirm the run + artifacts appear in the enterprise MLflow (dashboard → MLflow at `…/mlflow`). **Open unknown**: does amortized's MLflow client path actually send `MLFLOW_TRACKING_TOKEN`? Check `src/amortized` MLflow usage; job pods too.

**Step C — Morty via OpenShell (Step 3f):**
1. Reuse `morty-aipcc` (or create a per-showcase sandbox). `openshell service expose morty-aipcc 4096 opencode`.
2. Set `AMORTIZED_AGENT_UPSTREAM_URL` on the showcase server to reach opencode. **Open unknown (the main one)**: the service-expose URL is gateway-served (auth); the amortized agent proxy does plain HTTP. Options: (a) use the gateway URL + handle auth, or (b) test a direct k8s Service to the sandbox pod + NetworkPolicy (RISK: supervisor in-pod nftables may drop inbound — unverified). Resolve empirically.
3. **Verify**: Studio chat (or a direct `POST /agent/session` + message) → Morty replies through the server proxy.

**Step D — Studio + dashboard plugin (hardest/unsupported):**
1. Deploy Studio (`ghcr.io/amortized-ai/studio:latest`) + oauth-proxy gateway in the showcase ns (reencrypt Route, OpenShift SSO). Point it at the showcase `amortized-server`.
2. Build the MF plugin (crib meyceoz's `cp-amortized-studio` chart) with `STUDIO_URL` = the showcase Studio route.
3. Register the `amortizedStudio` entry in `federation-config` (+ restart `rhods-dashboard`). **Document the reconcile-reversion behavior** (see §3.3) — this is where "turnkey" is at risk.
4. **Verify**: dashboard → "Amortized Studio" nav → iframe loads Studio.

**Step E — end-to-end showcase verify:** dashboard → Amortized Studio → chat Morty → run SDG/train → see runs/artifacts in the enterprise MLflow. That's the demo.

**Throughout**: capture each working command/manifest into overlay/chart form, and tag each step **[automate]** (goes in the kit as `oc apply -k`/Helm/script) vs **[SSA-manual]** (permission grants, token creation, one-time IdP/OIDC).

---

## 6. Packaging direction (AFTER prove-out)

- **Model to mirror**: `red-hat-data-services/agentic-starter-kits/agents/opencode` — the exact RHOAI kit pattern (kustomize `manifests/` + overlays + `Containerfile` + in-cluster `BuildConfig` + a README). Our showcase can be a sibling kit (`agents/amortized` or a new repo).
- **Turnkey target for an SSA**: `oc apply -k <overlay>` + grant a couple of SCCs/roles (`privileged` for OpenShell; RBAC for MLflow SSAR) + 1–2 `oc start-build`s + register the dashboard plugin. ~5 commands, not 20 manual steps.
- **Repo/PR decision (decide after proving, when shapes are known):**
  - amortized app repo: add an `k8s/overlays/rhoai` overlay + the RHOAI-integration env wiring.
  - a deploy/kit repo (or contribute to `agentic-starter-kits`): the OpenShell build + sandbox manifests, the Studio MF plugin chart, the bootstrap script + permission-grant list.
- **Prereqs the SSA must satisfy** (document clearly): RHOAI with the MLflow operator component enabled + dashboard; cluster-admin; a model backend (in-cluster vLLM recommended for RHOAI-native, or Vertex ADC); block storage class for PVCs.

---

## 7. Lessons learned / decisions made (do not re-discover)

**OpenShell / Morty (from building the esivaram piece):**
1. **Build the sandbox image in-cluster** (BuildConfig, amd64). Local Mac `podman` builds **arm64** → pod init `Exec format error`. And **pin the sandbox `--from` by DIGEST** — nodes cache the tag and serve a stale arch.
2. **Gateway auth on the k8s driver**: mTLS is **transport only**, NOT API auth. Install with **`server.auth.allowUnauthenticatedUsers=true`** (RHOAI-documented eval flag) OR configure OIDC. Without it every CLI call = "missing authorization header".
3. **CLI → cluster gateway**: `oc port-forward svc/openshell 8080:8080`; put the `openshell-client-tls` secret's `ca.crt`/`tls.crt`/`tls.key` in `~/.config/openshell/gateways/<name>/mtls/`; metadata `auth_mode=mtls`, endpoint `https://localhost:8080`; connect **without `--gateway-insecure`** (server SAN includes localhost; the insecure flag makes the CLI drop the client cert).
4. **Reach opencode `:4096`**: it's in a supervisor sub-netns → `nsenter` into the opencode PID's netns. External → `openshell service expose`.
5. **aipcc base deltas** vs the NVIDIA community base: no `node` (dnf install), no `/sandbox`/sandbox user (bake persona at `/workspace`, run `sh -c 'cd /workspace && HOME=/workspace opencode serve ...'`, ADC at `/workspace/adc.json`), no `ps`/`pgrep` (use `/proc`), and **opencode.exe at `/usr/local/lib/node_modules/...`** (npm prefix `/usr/local`) → the egress policy binary path + `read_write:[/workspace]` must match or Vertex egress is blocked. Smaller image (~1.5 GB vs 3.8 GB).

**Decisions:**
- Standardized Morty on the **aipcc base** (`morty-aipcc`) — RHOAI-maintained, smaller; the NVIDIA-base variant was removed. (`*.nvidia` files kept for reference.)
- **Prove-once-then-package** (not design packaging blind).
- **Sequence integration by risk**: MLflow (config, supported) → Morty (3f) → dashboard plugin (unsupported).

**MLflow:**
- Enterprise MLflow uses **K8s-token auth** + **serveArtifacts** (clients need no S3 creds). Two stores (Postgres metadata + MinIO artifacts). Its backend Postgres is currently **ephemeral** — must get a PVC.

**Dashboard plugin:**
- MF-remote + iframe is the mechanism; **no supported 3rd-party CR path**; meyceoz's works only because the dashboard-operator is crashlooping. This is the least turnkey piece — plan the reproducibility story explicitly.

---

## 8. Pointers (existing artifacts, memories, sources)

- **Repo docs**: `docs/openshell-morty-cluster-plan.md` (OpenShell build/deploy detail); `docs/openshell-step3/` (Dockerfile.aipcc, policy.aipcc.yaml, opencode.cluster.json, README with verified commands; `*.nvidia` reference); `docs/architecture.md` (amortized arch: Postgres=job DB, MLflow→S3/MinIO artifacts).
- **Memories**: `openshell-morty-pawshift` (deployment + gotchas), `pawshift-cluster-access` (RBAC/cluster-admin), `morty-adc-rotation` (kind-cluster morty ADC — older/other cluster).
- **RHOAI kit (the packaging model + opencode-on-RHOAI reference)**: `github.com/red-hat-data-services/agentic-starter-kits/tree/main/agents/opencode`.
- **Config knobs**: `src/amortized/config.py` (mlflow_tracking_uri, agent_upstream_url, external_url, database_url, gateway_url); `k8s/base/` env wiring.
- **Coordinate with**: xingyu (`xingliu@`, enterprise MLflow + the shared-ns `mlflow-postgres`), meyceoz (`meyceoz@`, Studio plugin). They actively change shared namespaces — don't clobber; the showcase should reuse their central services, not fork them.

---

## 9. Open questions / risks (to resolve during prove-out)

- **Morty↔server reachability (Step 3f)** — the main technical unknown. Gateway service-expose URL (auth) vs direct Service+NetworkPolicy (possible in-pod nftables block). Empirical.
- **amortized MLflow client K8s-token auth** — does the amortized code path (server + job pods) actually pass `MLFLOW_TRACKING_TOKEN` / trust the service CA? Verify in code; may need a small change.
- **Dashboard plugin reproducibility** — unsupported; operator-reversion. Biggest threat to "turnkey." Decide: documented manual patch + reconcile-hold, or operator fork.
- **MLflow backend PVC** — ephemeral today; must be durable for a showcase.
- **Morty model** — keep Vertex (needs ADC + egress) or swap to in-cluster vLLM (RHOAI-native, matches the RHOAI kit). Affects the SSA prereqs.
- **Namespace/tenancy model** — one `amortized-showcase` ns for the demo vs the per-user `amortized-u-<user>` pattern for the kit.
- **Auth cohesion** — 3 different auth models (dashboard SSO for Studio; K8s-token for MLflow; gateway allowUnauth/OIDC for OpenShell). Fine for a POC; note for prod.
