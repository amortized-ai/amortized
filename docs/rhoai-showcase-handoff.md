# RHOAI Showcase — Session Handoff (2026-08-31)

For a fresh session to fix the remaining pieces. Companion to
`docs/rhoai-showcase-packaging.md` (§D = packaging kinks) and memory
`amortized-showcase-proveout`. Cluster: **pawshift**, user backend ns **`amz-esivaram`**,
sandboxes in ns **`openshell`**, shared tier in ns **`amortized-showcase`**.

## >>> NEXT SESSION — START HERE. GOAL: PMs run Amortized Studio on THEIR RHOAI cluster (turnkey) ASAP.

**Mission:** enable Mustafa (`meyceoz`) + Xingyu to stand up Amortized Studio on a fresh RHOAI cluster with a **healthy operator** (not relying on pawshift's dead dashboard-operator). Two workstreams, both required: **7a (install + reconcile-safe plugin registration)** and **meta#3 (automated per-user provisioning)**. *(MLflow artifact-storage OOM = §5 is owned by another dev — out of scope here.)*
**Definition of done:** a documented, repeatable path (scripted/manifested where possible; a clean manual runbook otherwise) that on a healthy-operator cluster: (1) installs the plugin + registers it so it **survives operator reconcile**, and (2) a per-user SSO login auto-provisions a fully-working Studio (backend + MLflow workspace + Morty). No step may depend on the operator being down.

### Task 1 (7a.1 — the registration blocker; DO THIS FIRST)
We proved (healthy RHOAI 3.3.1, `lab-cluster2`) that the RHODS-doc `oc set env MODULE_FEDERATION_CONFIG=<literal>` override is **NOT** clean: the `dashboard` controller owns that env via SSA and re-asserts `valueFrom`, so a literal `value:` collides (invalid) → Dashboard `Ready=False (Error)`. Editing the `federation-config` CM also reverts. Details: §7a, §10-A.
**Find the REAL reconcile-safe registration by reading the actual code (not the doc one-liner):**
- `github.com/rh-ai-community-plugins/community-plugins-admin` — its **BFF/Helm** registration path. Does it SSA-force-apply (take env-field ownership)? strip `valueFrom` and set `value:` cleanly? How does it bootstrap ITSELF without breaking the reconcile? (docs/architecture `PLUGIN_SYSTEM.md`, `BFF_PATTERN.md`; the code.)
- `github.com/rh-ai-community-plugins/charter` — `docs/plugin-spec.md` (the supported `plugin.yaml` contract; Mustafa's plugin already conforms — `rhoai-plugin/plugin.yaml`).
- `github.com/opendatahub-io/odh-dashboard` — does the controller support a **community/labeled-CM MERGE** hook in federation-config generation (esp. newer 3.4.x, matching pawshift)? Look at the modular-arch onboarding docs + the federation-config controller code.
**Deliverable:** the correct way to register `amortizedStudio` (embedded nav) that survives a healthy operator → **validate on `lab-cluster2`** (healthy 3.3.1; esivaram provides oc login; full backup + restore — see the tested procedure/gotchas below) → then apply on pawshift + document for the PMs' cluster.
**Fallback if no clean embedded-nav path exists:** `OdhApplication` tile (supported-ish — verify the modular dashboard renders tiles); a launcher tile that opens Studio is acceptable for turnkey.

### Task 2 (meta#3 — automate per-user provisioning; details in §3)
Fold ALL the currently-manual per-user wiring into the gateway provisioner (`studio-gateway/src/manifests.js` + `provision.js`) so a fresh `amz-<user>` login provisions FULLY with zero manual steps:
- Add: `default`-SA MLflow RoleBindings; teacher-model keys (provider-agnostic — a shared secret the gateway stamps per-ns) + `AMORTIZED_FORWARD_ENV`.
- **Automate the Morty sandbox**: templated `Sandbox` CR + **secret-mounted ADC** (not ephemeral `/workspace/adc.json`) + **FQDN egress policy** (`amortized-server.amz-<user>.svc.cluster.local` — short `.svc` → 403, opencode silently loses MCP tools) + `service expose`. Parameterize model creds (Vertex ADC vs Anthropic key).
- Replace hardcoded cluster values with **discovery**: `OPENSHELL_GATEWAY_IP` (look up `svc/openshell` ClusterIP), DNS resolver, MLflow ClusterRole names, MLflow CA, `workspace==ns`. Full list: §3.

### Already working — do NOT redo
App fixes deployed + verified (server `f933958`, gateway `174d33b4`): stateless MCP, SDG gateway→provider remap, async agent (no 504), MLflow-UI base path. 3 tester backends (`amz-esivaram`/`amz-xingliu`/`amz-meyceoz`) SDG-verified. Commits on `rhoai-integration` (worktree `~/workspace/amortized-pr419`) → **PR #420** (stacked on Mustafa's #419).

### Small user-facing bugs to also fix for the PMs (§9 — quick, same base-path class)
- Morty option-card "view dataset/job/model" 404 in embed: `job-monitor-card.tsx` uses absolute `href="/jobs…"` → use React Router `navigate`.
- "View in MLflow" opens unscoped/blank: `mlflowUiHref` omits `?workspace=<ns>` → append it.

### Resources / access / gotchas
- **This doc** (§1-§10) is the backlog + hand-patches ledger (§10). Memory: `amortized-showcase-proveout`. Packaging: `docs/rhoai-showcase-packaging.md` (§B2/§C/§D).
- Clusters: **pawshift** (`api.h0p8o2c2b7w3r1y…`; operator OOMKilled — our hand-patches survive there but WON'T on a healthy cluster, see §10-A) + **lab-cluster2** (`api.lab-cluster2.qrj7…`; healthy RHOAI 3.3.1 — use for reconcile-safety tests). oc tokens expire (~hours); esivaram re-logs.
- openshell CLI: `oc -n openshell port-forward svc/openshell 8080` + `~/.config/openshell/gateways/cluster/mtls`; `exec` needs `-n <name>`; `download` restricted to `/sandbox` (ADC source = local `~/.config/gcloud/application_default_credentials.json`, lightwell-devel). morty:aipcc digest + sandbox recipe: bottom of this doc.
- **lab-cluster2 dashboard test (safe, reversible):** back up `rhods-dashboard` Deployment env + `federation-config` CM → mutate → force reconcile via `oc annotate dashboard default-dashboard …` → observe → restore by `oc patch` the env entry back to `valueFrom: configMapKeyRef(federation-config/module-federation-config.json)` (container[0]/env[0]) → force reconcile → confirm `Dashboard Ready=True`.

---

## 2026-09-01 UPDATE — most gaps CLOSED (see the original handoff below for history)

Fixed + deployed + verified this session (branch `rhoai-integration` = PR #419;
commits `a4d538f`, `aab485a`, `3279a97`; server digest `f933958`, gateway `174d33b4`
live in `amz-esivaram` + the shared tier). Browser-verified by esivaram (SDG e2e works):

- **MCP statelessness** (top meta-issue): fastapi-mcp forced to stateless → server
  redeploys no longer break Morty's opencode tools.
- **gap #1 — SDG `gateway` provider**: `jobs/sdg.py` remaps the baked `gateway`
  provider to the primary enabled one; a real chat→SDG→MLflow run succeeds.
- **gap #2 — "View in MLflow" blank**: anchors now base-prefixed via `mlflowUiHref()`
  → the workspace-scoped MLflow UI opens in the embed (root cause was the missing SPA
  base on the anchor, NOT the workspace header, which the gateway does inject).
- **gap #4 — embed 504**: async agent (turn_id + poll) deployed.

## ISSUE / TODO BACKLOG (curated — for a future session to fix properly)

> Living backlog, grouped by area. Each item: what's wrong, root cause, where, proper fix.
> Nothing here blocks the current chat+SDG tester round unless flagged. Append new issues
> under the right section. **Tester setups `amz-esivaram` / `amz-xingliu` / `amz-meyceoz`
> are all wired + SDG-verified as of 2026-09-01.**

### 1. Ad-hoc stopgaps to REMOVE (band-aids over stale baked images / the missing model gateway)
Added to make the showcase work; delete once the root causes (§2, §4) are fixed.
**Sequencing: fix the baked images/gateway FIRST, then remove these — else chat→SDG breaks.**
- **SDG provider remap** `gateway→<enabled>` — `src/amortized/jobs/sdg.py` (commit `a4d538f`). Rewrites any model_config `provider` not backed by a key to the primary enabled one. Papers over the baked skill's hardcoded `provider: "gateway"`.
- **`model_providers.yaml` injection + `DATA_DESIGNER_HOME`** — `src/amortized/jobs/sdg.py` (commit `7588b15`). Papers over the job image's baked `gateway` default provider.
- **`list_models` direct-provider fallback** + `src/amortized/core/model_catalog.py` — `src/amortized/api/models.py` (commit `7588b15`). Papers over the absent MLflow AI Gateway → make first-class or replace with a real gateway (§4).

### 2. Morty image / skill rebuild (root cause of most of §1 + the gpt-5 bug)
The **running morty image's baked skills are stale** (old copy; differ from the repo's #397 revamp). From inside a sandbox, `/workspace/skills/sdg/knowledge-ingestion/reference-payload.json` hardcodes:
- `provider: "gateway"` → should take `provider` from `list_models` (fixing this removes §1's remap).
- `temperature: 0.7` → **breaks gpt-5** (reasoning models reject non-default temperature: *"Unsupported value: 'temperature' does not support 0.7 … only default (1)"*; confirmed on failed jobs `df6c23f4`/`f11fba54`). Don't hardcode temperature, or make it model-aware. (esivaram declined a server-side shim — fix at the skill.)
- Proper fix: decide the canonical skill source + **rebuild the morty image** (ties to §3), then recreate sandboxes.

### 3. meta#3 — turnkey per-user provisioning (BIGGEST piece; must generalize to any OpenShift/RHOAI cluster)
Gateway provisioner (`studio-gateway/src/manifests.js` + `provision.js`) creates the backend but not the full wiring, and hardcodes cluster values.
- **Missing from the provisioner (currently hand-done per user):**
  - `default`-SA MLflow RoleBindings (`amortized-job-mlflow-{view,edit}`) — job-pod MLflow auth.
  - Teacher-model keys: `amortized-llm-keys` secret + `OPENAI_API_KEY` secretKeyRef env + `AMORTIZED_FORWARD_ENV`. Make **provider-agnostic** (list of key env names).
  - The **Morty sandbox itself** — 100% manual (openshell CLI). Automate as a templated `Sandbox` CR the gateway applies.
- **Morty creds delivery:** replace the ephemeral `/workspace/adc.json` upload with a **mounted Secret** (restart-safe). Parameterize provider: Vertex-Anthropic (ADC + project/location) vs direct **Anthropic key** vs other. Current default model = `google-vertex-anthropic/claude-opus-4-8`.
- **Hardcoded cluster values to generalize:**
  - `OPENSHELL_GATEWAY_IP=172.30.119.72` (shared-tier → hostAlias) → **discover** `svc/openshell` ClusterIP (or use DNS).
  - `DNS_RESOLVER=172.30.0.10` → discover cluster DNS IP.
  - Morty landlock egress host MUST be the **FQDN** `amortized-server.amz-<user>.svc.cluster.local` — short `.svc` → **403** and opencode silently loses its MCP tools.
  - MLflow ClusterRole names `mlflow-operator-mlflow-{view,edit}` (RHOAI-specific); MLflow CA `openshift-service-ca.crt` (OpenShift-specific); `workspace==namespace` assumes `--enable-workspaces`.
  - internal-registry image refs; DB creds `amortized/amortized` (generate a secret); GPU quota (`nvidia.com/gpu`, 1/user); storage class (cluster default).
  - `EMBED_BASE_PATH=/amortized-studio-embed` must match the dashboard federation entry (paired constant).

### 4. Model-provider story (decide the proper approach; removes §1's "fallback" framing)
No MLflow AI Gateway on this cluster. Decide: stand up a gateway alternative (e.g. a LiteLLM proxy) vs. make the direct-provider path first-class. Affects `list_models`, the job provider config, and teacher-key delivery (§3).

### 5. Training
- **Artifact-upload OOM (root-caused).** The RHOAI operator-managed, cluster-wide MLflow (`MLflow` CR, 3Gi limit, `--serve-artifacts` proxy) is **OOMKilled** uploading the ~1.6GB `model.safetensors` (osft full model, Qwen3.5-0.8B). Same MinIO backend as KIND (`minio.amortized.svc:9000`); KIND worked only because its MLflow had no memory cap. **All artifact traffic goes through MLflow (AD-3; #344 removed the S3 bypass) — do NOT reintroduce direct-S3.** Options: (a) raise the operator MLflow memory via its CR (platform/cluster-admin; won't scale to 4B/9B); (b) [explore] `max_shard_size` so each proxied file stays under the cap (app-side, scales).
- Full training e2e still unverified (GPU available on the cluster).

### 6. Gateway polish (small, batch into one gateway rebuild)
- `studio-gateway/src/manifests.js:36` stale JSDoc example `amortized-u-meyceoz` → `amz-meyceoz`.
- **[SECURITY]** `/mlflow` proxy: strip any client-supplied `X-MLFLOW-WORKSPACE`/`Authorization` before injecting (defense-in-depth; SSO-gated today).
- `Cache-Control: no-cache` on the SPA `index.html` (avoids stale-bundle-after-redeploy; §D-5).
- Standalone route `/` → `/amortized-studio-embed/` redirect (white-page fix).

### 7. Install & ownership boundary — what AMORTIZED sets up vs. true external prereqs
Reframed (esivaram, 2026-09-01): standing up the amortized stack on a RHOAI cluster is **amortized's responsibility** — the starter-kit code/setup today, a future **`amortized controller`/operator** long-term — NOT a prereq. Only genuinely external platform infra are prereqs.

**7a. Amortized install responsibilities (starter kit today → candidate `amortized controller` jobs):**
- **Community dashboard plugin install.** Build + deploy the plugin + `studio-gateway` (shared tier) AND **register it in the dashboard**. Registration is the hard part: **the `federation-config` takeover is UNSUPPORTED** — `module-federation-config.json` is generated by the dashboard-operator from a compiled-in Go registry; our hand-patch (`amortizedStudio` → `amortized-studio-gateway-http`) survives only because that operator is OOMKilled. Real paths (ranked, packaging §C research): (1) custom dashboard-operator build with `amortizedStudio` in the registry (reconcile-safe but invasive on a shared cluster); (2) hand-edit + keep operator from reconciling (current; fragile); (3) OdhApplication tile (supported-ish; a tile, not embedded nav); (4) upstream RFE for a labeled-CM merge extension. **[#1 generalization blocker.]** Currently written as manual [DONE] steps in packaging §B2 — needs to become first-class kit/controller install. **Tested 2026-09-01 (healthy RHOAI 3.3.1, lab-cluster2):** the RHODS-community-doc `oc set env MODULE_FEDERATION_CONFIG=<literal>` override is NOT clean — it collides with the controller's SSA `valueFrom` ownership and errors the Dashboard reconcile (see §10-A). So this is still a real blocker; the path is the **rh-ai-community-plugins framework** (`community-plugins-admin` + `charter`, which Mustafa's `plugin.yaml` already targets) — next step is to validate that tool's *actual* registration mechanism (not the doc's one-liner) on the target RHOAI version.
- **OpenShell for sandboxed Morty.** Install the OpenShell gateway + agent-sandbox (privileged SCC, mTLS certs). It exists to sandbox Morty (an amortized feature), so roll it up as an amortized-controller install job — not a bare prereq.
- **Per-user provisioning** — the gateway provisioner (meta#3, §3): backend + MLflow RBAC + teacher keys + Morty sandbox.
- The shared tier itself (gateway/plugin images, oauth-proxy, cross-ns RBAC, secrets).

**7b. True external prerequisites (platform must already provide; amortized does NOT own — see Q1/Q2 MLflow boundary):**
- RHOAI dashboard + MLflow operator (+ `--enable-workspaces`) + a **durable** cluster MLflow (backend DB + artifact store).
- MinIO/S3 (or the MLflow's artifact backend).
- cluster-admin for cross-ns RBAC / CRD installs.

> **`amortized controller` (future):** an operator whose job is to install/reconcile all of 7a (plugin registration, OpenShell, per-user provisioning) — the clean long-term home for these responsibilities, so a fresh RHOAI cluster gets amortized turnkey.

### 8. Housekeeping / status caveats
- `get_turn` MCP exclude (`aab485a`) committed but not yet in the deployed server image (rides the next server build; cosmetic).
- PR **#420** (stacked on #419) — needs review/merge coordination with Mustafa's #419.
- **`mlflow-postgres` durability — NOT amortized's concern (flag to the MLflow owner).** The cluster MLflow's backend/metadata store is `postgresql://mlflow-postgres.amortized.svc:5432/mlflow` — a **hand-rolled** postgres in ns `amortized` (no ownerRefs/labels, created 2026-08-27; not controller-managed) on **`emptyDir`**. A pod restart loses ALL run/experiment metadata (MinIO artifacts survive but get orphaned). Per AD-1 (plug into infra, don't own it), amortized should **assume a durable MLflow is provided**; the emptyDir is a setup shortcut by whoever stood up the enterprise MLflow (xingyu) → flag to them, don't fix in amortized. (Recorded here only as a known showcase-env risk; see §7 prereqs.)
- **Stale `amortized-u-*` backends coexist with `amz-*`.** `amortized-u-{meyceoz,xingliu}` (meyceoz's original PR #419 dev backends, ~4d old) still exist alongside the new `amz-*`. The dashboard federation routes to OUR gateway (`amortized-studio-gateway-http`@`amortized-showcase` → `amz-<user>`), so testers' NEW data lands in `amz-<user>`; but the MLflow workspace dropdown (gateway SA has cluster-wide MLflow access) also shows the old `amortized-u-*`, causing confusion (Mustafa picked `amortized-u-meyceoz`). Fixing §9's `?workspace=` auto-scoping sidesteps it; also consider deleting the stale `amortized-u-*` backends. (esivaram has no `amortized-u-esivaram`, so he didn't see the extra option.)

### 9. Studio (frontend) — embed base-path navigation bugs
- **Morty option-card "view" actions 404 in the embed** (reported 2026-09-01, xingliu + meyceoz). Clicking Morty's **"view dataset"** (after SDG finishes), **"view job"** (on job start), or **"view model"** navigates to the RHOAI dashboard's "We can't find that page" — and can render the whole dashboard *inside* the Studio iframe ("rhoai inside rhoai"). Root cause: `studio/src/features/chat/components/job-monitor-card.tsx` uses **raw absolute anchors** — `href="/jobs?job=…"` (L218), `href={mlflowRunId ? "/models?run=…" : "/models"}` (L226), `href="/datasets?job=…"` (L233) — which bypass the SPA router basename (`/amortized-studio-embed`), so the browser does a full-page nav to the dashboard origin instead of SPA-routing. (Other chat navs use React Router `navigate()` and work.) **Fix:** use React Router (`Link`/`useNavigate`) for these SPA-internal routes (or base-prefix them). Same class as the gap #2 MLflow-link fix (`mlflowUiHref`). Embed-only — standalone (base `/`) is unaffected. Small, high-visibility (breaks a core demo interaction).
- **"View in MLflow" opens a blank/unscoped MLflow home** (reported 2026-09-01, Mustafa; esivaram hit it too). The enterprise MLflow UI (`--enable-workspaces`) needs a `?workspace=<ns>` param to scope; `mlflowUiHref` (`studio/src/lib/api-client.ts`) builds `/mlflow/#<hash>` **without** it, so MLflow opens with **no workspace selected → blank home** until the user manually picks their workspace from the dropdown. Fix: append `?workspace=<user-workspace>` to the URL. The studio must know its workspace (== the backend's `AMORTIZED_MLFLOW_WORKSPACE` == namespace) — expose it (e.g. an `/api` config field) and append it. Must agree with the gateway's injected `X-MLFLOW-WORKSPACE` on the `/mlflow` proxy. (Pairs with the `mlflowUiHref` base-path fix already shipped in gap #2.)

### 10. Hand-patches ledger (out-of-band cluster changes NOT in committed manifests)
Best-effort consolidation from our records + this session — **not guaranteed exhaustive**; a definitive audit = diff the live cluster vs the manifests. Everything here must be reproduced by the kit / `amortized controller` (§7a) or it won't survive a fresh (or healthy-operator) cluster.

**A. Operator-owned resources we patched → THESE REVERT ON A HEALTHY CLUSTER (dangerous):**
- **`federation-config` CM `amortizedStudio` entry** (plugin registration). Owned by the dashboard controller (SSA; regenerates the CM from a built-in registry AND owns the Deployment's `MODULE_FEDERATION_CONFIG` env, bound `valueFrom: configMapKeyRef`). Currently survives on pawshift only because the dashboard-operator is OOMKilled. **CORRECTION (tested 2026-09-01 on a healthy RHOAI 3.3.1, lab-cluster2): the RHODS-doc `oc set env … MODULE_FEDERATION_CONFIG=<literal>` override does NOT cleanly work** — the controller re-asserts `valueFrom`, colliding with the literal `value:` (`value`+`valueFrom` is invalid) → the controller's apply is rejected and the Dashboard goes `Ready=False (Error)`. So the env-override is NOT a clean reconcile-safe fix (at least on 3.3.1). Real path = the rh-ai-community-plugins framework (community-plugins-admin + charter; Mustafa's plugin.yaml targets it) — but its *actual* registration mechanism must be validated (the doc's one-liner is insufficient); still likely needs proper SSA field-ownership or a controller merge-hook / newer dashboard version.
- **`rhods-dashboard` HTTPRoute `timeouts.request/backendRequest=300s`** (the 504 workaround). Owned by `Dashboard/default-dashboard` (opendatahub-operator). **Likely OBSOLETE** now (the async-agent fix removed long single requests) → reverting on a healthy cluster should be fine; verify async fully covers it before relying on that.

**B. One-time cluster setup (not operator-owned, but not in committed manifests → kit must reproduce):**
- `amortized-studio-gateway-oauth` cookie secret (out-of-band).
- `openshell-client-tls` secret copied from ns `openshell` into the shared tier + each per-user ns.
- Cross-ns image-puller RBAC (`system:image-puller` → `system:serviceaccounts` on `amortized-showcase`).
- Gateway SA cluster-wide `mlflow-operator-mlflow-{view,edit}` bindings (for the `/mlflow` UI proxy).
- `amortized-studio` Route (ns `amortized-showcase`, reencrypt TLS).
- In-cluster image builds (BuildConfigs `amortized`/`studio-gateway`/`amortized-studio-plugin`) + `oc set image` deploys (server `f933958`, gateway `174d33b4`) — internal-registry images, not a published kit registry.

**C. Per-user manual wiring (also in §3 meta#3 — recorded, not yet in the provisioner):**
- `amz-<user>`: `amortized-llm-keys` secret + `OPENAI_API_KEY` env + `AMORTIZED_FORWARD_ENV` + `default`-SA MLflow RoleBindings.
- Morty sandboxes (openshell CLI: create + ADC upload + `service expose` + FQDN egress policy).

**Recorded today across:** packaging §C (cluster-side) / §B2 / §D; this backlog §3/§7; memory `amortized-showcase-proveout`. **TODO:** converge all of A/B/C into kit install manifests + the `amortized controller` so none are manual, and run a live-vs-manifests diff to catch anything this ledger missed.

---

## TL;DR

We started from a working chat+embed stack whose only gaps were (a) dispatched jobs
failed to log to MLflow and (b) no teacher models were selectable. Those two were fixed
and verified. Then an async-agent change (for the embed 504) triggered repeated server
rebuilds, which cascaded into MCP-session breakage and a broken/rebuilt Morty sandbox.
Net current state: **core server fixes are solid; the browser demo has 4 open gaps** (below).

## Deployed state (as of handoff)

| Component | Image / state | Notes |
|---|---|---|
| `amz-esivaram` amortized-server | `amortized@sha256:d0b2311` (SYNC, known-good) | has blocker#1 (job→MLflow auth) + blocker#2 (list_models/SDG provider file). Async (blocker#3) is REVERTED here. |
| `amortized-studio-gateway` (`gateway` container) | `studio-gateway@sha256:5612b9b3` | serves the embed studio (baked, `VITE_BASE_PATH=/amortized-studio-embed`) + async-capable client (backward-compat with the sync server). |
| `amortized-studio-static` | `ghcr.io/amortized-ai/studio:latest` (reverted) | NOT used by the embed (the gateway bakes the studio); vestigial. |
| Morty sandbox `morty-esivaram` (ns openshell) | Ready (recreated via openshell) | MCP tools verified working; ADC is EPHEMERAL (`/workspace/adc.json`, lost on pod restart). |

## What WORKS (verified this session)

- **Job pods → enterprise MLflow** (blocker #1): document/upload job logged download + all
  artifacts to the `amz-esivaram` MLflow workspace; job succeeded. Mechanism: worker ships a
  `sitecustomize.py` RequestHeaderProvider (bearer + `X-MLFLOW-WORKSPACE`) + service-CA trust.
- **Teacher-model discovery** (blocker #2): `list_models` returns `gpt-4.1`, `gpt-5` (provider
  `openai`) via the direct-provider fallback (dropped-in `OPENAI_API_KEY`).
- **SDG generation itself** works when the provider is right: earlier a hand-built SDG job
  (provider `openai`, model `gpt-4.1`) generated 3 records and logged the dataset to MLflow.
- **Morty chat + MCP tools**: chat replies via Vertex; `list_models` tool call succeeds
  (`POST /mcp 200` + `GET /api/v1/models 200`). The studio embed renders (base-path fix).
- **Server-side MLflow I/O**: datasets/runs are created + visible via `/api/v1/datasets`.

## The 4 GAPS (with root cause + fix direction)

### 1. SDG job from chat fails: `No provider named 'gateway' registered`
- **Root cause (confirmed):** Morty's **baked SDG skills in the `morty:aipcc` image**
  (`/workspace/.opencode/agents/sdg.md`, `/workspace/skills/sdg/*/guide.md`) hardcode
  `model_configs: [{... "provider": "gateway" ...}]` (the old MLflow-AI-Gateway convention).
  Morty fills `model` from `list_models` (e.g. `gpt-5`) but keeps `provider: "gateway"`.
  The SDG job writes a `model_providers.yaml` that only defines the ENABLED providers
  (`openai`), so `gateway` isn't registered → data-designer errors.
  NOTE: the amortized repo's `agent/skills/.../reference-payload.json` (which was updated to
  `<from-list_models>`) is a DIFFERENT copy than the morty image's baked `/workspace/skills/` —
  fixing the repo copy did NOT change what Morty runs.
- **Fix options:**
  - (a) **Rebuild the `morty:aipcc` image** with SDG skills/agent prompt that set
    `provider` from `list_models` (i.e. `openai`), not `gateway`. Then recreate sandboxes.
    Cleanest long-term (skills should never hardcode a provider).
  - (b) **Server shim in `jobs/sdg.py`**: also emit a `gateway` provider entry aliased to the
    primary enabled provider (openai endpoint + `OPENAI_API_KEY`). Unblocks without a morty
    rebuild, but conflates names. Requires a server redeploy (→ see meta-issue MCP).

### 2. "View in MLflow" opens a blank UI (`/mlflow/#/`, no experiments/runs)
- **What's true:** server-side MLflow config is correct (`AMORTIZED_MLFLOW_TRACKING_URI`,
  `AMORTIZED_MLFLOW_WORKSPACE=amz-esivaram`), and server-side MLflow I/O works (runs/datasets
  are created + listed via `/api/v1/datasets`). So MLflow connectivity is NOT the problem.
- **Likely cause (needs diagnosis):** the gateway's `/mlflow` **UI proxy** isn't delivering
  workspace-scoped data to the MLflow UI SPA. The enterprise MLflow requires the
  `X-MLFLOW-WORKSPACE` header on every request (incl. the UI's `ajax-api` calls); if the
  gateway isn't injecting it for those, MLflow returns "workspace required" and the UI shows
  nothing. Earlier the memory recorded `/mlflow/api/.../experiments/search` returning 200 via
  the gateway — so verify whether that regressed on the current gateway build (`5612b9b3`) or
  whether it's only the UI's ajax-api path that's missing the header.
- **Where to look:** `studio-gateway/src/server.js` `/mlflow/**` proxy (auth + workspace
  injection + `MLFLOW_CA_FILE`).

### 3. Training almost certainly untested / blocked
- **MLflow logging** should work: TRL `report_to: mlflow` uses the same vanilla-mlflow client
  path that blocker#1's `sitecustomize` fixes, and that's present in the deployed `d0b2311`.
- **But** training is blocked upstream: it needs a **dataset** (SDG must succeed first — gap #1)
  and **GPU** capacity, and the SDG→training chaining (`mlflow artifacts download` of the parent
  run) uses the same job-pod auth (should work). Untested end-to-end.

### 4. Embedded chat 504 on long turns
- **Cause:** the async-agent fix (blocker #3) is REVERTED at the deploy level (server is the
  synchronous `d0b2311`), so a delegating turn (e.g. Morty building the SDG config) exceeds the
  dashboard's internal proxy timeout → 504 in the iframe. (Short turns are fine.)
- **Fix:** redeploy the async server (`4c38c59`, code committed) — the async studio client is
  already live in the gateway. BUT this restarts the server → breaks Morty's MCP (meta-issue).

## META-ISSUES (systemic — fix these to make the above safe/repeatable)

- **MCP sessions don't survive server redeploys.** MCP is `fastapi-mcp` `mount_http` (stateful).
  Any amortized-server restart invalidates opencode's MCP session → Morty's tools fail
  ("session not recognized"), and opencode does NOT auto-reconnect. **Investigate `fastapi-mcp`
  stateless mode** (or an opencode MCP auto-reconnect). This is the highest-leverage fix — it
  removes the "every redeploy breaks Morty" trap that caused most of this session's pain.
- **Morty sandbox lifecycle is manual + fragile** (see packaging §D 4b):
  - ADC is on ephemeral `/workspace/adc.json` → lost on pod restart. Make it a mounted Secret.
  - Must be managed THROUGH openshell (`sandbox create/stop/start/delete`), NEVER `oc delete
    pod`/`kill` — out-of-band pod ops leave openshell's Sandbox in `Error` (unroutable, 412),
    unrecoverable except delete+create.
  - `service expose` goes stale on pod recreation; needs re-expose.
  - **Must be automated in the gateway provisioner** (templated Sandbox CR + secret ADC + egress
    + expose) so a new studio user gets Morty reproducibly.
- **Studio is baked into the `studio-gateway` image** (not `studio-static`); update the frontend
  by rebuilding `studio-gateway` from the repo root. The deployed gateway `src/` was uncommitted
  (now committed — see below).
- **Standalone Studio route shows a white page at `/`** because the studio is built with base
  `/amortized-studio-embed`; it only renders under `.../amortized-studio-embed/`. Add a gateway
  redirect `/ → /amortized-studio-embed/`.

## Where the code lives

- **`rhoai-integration` branch** (in worktree `~/workspace/amortized-pr419`, off PR #419's
  `feat/rhoai-studio-plugin`) — the integration branch. NEW commits this session:
  1. `feat(gateway): fold per-user MLflow + OpenShell Morty into provisioning; shared tier`
     (the deployed gateway `src/` that previously existed ONLY in the pod, + `rhoai-hybrid` overlay).
  2. cherry-pick of the app work (below).
- **`rhoai-showcase-e2e` branch** (main worktree `~/workspace/amortized`, off `main`) — safety
  copy of the app work: `feat: rhoai showcase e2e — job MLflow auth, direct model providers,
  async agent` (worker.py + `_mlflow_job_sitecustomize.py` + `api/models.py` + `core/model_catalog.py`
  + `jobs/sdg.py` + async `api/agent.py` + async `studio/src/lib/api-client.ts` + `k8s/overlays/rhoai` + docs).
- **Backups on disk:** `~/workspace/_amortized_gateway_deployed_backup/` (deployed gateway src),
  `~/workspace/_amortized_new_files_backup/`, `/tmp/amz-tracked-*.patch`.
- **This doc + `docs/rhoai-showcase-packaging.md` §D** = the packaging kinks. NOTE: docs edited
  on `rhoai-showcase-e2e`; may need syncing to `rhoai-integration`.

## Recommended fix order for the fresh session

1. **MCP statelessness** (`fastapi-mcp`) — so server redeploys stop breaking Morty. Unblocks
   iterating on everything else.
2. **SDG provider** (gap #1) — fix the morty image's SDG skills to use the `list_models`
   provider (or the sdg.py shim). Then a real chat→SDG→MLflow run should complete.
3. **Automate the Morty sandbox provisioning** (packaging §D 4b) — templated CR + secret ADC +
   egress + expose in the gateway provisioner; also fold per-user MLflow RBAC + OPENAI key.
4. **`/mlflow` UI proxy** (gap #2) — workspace-header injection for the UI's ajax-api calls.
5. **Redeploy async server** (gap #4) once MCP is stateless — kills the embed 504.
6. **Training** (gap #3) — verify once SDG produces a dataset + GPU is available.

## Exact recovery recipes (validated this session)

- **Recreate a Morty sandbox** (openshell CLI; needs `oc -n openshell port-forward svc/openshell
  8080:8080` + `~/.config/openshell/gateways/cluster/mtls`):
  `openshell -g cluster sandbox create --name morty-<user> --from <morty-aipcc-digest>
  --policy <policy.yaml with amortized-server.amz-<user>.svc:8000 egress>
  --auto-providers --provider google-vertex-ai --env GOOGLE_CLOUD_PROJECT=lightwell-devel
  --env VERTEX_LOCATION=global --env USER_NS=amz-<user> --env GOOGLE_APPLICATION_CREDENTIALS=/workspace/adc.json
  --detach -- sh -c 'cd /workspace && sed -i "s|amortized-server.amortized.svc|amortized-server.$USER_NS.svc|" opencode.json && HOME=/workspace opencode serve --port 4096 --hostname 0.0.0.0'`
  then `openshell -g cluster sandbox upload morty-<user> <ADC> /workspace/adc.json`
  then `openshell -g cluster service expose morty-<user> 4096 opencode`.
  (`--upload` cannot be combined with a `-- command`; upload separately. ADC must be world-readable.)
- morty:aipcc image digest: `image-registry.openshift-image-registry.svc:5000/openshell/morty@sha256:a357aa0ef75f091351220de5b6d2faa3f62d9cb333ce3cf7010c82e580f4c8c8`
