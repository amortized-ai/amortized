# RHOAI Showcase — Session Handoff (2026-08-31)

For a fresh session to fix the remaining pieces. Companion to
`docs/rhoai-showcase-packaging.md` (§D = packaging kinks) and memory
`amortized-showcase-proveout`. Cluster: **pawshift**, user backend ns **`amz-esivaram`**,
sandboxes in ns **`openshell`**, shared tier in ns **`amortized-showcase`**.

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

## KNOWN GAPS & CAVEATS — mind these for testing (do NOT block the first tester round)

1. **[SECURITY TODO] `/mlflow` UI proxy header hardening.** The gateway holds a
   *cluster-wide* MLflow SA and isolates each user only by injecting
   `X-MLFLOW-WORKSPACE=nsForUser(user)` from the SSO identity (`setHeader`, which
   overwrites). TODO: explicitly *strip* any client-supplied `X-MLFLOW-WORKSPACE` /
   `Authorization` on `/mlflow` before injecting (defense-in-depth). SSO-gated today so
   not exploitable, but close before wider exposure. (Also in packaging §D.)
2. **[meta#3, TOP packaging gap] Per-user Morty sandbox is manual + fragile.** ADC lives
   on ephemeral `/workspace/adc.json` (lost on pod restart); create/egress/expose are
   hand-done. Must become a templated Sandbox CR + mounted-Secret ADC in the gateway
   provisioner. Until then a Morty pod restart silently loses auth.
3. **Backend/naming mismatch.** Backends: `amz-esivaram`, `amz-xingliu`. Morty sandboxes:
   esivaram / meyceoz / xingyu. `amz-xingliu` ≠ `morty-xingyu` — confirm who's who.
   Mustafa (`meyceoz`) has a sandbox but NO backend yet (auto-provisions on first login,
   gets server `:showcase`=f933958).
4. **Testers need the new server image.** `amz-xingliu` runs the `:showcase` tag on a
   stale pod → restart it to pull `f933958`. New provisions get it automatically.
5. **[§D5] Stale browser cache after a gateway redeploy.** SPA `index.html` has no
   `Cache-Control` → hard-refresh (Cmd+Shift+R) after any gateway rebuild or you load an
   old/broken bundle. TODO: `Cache-Control: no-cache` on the SPA HTML.
6. **gap #3 — training UNTESTED.** Unblocked on data (an SDG dataset exists) but needs a
   GPU; SDG→training chaining + `report_to: mlflow` unverified e2e.
7. **`get_turn` MCP exclude** committed (`aab485a`) but not yet on the running server
   (cosmetic; deploys with the next server build).
8. **Direct-provider stopgap.** Teacher models come from a dropped-in `OPENAI_API_KEY`
   (no MLflow AI Gateway); a funded key is required for a real SDG run.
9. **Standalone Studio route** white-pages at `/` (built with base `/amortized-studio-embed`);
   needs a `/`→`/amortized-studio-embed/` redirect. The embed path itself is fine.
10. **mlflow-postgres durability.** The enterprise MLflow's backend postgres is `emptyDir`
    → needs a PVC before any durable showcase.

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
