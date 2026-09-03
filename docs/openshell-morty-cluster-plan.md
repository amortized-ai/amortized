# OpenShell + Morty on OpenShift: Deployment Plan (Handoff)

> **For the next session**: this is a self-contained handoff. Read "Current state" first to know what's already proven, then follow Steps 1-4. The reusable artifacts (Dockerfile, opencode.json, policy.yaml) are inline and are copy-paste ready. Commands marked **[USER-RUN]** must be run by the human (the assistant's Bash tool blocks `sudo`, and cluster login needs a token the human holds). Commands marked **[CLUSTER OWNER]** need cluster-admin sign-off.
>
> **Security note**: this doc contains internal infra addresses but NO secrets (no cluster token, no ADC contents). Do not paste the `oc login` token or ADC JSON into this file. Keep this doc out of any public commit.

Goal: deploy the amortized control plane on the ROSA/OpenShift cluster with **Morty (the opencode agent) running inside an NVIDIA OpenShell sandbox** — the Red Hat-aligned, security-enhanced way to run agents. The full pattern was validated on a Linux VM (2026-08-26); this ports it to the cluster.

---

## Access & environment

- **Cluster (ROSA)**: `oc login --token=<HUMAN PROVIDES> --server=https://api.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com:443` — **[USER-RUN]** (token is a secret). RHOAI **3.4.3**; 5x NVIDIA **L40S** GPU nodes; shared cluster (~127 projects).
- **Linux VM (prototype host)**: `ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -J jump@169.62.18.122 esivaram@10.138.37.252`. For multi-line scripts append `'bash -s' <<'REMOTE' ... REMOTE`. CentOS Stream 10, root via passwordless sudo, docker 29.7.2 + podman 6.0.2.
- **GCP / Vertex**: project `lightwell-devel`, region `global`, model `claude-opus-4-8`. ADC lives at `~/.config/gcloud/application_default_credentials.json` on both the Mac and the VM (authorized_user; mints a Vertex token). `lightwell-devel` Vertex quota is low -> intermittent 429/404 throttling (expected, retry).
- **Repos**: app repo `~/workspace/amortized` (has `agents/` prompts, `agent/skills/`, `k8s/overlays/rosa`); deploy repo `~/workspace/amortized-deploy` (user overlays with `opencode.json`/model config, Vertex secret setup).
- **Assistant constraint**: the Bash tool's hook blocks any command containing `sudo`. All `sudo` steps must be **[USER-RUN]** (e.g. via the `!` prefix).

---

## Current state (done vs pending)

**DONE — proven on the VM (`10.138.37.252`)**:
- OpenShell installed; gateway healthy on the **docker** driver; systemd `--user` **linger enabled**.
- Morty image built (`morty-proto:local`), `google-vertex-ai` provider created from ADC, egress policy applied, ADC uploaded to the sandbox.
- **Verified**: the opencode agent replied via Vertex Claude Opus 4.8 from inside the OpenShell sandbox ("...I run inside an OpenShell sandbox, confirmed by my working directory `/sandbox`...").
- **(a) DONE (2026-08-27)** — Morty persona baked + verified. Rebuilt `morty-proto:local` with the REAL agent files assembled by `make prompt` (`morty.md` = orchestrator `identity.md`+`workflow.md` concatenated; `sdg.md`; `training.md`; skills tree), recreated the `morty` sandbox, and Morty replied IN-CHARACTER via Vertex Opus 4.8: *"I'm Morty, the Amortized Studio assistant — I help you replace expensive frontier model API calls with smaller, fine-tuned task models... My current working directory is `/sandbox`."* opencode selected the primary agent via `agent=morty` (matches the amortized proxy in `src/amortized/api/agent.py`).
- Caveat: the VM build context (`/tmp/morty-openshell/`, `/tmp/api_talk.py`) may be cleared on reboot — the artifacts below let you rebuild from scratch. There is NO repo on the VM (`~/workspace/amortized` does not exist), so the persona files must be assembled on the Mac (`make prompt`) and transferred.

**PENDING**:
- Steps 1-2: cluster-owner prerequisites (**CONFIRMED needed 2026-08-27** — as a `pawshift-dedicated-admins` member you lack `create CRD`, `create namespace`, and `use scc/privileged`; see Steps 1-2).
- Step 3: OpenShell + Morty on the cluster (kubernetes driver).
- Step 4: redeploy the current amortized stack (the existing one is ~55-67 days old and pre-PostgreSQL).

---

## Reusable artifacts (copy-paste ready)

### `Dockerfile` (glibc base + opencode + Morty persona) — VERIFIED 2026-08-27
Build context is a self-contained dir (NOT the repo). Stage it with `make prompt` output:
```bash
# on the Mac, in ~/workspace/amortized:
make prompt   # assembles k8s/base/morty-prompt.md (=identity+workflow) + morty-skills/ tree
mkdir -p /tmp/morty-build-ctx/skills
cp k8s/base/morty-prompt.md            /tmp/morty-build-ctx/morty.md
cp k8s/base/morty-sdg-workflow.md      /tmp/morty-build-ctx/sdg.md
cp k8s/base/morty-training-workflow.md /tmp/morty-build-ctx/training.md
cp -R k8s/base/morty-skills/sdg k8s/base/morty-skills/training /tmp/morty-build-ctx/skills/
# add Dockerfile + opencode.json + policy.yaml (below), then transfer + build on the VM.
# NOTE: tar with COPYFILE_DISABLE=1 to avoid baking macOS ._* files into /sandbox/skills.
```
```dockerfile
FROM ghcr.io/nvidia/openshell-community/sandboxes/base:latest
USER root
RUN npm install -g opencode-ai
# Morty persona. Baked under /sandbox (SURVIVES sandbox create — verified). cwd for
# `opencode serve` is /sandbox, so opencode discovers agents from /sandbox/.opencode/agents
# and reads skills at skills/ (mirrors the k8s deploy: workdir /app/workspace).
COPY morty.md    /sandbox/.opencode/agents/morty.md
COPY sdg.md      /sandbox/.opencode/agents/sdg.md
COPY training.md /sandbox/.opencode/agents/training.md
COPY skills/     /sandbox/skills/
COPY opencode.json /sandbox/opencode.json
RUN chown -R sandbox:sandbox /sandbox/.opencode /sandbox/skills /sandbox/opencode.json
USER sandbox
```
Rationale / fixes vs the old draft:
- `morty.md` MUST be `identity.md`+`workflow.md` **concatenated** (that's what `make prompt` does). The old draft copied only `identity.md` and made a bogus separate `morty-workflow` agent.
- Subagents come from `agents/sdg/workflow.md` and `agents/training/workflow.md`; skills from `agents/<agent>/skills/` (plural) — NOT `agent/skills/` (that dir has one stray file).
- Put `opencode.json` at the workdir root (`/sandbox/opencode.json`) and drop `OPENCODE_CONFIG`/`/opt/morty` — this mirrors the proven k8s mount (`/app/workspace/opencode.json`) so opencode auto-loads config AND discovers `.opencode/agents`.
- opencode's official image is Alpine/musl, incompatible with OpenShell's glibc supervisor; build on the community glibc base. `npm i -g` needs root (base runs as `sandbox`).

### `opencode.json` (VERIFIED with the Morty persona; add `mcp` when wiring into amortized)
```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "google-vertex-anthropic/claude-opus-4-8@default",
  "agent": { "build": { "disable": true }, "plan": { "disable": true } }
}
```
Disabling `build`/`plan` is SAFE here because `morty.md` declares `mode: primary` (from `identity.md` frontmatter) — there is still a primary visible agent, so no `no primary visible agent found`. This matches the k8s configmap (`k8s/base/opencode-configmap.yaml`). When integrating with amortized, add:
```json
  "mcp": { "amortized": { "type": "remote", "url": "http://amortized-server.<ns>.svc.cluster.local:8000/mcp", "enabled": true } }
```
Do NOT disable all agents (`"agent": {"build":{"disable":true},...}`) without providing a primary agent -> causes `no primary visible agent found`.

### `policy.yaml` (deny-all by default; allowlist hosts AND resolved binaries)
```yaml
version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /lib64, /bin, /sbin, /proc, /dev/urandom, /app, /etc, /opt, /var/log]
  read_write: [/sandbox, /tmp, /dev/null]
landlock:
  compatibility: best_effort
network_policies:
  morty_egress:
    name: morty-egress
    endpoints:
      - {host: aiplatform.googleapis.com, port: 443, protocol: rest, enforcement: enforce, access: read-write}
      - {host: oauth2.googleapis.com, port: 443, protocol: rest, enforcement: enforce, access: read-write}
      - {host: accounts.google.com, port: 443, protocol: rest, enforcement: enforce, access: read-write}
      - {host: sts.googleapis.com, port: 443, protocol: rest, enforcement: enforce, access: read-write}
      - {host: www.googleapis.com, port: 443, protocol: rest, enforcement: enforce, access: read-write}
      - {host: iam.googleapis.com, port: 443, protocol: rest, enforcement: enforce, access: read-write}
      - {host: models.opencode.ai, port: 443, protocol: rest, enforcement: enforce, access: read-only}
      - {host: models.dev, port: 443, protocol: rest, enforcement: enforce, access: read-only}
      - {host: registry.npmjs.org, port: 443, protocol: rest, enforcement: enforce, access: read-only}
      # cluster only: add the amortized MCP host so Morty can call its tools
      # - {host: amortized-server.<ns>.svc.cluster.local, port: 8000, protocol: rest, enforcement: enforce, access: read-write}
    binaries:
      - {path: /usr/lib/node_modules/opencode-ai/bin/opencode.exe}
      - {path: /usr/bin/node}
      - {path: /usr/bin/curl}
```
Notes: egress is deny-all AND enforced **per-binary** (omitting `binaries` = deny-all). opencode's real binary is `opencode.exe` (npm symlinks `/usr/bin/opencode` -> it); the policy must list the resolved path. opencode fetches its model catalog from `models.opencode.ai` (not models.dev). The `python3.14` binary was only needed for the standalone Vertex probe, not for Morty.

---

## Appendix A — Rebuild/verify the VM prototype from scratch (if /tmp was wiped)

On the VM (`10.138.37.252`). One-time gateway setup if not already active:
```bash
# driver = docker (docker daemon is root-backed on this VM)
printf 'OPENSHELL_DRIVERS=docker\n' > ~/.config/openshell/gateway.env
```
```bash
# [USER-RUN] persistence + start (contains sudo):
sudo loginctl enable-linger esivaram && systemctl --user restart openshell-gateway && sleep 5 && openshell status
```
Build + provider + sandbox + verify:
```bash
mkdir -p /tmp/morty-openshell && cd /tmp/morty-openshell
# write Dockerfile, opencode.json, policy.yaml from the artifacts above
docker build -t morty-proto:local /tmp/morty-openshell
openshell provider create --type google-vertex-ai --name google-vertex-ai --from-gcloud-adc
openshell sandbox delete morty 2>/dev/null
openshell sandbox create --name morty --detach --provider google-vertex-ai --auto-providers \
  --env GOOGLE_CLOUD_PROJECT=lightwell-devel --env VERTEX_LOCATION=global \
  --env GOOGLE_APPLICATION_CREDENTIALS=/sandbox/adc.json \
  --policy /tmp/morty-openshell/policy.yaml \
  --from morty-proto:local -- opencode serve --port 4096 --hostname 0.0.0.0
openshell sandbox upload morty ~/.config/gcloud/application_default_credentials.json /sandbox/adc.json
```
Verify (talk): drive the serve API via a small python script hitting `http://localhost:4096` (POST `/session`, POST `/session/{id}/message` with `{"parts":[{"type":"text","text":"..."}]}`, read assistant text parts). A Claude reply = success.

---

## (a) Make it Morty — DONE + VERIFIED (2026-08-27)

Recreate + verify recipe (agent-file layout `.opencode/agents/{morty,sdg,training}.md` confirmed):
```bash
# on the VM (10.138.37.252), after building morty-proto:local from the corrected Dockerfile:
openshell sandbox delete morty
openshell sandbox create --name morty --detach --provider google-vertex-ai \
  --env GOOGLE_CLOUD_PROJECT=lightwell-devel --env VERTEX_LOCATION=global \
  --env GOOGLE_APPLICATION_CREDENTIALS=/sandbox/adc.json \
  --policy /tmp/morty-openshell/policy.yaml \
  --from morty-proto:local -- opencode serve --port 4096 --hostname 0.0.0.0
openshell sandbox upload morty ~/.config/gcloud/application_default_credentials.json /sandbox/adc.json
```
**Verifying opencode:4096 is the tricky part (see new gotcha #8).** opencode runs in a SEPARATE
network namespace (`sandbox_ip 10.200.0.2`), and `openshell sandbox exec`'s landlock blocks the
port-4096 connect. Reach it like an external client would, from the container's root netns:
```bash
CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep morty | awk '{print $1}')
docker exec "$CID" curl -s http://10.200.0.2:4096/api/health      # {"healthy":true}
# full probe: POST /session ; POST /session/$SID/message {"agent":"morty","parts":[{"type":"text","text":"..."}]} ; GET /session/$SID/message
```
Verified reply (agent=morty): *"I'm Morty, the Amortized Studio assistant — I help you replace
expensive frontier model API calls with smaller, fine-tuned task models... My current working
directory is `/sandbox`."*

---

## Step 1 — [CLUSTER OWNER] Install agent-sandbox CRDs + controller

OpenShell's kubernetes driver (used on-cluster) needs the upstream Kubernetes SIG CRDs ([kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox)). (Not needed on the VM, which used the docker driver.)

> **Access status (2026-08-27):** `esivaram@redhat.com` CANNOT do this — `oc auth can-i create customresourcedefinitions` = **no**, `create namespaces` = **no** (as a `pawshift-dedicated-admins` member). Needs cluster-admin. The clean fix is to have the team added to the **`pawshift-cluster-admins`** group (bound to ClusterRole `cluster-admin` via `pawshift-cluster-admins-binding`); current members who can grant it: `aaye@redhat.com`, `osilkin@redhat.com`, `yizheng@redhat.com`. Note the `pawshift-group-sync-binding` — group membership may be IdP/OCM-synced, so it may need to be changed at that source.
```bash
# CORRECTED 2026-08-27: the old .../releases/latest/download/manifest.yaml URL now 404s.
# From v0.5.4+ the release asset is sandbox.yaml (core) / sandbox-with-extensions.yaml. Pin v0.5.6.
kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v0.5.6/sandbox.yaml
kubectl -n agent-sandbox-system rollout status deploy/agent-sandbox-controller
```
Adds (cluster-scoped): `Sandbox` CRD (`agents.x-k8s.io`, v1beta1+v1alpha1), a controller Deployment in `agent-sandbox-system` (runs fine under `restricted-v2` — **no SCC grant needed for the controller**), and RBAC. Additive.
Verify: `oc get crd | grep agents.x-k8s.io && oc get deploy -n agent-sandbox-system`.

## Step 2 — [CLUSTER OWNER][USER-RUN] Grant privileged SCC to the openshell-sandbox SA
```bash
oc create ns openshell
oc adm policy add-scc-to-user privileged -z openshell-sandbox -n openshell
```
Why (validated on the VM): OpenShell's supervisor needs privileged/rootful to set up the sandbox; rootless failed on the Mac, worked with root-backed docker on the VM. On OpenShift the `privileged` SCC is the equivalent. Scoped to one SA in `openshell`.

> **Access status (2026-08-27):** `esivaram@redhat.com` CANNOT do this — `oc auth can-i use scc/privileged` = **no**, `update scc/privileged` = **no** (RBAC escalation prevention: you can't grant an SCC you don't hold). Same fix as Step 1: cluster-admin via the `pawshift-cluster-admins` group.

## Step 3 — Sandbox opencode-Morty with OpenShell (on-cluster)

**Full ready-to-run artifacts + runbook are in [`docs/openshell-step3/`](openshell-step3/) (staged 2026-08-27).**

> **DONE on cluster (2026-08-27):** Steps 1, 2, 3a-3e complete and VERIFIED. Live sandbox **`morty-aipcc`** in ns `openshell` (image `morty:aipcc`, on the RHOAI base `quay.io/aipcc/agentic-ci/openshell:0.3.27`) replied in-character via Vertex Opus 4.8. (The first-proven NVIDIA-base `morty`/`morty:step3` was removed after we standardized on aipcc.) Full wiring + gotchas captured in the `openshell-morty-pawshift` memory. Key deltas vs the runbook: build the image **in-cluster** (BuildConfig, amd64) and pin the sandbox `--from` by **digest** (Mac podman builds arm64 → `Exec format error`; nodes cache the tag). Gateway installed with `server.auth.allowUnauthenticatedUsers=true` (RHOAI-documented). Reach opencode via `nsenter` into its sub-netns; external via `openshell service expose`. **Step 3f (amortized→Morty) remains** — needs the current amortized server (Step 4) + `AMORTIZED_AGENT_UPSTREAM_URL` → the service-expose URL (Host header + TLS).

3a. **Deploy the gateway (Helm, kubernetes driver)** — pin **0.0.113** (matches the VM CLI):
```bash
helm install openshell oci://ghcr.io/nvidia/openshell/helm-chart --version 0.0.113 -n openshell \
  --set server.disableTls=true --set podSecurityContext.fsGroup=null --set securityContext.runAsUser=null
# optional external access: --set openshiftRoute.enabled=true (TLS passthrough; needs disableTls=false)
```
Chart creates the sandbox SA `openshell-sandbox` (grant it the SCC per Step 2) and defaults sandbox pods to the release ns. `fsGroup=null`/`runAsUser=null` clear the chart's hardcoded UID so OpenShift SCC admission assigns them.
3b. **Morty image**: build the persona image from (a); push to a registry the cluster can pull (or build in-cluster). Add the `mcp` block to `opencode.json` pointing at the in-cluster amortized server.
3c. **Egress policy**: same `policy.yaml`, plus the amortized MCP host (uncomment that line, set `<ns>`).
3d. **Provider + creds**: create the `google-vertex-ai` provider from ADC; deliver the ADC as a K8s Secret + set `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, `VERTEX_LOCATION`.
3e. **Create + verify**: create the Morty sandbox (via the on-cluster gateway / `Sandbox` CR); confirm Morty replies AND can call an amortized MCP tool.
3f. **Wire into amortized (integration)**: the amortized server `/agent/*` proxy reaches opencode via `AMORTIZED_AGENT_UPSTREAM_URL` (default `http://opencode:4096`). OpenShell's NetworkPolicy isolates sandbox ingress (only :2222 from the gateway), so `:4096` isn't directly reachable by default. **Two paths (details + YAML in `docs/openshell-step3/README.md`):** (1) `openshell service expose` — supported, keeps the policy model, but gives an authenticated in-cluster HTTPS URL (the proxy would need to handle gateway auth); (2) a direct ClusterIP Service (selector `openshell.ai/sandbox-name: morty`) + NetworkPolicy allowing the server — simpler plain HTTP, but there's an UNVERIFIED risk the supervisor's in-pod nftables drops inbound to :4096. **Recommendation: test (2) first (cheapest if it works); fall back to (1).**

## Step 4 — Redeploy the latest amortized (independent of 1-3)

4a. **[BEFORE TEARDOWN] Check PVC data** in the existing `amortized` namespace — do not delete blindly:
- `minio-data` (**200Gi** — S3 artifacts: datasets, model adapters) — the important one.
- `mlflow-data` (10Gi — MLflow metadata). `amortized-server-data` (10Gi — old SQLite, obsolete on PostgreSQL).
Decide preserve/migrate vs accept loss (demo data). Confirm with owner.

4b. **Tear down** the outdated stack (Deployments/StatefulSets, public Routes `amortized-studio` + `mlflow`); keep/delete PVCs per 4a.

4c. **Deploy current `main`** via `k8s/overlays/rosa` (PostgreSQL-based; migration landed 2026-08-05..07, #349). Set prod env: MLflow URI, S3/bucket, `AMORTIZED_DATABASE_URL`.

4d. **Verify**: Studio Route reachable; server healthy; a test SDG/training job dispatches to `amortized-jobs` and runs on an L40S node; Morty (Step 3) drives a job end-to-end.

---

## Gotchas (all learned on the VM — do not re-discover)

1. Rootless can't run OpenShell's supervisor -> needs privileged/rootful (the `privileged` SCC on-cluster).
2. opencode's official image is Alpine/musl -> incompatible with OpenShell's glibc supervisor. Use the community glibc base + `npm i -g opencode-ai`.
3. Egress is deny-all AND per-binary -> allowlist hosts + resolved binary paths (`opencode.exe`, not the `/usr/bin/opencode` symlink).
4. opencode's model catalog host is `models.opencode.ai` (not models.dev).
5. opencode config must keep a primary visible agent (`no primary visible agent found` otherwise).
6. systemd `--user` gateway needs `loginctl enable-linger` (VM only; on-cluster it's a Deployment).
7. Auth: ADC file + `GOOGLE_APPLICATION_CREDENTIALS` + `GOOGLE_CLOUD_PROJECT` + `VERTEX_LOCATION` (direct-ADC is the proven path). Upload the ADC after sandbox create (env points to it); on-cluster deliver as a Secret.
8. **Reaching opencode's `:4096` (docker driver).** opencode runs in a dedicated netns (`sandbox_ip 10.200.0.2`); `openshell sandbox exec` is IN that netns but its landlock blocks the connect to port 4096 (only 443 egress is allowed). To probe, connect from OUTSIDE the sandboxed process: `docker exec <cid> curl http://10.200.0.2:4096/...` (root netns → sandbox IP), or `docker exec <cid> nsenter -t <opencode_pid> -n curl http://localhost:4096/...`. This is fine — opencode `accept()` isn't policy-restricted; only in-sandbox egress is. On-cluster (k8s driver) this is Step 3f: expose `:4096` via a Service (ingress is unaffected by the egress policy).
9. **Baked `/sandbox` content SURVIVES `sandbox create`** (verified: a file baked at `/sandbox/.config/opencode/opencode.json` was present in the running sandbox). So the persona can be baked into the image rather than uploaded. The ADC is still uploaded post-create (secret, not baked). Also: there is NO repo on the VM — assemble persona files on the Mac and transfer.

## Open questions / risks

- **Server -> sandboxed-Morty reachability** (Step 3f): OpenShell restricts sandbox ingress; exposing Morty's `:4096` to the amortized server needs a Service + policy exception. Main unknown.
- **MCP egress**: Morty's policy must allow the in-cluster amortized MCP endpoint (added in 3c).
- **OpenShell K8s path is experimental** — POC/demo scope; expect rough edges.
- **Shared cluster**: Steps 1-2 are cluster-wide/privileged — require owner sign-off.
- **Vertex quota**: `lightwell-devel` throttles under load (429/404) — retry.
- **RHOAI 3.5 not required** — 3.4.3 is fine for OpenShell.

## Sequencing

Do (a) on the VM first (validate the Morty persona image). Steps 1-2 (cluster owner) gate Step 3, which reuses the VM image/policy. Step 4 is independent and can run in parallel.
