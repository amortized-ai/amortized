# Step 3 — OpenShell + Morty on the cluster (artifacts + runbook)

Artifacts for putting the Morty persona onto ROSA/`pawshift` inside an OpenShell sandbox.

> **DEPLOYED STANDARD (2026-08-27): the aipcc variant.** Live sandbox `morty-aipcc` in ns `openshell`
> (image `morty:aipcc`), verified answering via Vertex. Use `Dockerfile.aipcc` + `policy.aipcc.yaml`.
> The `*.nvidia` files are the first-proven variant, kept for reference.

| File | Purpose |
|---|---|
| `Dockerfile.aipcc` **(standard)** | Persona image on RHOAI base `quay.io/aipcc/agentic-ci/openshell:0.3.27` (+ node + persona at `/workspace`). |
| `policy.aipcc.yaml` **(standard)** | Egress policy: Vertex + MCP host; `read_write` includes `/workspace`; opencode.exe at `/usr/local/lib/...`. |
| `opencode.cluster.json` | opencode config WITH the amortized MCP block (`/mcp` on `:8000`). |
| `Dockerfile.nvidia` / `policy.nvidia.yaml` | First-proven variant (NVIDIA community base, home `/sandbox`). Reference only. |

Versions (verified 2026-08-27): agent-sandbox **v0.5.6**, OpenShell chart **0.0.113**.

### Verified deployment (aipcc) — the actual working commands
```bash
# build in-cluster (amd64) — local podman builds arm64 -> Exec format error
oc create imagestream morty -n openshell 2>/dev/null; \
oc apply -n openshell -f - <<'BC'
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata: {name: morty-aipcc, namespace: openshell}
spec:
  output: {to: {kind: ImageStreamTag, name: 'morty:aipcc'}}
  source: {type: Binary}
  strategy: {type: Docker, dockerStrategy: {dockerfilePath: Dockerfile}}
BC
# context = persona files (make prompt) + Dockerfile.aipcc as Dockerfile + opencode.cluster.json as opencode.json
oc start-build morty-aipcc --from-dir=<ctx> --follow -n openshell
DIGEST=$(oc get istag morty:aipcc -n openshell -o jsonpath='{.image.dockerImageReference}')
# create sandbox pinned by DIGEST (tag caching serves stale arch); force cwd/HOME=/workspace
openshell -g cluster sandbox create --name morty-aipcc --detach --auto-providers \
  --provider google-vertex-ai --env GOOGLE_CLOUD_PROJECT=lightwell-devel \
  --env VERTEX_LOCATION=global --env GOOGLE_APPLICATION_CREDENTIALS=/workspace/adc.json \
  --policy policy.aipcc.yaml --from "$DIGEST" \
  -- sh -c 'cd /workspace && HOME=/workspace opencode serve --port 4096 --hostname 0.0.0.0'
openshell -g cluster sandbox upload morty-aipcc ~/.config/gcloud/application_default_credentials.json /workspace/adc.json
# verify (no ps in image; find PID via /proc, nsenter into opencode netns):
#   PID=$(oc exec default--morty-aipcc -n openshell -c agent -- sh -c 'for p in /proc/[0-9]*; do c=$(tr "\0" " " <"$p/cmdline" 2>/dev/null); case "$c" in "opencode serve "*) basename "$p";; esac; done')
#   oc exec -i default--morty-aipcc -n openshell -c agent -- nsenter -t $PID -n sh  # then curl localhost:4096 (POST /session, message agent=morty)
```
CLI→gateway prereq: `oc port-forward svc/openshell 8080:8080` + gateway `cluster` registered with mTLS
certs from secret `openshell-client-tls` (see the `openshell-morty-pawshift` memory).

## Prereqs (Steps 1-2) — cluster-admin, get the team into `pawshift-cluster-admins`

Order matters: install agent-sandbox FIRST, then ns + SCC, then the gateway.
```bash
# Step 1 — Agent Sandbox controller + CRDs (cluster-scoped). NOTE: the plan's old
# manifest.yaml URL now 404s; from v0.5.4+ the asset is sandbox.yaml. Pin v0.5.6.
kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v0.5.6/sandbox.yaml
kubectl -n agent-sandbox-system rollout status deploy/agent-sandbox-controller
# (the controller runs fine under restricted-v2 — no SCC grant needed for it)

# Step 2 — namespace + privileged SCC for the OpenShell sandbox SA
oc create ns openshell
oc adm policy add-scc-to-user privileged -z openshell-sandbox -n openshell
```

## 3b — Build + push the Morty image

```bash
# In ~/workspace/amortized on the Mac: assemble persona files (proven make target)
make prompt
mkdir -p /tmp/morty-build-ctx/skills
cp k8s/base/morty-prompt.md            /tmp/morty-build-ctx/morty.md
cp k8s/base/morty-sdg-workflow.md      /tmp/morty-build-ctx/sdg.md
cp k8s/base/morty-training-workflow.md /tmp/morty-build-ctx/training.md
cp -R k8s/base/morty-skills/sdg k8s/base/morty-skills/training /tmp/morty-build-ctx/skills/
cp docs/openshell-step3/Dockerfile docs/openshell-step3/.dockerignore /tmp/morty-build-ctx/
cp docs/openshell-step3/opencode.cluster.json /tmp/morty-build-ctx/opencode.json
docker build -t morty:step3 /tmp/morty-build-ctx      # or build on the VM (docker present there)
```
Push to a registry the cluster can pull:
```bash
# A. OpenShift internal registry (self-contained; route confirmed live):
REG=default-route-openshift-image-registry.apps.rosa.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com
oc registry login --registry="$REG"
docker tag morty:step3 "$REG/openshell/morty:step3"
docker push "$REG/openshell/morty:step3"
# In-cluster pull path: image-registry.openshift-image-registry.svc:5000/openshell/morty:step3
# B. or ghcr.io/amortized-ai/morty:step3 (cluster already pulls that org).
```

## 3d — Provider + credentials (ADC)

ADC is a secret — create it from the local file, never commit it (see `morty-adc-rotation` memory):
```bash
oc create secret generic opencode-gcp -n openshell \
  --from-file=adc.json="$HOME/.config/gcloud/application_default_credentials.json"
```
Sandbox needs `GOOGLE_APPLICATION_CREDENTIALS=/sandbox/adc.json`, `GOOGLE_CLOUD_PROJECT=lightwell-devel`,
`VERTEX_LOCATION=global`.

## 3a — Deploy the OpenShell gateway (Helm, kubernetes driver) — VERIFIED values

```bash
helm install openshell oci://ghcr.io/nvidia/openshell/helm-chart \
  --version 0.0.113 --namespace openshell \
  --set server.disableTls=true \
  --set podSecurityContext.fsGroup=null \
  --set securityContext.runAsUser=null
oc -n openshell rollout status statefulset/openshell
```
Why: `disableTls` = plaintext for eval; `fsGroup=null`/`runAsUser=null` clear the chart's hardcoded
UID so OpenShift SCC admission assigns them. The chart creates the sandbox SA `openshell-sandbox`
(so grant the SCC to it, as in Step 2) and, by default, `sandboxNamespace` = the release ns
(`openshell`). `networkPolicy.enabled=true` (default) restricts sandbox ingress to :2222 from the
gateway — relevant to 3f. Optional external access: `--set openshiftRoute.enabled=true` (TLS
passthrough; needs `disableTls=false` + a server cert whose SANs include the Route host).

## 3c / 3e — Create the Morty sandbox + verify

Use the OpenShell CLI against the on-cluster gateway (keeps the supervisor + egress policy — the
"opencode inside OpenShell" model, same command shape proven on the VM). Point the CLI at the
in-cluster gateway (`--gateway-endpoint`, or `oc port-forward svc/openshell -n openshell`, or the
Route from `openshiftRoute.enabled`):
```bash
openshell provider create --type google-vertex-ai --name google-vertex-ai --from-gcloud-adc
openshell sandbox create --name morty --detach --provider google-vertex-ai \
  --env GOOGLE_CLOUD_PROJECT=lightwell-devel --env VERTEX_LOCATION=global \
  --env GOOGLE_APPLICATION_CREDENTIALS=/sandbox/adc.json \
  --policy docs/openshell-step3/policy.cluster.yaml \
  --from image-registry.openshift-image-registry.svc:5000/openshell/morty:step3 \
  -- opencode serve --port 4096 --hostname 0.0.0.0
openshell sandbox upload morty "$HOME/.config/gcloud/application_default_credentials.json" /sandbox/adc.json
```
Notes: `--policy` is NOT in the OpenShell docs but is VERIFIED working on the VM (v0.0.113) — the
sandbox came up with `morty_egress` enforced. This creates an `agents.x-k8s.io/v1beta1` `Sandbox` CR
(everything under `spec.podTemplate`); the OpenShell-managed CR does NOT set `spec.service`, so no
Service is auto-created — see 3f. Verify by reaching `:4096` (below), `POST /session`,
`POST /session/{id}/message {"agent":"morty",...}`, and confirm Morty replies in-character AND can
call an amortized MCP tool.

> Alternative (NOT recommended here): a hand-authored `Sandbox` CR with `spec.service: true` gets you
> a headless Service (`status.serviceFQDN`), but it runs the pod WITHOUT the OpenShell supervisor/policy
> — that's "agent-sandbox as a plain orchestrator", not sandboxed-Morty. Only use if you drop OpenShell.

## 3f — Wire into amortized (server → sandbox) — THE key decision

`amortized-server` reaches opencode via `AMORTIZED_AGENT_UPSTREAM_URL` (default `http://opencode:4096`;
currently unset). OpenShell isolates sandbox ingress (NetworkPolicy allows only :2222 from the gateway),
so `:4096` is not directly reachable by default. Two paths:

- **Approach 1 — `openshell service expose` (supported; keeps policy).**
  `openshell service expose morty 4096 opencode` → a gateway-managed URL. In-cluster that's an
  **authenticated HTTPS** URL through the gateway (`openshell.openshell.svc.cluster.local:8080`).
  Cost: the amortized agent proxy currently does plain HTTP — it would need to handle gateway auth
  (mTLS/token) to use this URL.

- **Approach 2 — direct Service + NetworkPolicy (simpler HTTP, UNVERIFIED).** Create a ClusterIP
  Service selecting the sandbox pod (`openshell.ai/sandbox-name: morty`, port 4096) + a NetworkPolicy
  allowing ingress from `amortized-server`, then set
  `AMORTIZED_AGENT_UPSTREAM_URL=http://<svc>.openshell.svc.cluster.local:4096`.
  RISK: OpenShell's in-pod supervisor nftables MAY drop unsolicited inbound to :4096 even after you
  open the k8s Service/NetworkPolicy (driver README: "sandbox pods do not need direct external ingress").
  **Test this first — it's the cheapest if it works; fall back to Approach 1 if blocked.**

  ```yaml
  apiVersion: v1
  kind: Service
  metadata: { name: morty-sandbox, namespace: openshell }
  spec:
    selector: { openshell.ai/sandbox-name: morty }   # verify: oc -n openshell get pod --show-labels
    ports: [{ name: opencode, port: 4096, targetPort: 4096 }]
  ---
  apiVersion: networking.k8s.io/v1
  kind: NetworkPolicy
  metadata: { name: allow-amortized-to-morty, namespace: openshell }
  spec:
    podSelector: { matchLabels: { openshell.ai/managed-by: openshell } }
    policyTypes: [Ingress]
    ingress:
      - from: [{ namespaceSelector: { matchLabels: { kubernetes.io/metadata.name: amortized } },
                podSelector: { matchLabels: { component: server } } }]   # adjust to real labels
        ports: [{ protocol: TCP, port: 4096 }]
  ```

opencode is started with `--hostname 0.0.0.0` so it binds a routable interface (works for either approach).

## Open questions (carried)

- **MCP egress**: `policy.cluster.yaml` allows the MCP host on plaintext `:8000` — UNVERIFIED through
  OpenShell's TLS-terminating egress proxy.
- **Sandbox ingress (3f)**: the Approach-2 nftables risk is the main unknown — test on-cluster.
- **K8s driver is experimental** (OpenShell flags the OpenShift path as such) — POC scope.
