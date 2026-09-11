// On-demand per-user provisioning for the RHOAI hybrid deployment.
//
// Each dashboard user (X-Forwarded-User -> amz-<user>) gets an isolated stack:
//   1. the amortized core chart, pulled from OCI and installed per user via
//      `helm upgrade --install oci://... --version <pinned>` in enterprise-MLflow
//      mode (server + Postgres + MLflow RBAC/wiring + the sandboxed-Morty mTLS
//      upstream, injected through the chart's server.extra* hooks);
//   2. an OpenShell-sandboxed Morty, created through the one cluster OpenShell
//      gateway with the `openshell` CLI over mTLS, reachable by the server via
//      that mTLS wiring.
// The gateway also applies a few residual objects the chart cannot (the namespace,
// the openshell client-cert Secret, an optional teacher-keys Secret, a GPU quota).
//
// The OpenShell CLI usage mirrors the RHOAI opencode starter kit's documented flow
// (`gateway add`, `sandbox create`, `policy update --add-endpoint`, `service expose`).
// The Morty model provider is deployment-level (vertex ADC, or an openai/anthropic
// key auto-discovered from the gateway env); per-user creds is a follow-up.
//
// Idempotent + non-blocking: ensureUserStack kicks off provisioning and returns the
// current state immediately; callers poll getState / retry.

const os = require('os');
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');
const k8s = require('@kubernetes/client-node');
const { userValues, residualObjects, mortyName } = require('./manifests');

const execFileP = promisify(execFile);

const kc = new k8s.KubeConfig();
kc.loadFromCluster();
const objApi = k8s.KubernetesObjectApi.makeApiClient(kc);
const coreApi = kc.makeApiClient(k8s.CoreV1Api);

const FIELD_MANAGER = 'studio-gateway';

// --- Helm: the core chart is pulled from OCI; the version is pinned by the deployer ---
const HELM_BIN = process.env.HELM_BIN || 'helm';
const CHART_OCI = process.env.AMORTIZED_CHART_OCI || 'oci://ghcr.io/amortized-ai/charts/amortized';
const CHART_VERSION = process.env.AMORTIZED_CHART_VERSION || '';
const HELM_TIMEOUT = process.env.HELM_TIMEOUT || '5m';

// --- OpenShell CLI (baked in the gateway image; talks to the one cluster gateway) ---
const OPENSHELL_BIN = process.env.OPENSHELL_BIN || 'openshell';
const OPENSHELL_GATEWAY = process.env.OPENSHELL_GATEWAY || 'openshift';
const OPENSHELL_ENDPOINT = process.env.OPENSHELL_ENDPOINT || 'https://openshell.openshell.svc.cluster.local:8080';
const OPENSHELL_MTLS_DIR = process.env.OPENSHELL_MTLS_DIR || '/etc/openshell-mtls';
// The OpenShell gateway Service — its ClusterIP is resolved at runtime for the
// per-user server's hostAlias (so nothing hardcodes the IP).
const OPENSHELL_NAMESPACE = process.env.OPENSHELL_NAMESPACE || 'openshell';
const OPENSHELL_SERVICE = process.env.OPENSHELL_SERVICE || 'openshell';
const OPENSHELL_CONFIG_HOME =
  process.env.OPENSHELL_CONFIG_HOME || path.join(os.homedir() || '/tmp', '.config', 'openshell');
// Backstop timeout on openshell CLI calls so a hang (e.g. a mis-typed gateway that
// blocks on auth) fails fast + retryable instead of wedging provisioning forever.
const OPENSHELL_TIMEOUT_MS = parseInt(process.env.OPENSHELL_TIMEOUT_MS || '300000', 10);

// --- Morty sandbox ---
const MORTY_IMAGE = process.env.MORTY_IMAGE || 'ghcr.io/amortized-ai/morty:latest';
const BASE_POLICY_FILE = process.env.EGRESS_POLICY_FILE || path.join(__dirname, '..', 'policy.aipcc.yaml');

// Model provider (deployment-level; per-user creds is a follow-up). `vertex` uses a
// Vertex ADC (project/location + an uploaded adc.json); `openai`/`anthropic` use the
// corresponding API key, auto-discovered from the gateway env by `--auto-providers`
// and injected into the sandbox as runtime env (never written to the sandbox disk).
const MODEL_PROVIDER = (process.env.MORTY_MODEL_PROVIDER || 'vertex').toLowerCase();
const MORTY_MODEL = process.env.MORTY_MODEL || 'google-vertex-anthropic/claude-opus-4-8@default';
const OPENSHELL_PROVIDER_TYPE = { vertex: 'google-vertex-ai', openai: 'openai', anthropic: 'anthropic' }[MODEL_PROVIDER];
// Vertex-only.
const MORTY_ADC_FILE = process.env.MORTY_ADC_FILE || '/etc/morty-adc/adc.json';
const GOOGLE_CLOUD_PROJECT = process.env.GOOGLE_CLOUD_PROJECT || '';
const VERTEX_LOCATION = process.env.VERTEX_LOCATION || 'global';

// Morty automation runs only when the OpenShell mTLS certs are mounted. Otherwise
// the core stack is provisioned without chat.
const MORTY_ENABLED = fs.existsSync(path.join(OPENSHELL_MTLS_DIR, 'tls.crt'));

// namespace -> { state: 'provisioning'|'ready'|'error', promise, error }
const stacks = new Map();

function nsForUser(user) {
  const local = String(user).split('@')[0];
  const slug = local.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'anon';
  // `amz-` (not `amortized-u-`) so this integrated per-user stack does not collide
  // with pre-existing `amortized-u-*` backends. Keep in sync with manifests.js.
  return `amz-${slug}`;
}

async function run(bin, args, opts = {}) {
  console.log(`  $ ${bin} ${args.join(' ')}`);
  try {
    const { stdout, stderr } = await execFileP(bin, args, { maxBuffer: 16 * 1024 * 1024, ...opts });
    const tail = (stderr || '').trim();
    if (tail) console.log(`    ${tail.split('\n').slice(-3).join('\n    ')}`);
    return stdout;
  } catch (err) {
    const detail = (err.stderr || err.stdout || err.message || '').toString().trim();
    throw new Error(`${bin} ${args[0]} failed: ${detail.split('\n').slice(-5).join(' ')}`);
  }
}

async function applyObject(obj) {
  // Create-if-not-exists (idempotent). We provision once per user, so update
  // semantics aren't needed — and this sidesteps apply-patch content-type
  // differences across @kubernetes/client-node versions.
  try {
    await objApi.create(obj, undefined, undefined, FIELD_MANAGER);
    console.log(`  created ${obj.kind}/${obj.metadata.name} (${obj.metadata.namespace || 'cluster'})`);
  } catch (err) {
    const code = err?.code ?? err?.statusCode ?? err?.response?.statusCode;
    const reason = err?.body?.reason || parseReason(err);
    if (code === 409 || reason === 'AlreadyExists') {
      console.log(`  exists  ${obj.kind}/${obj.metadata.name}`);
      return;
    }
    throw err;
  }
}

function parseReason(err) {
  try {
    const body = typeof err?.body === 'string' ? JSON.parse(err.body) : err?.body;
    return body?.reason;
  } catch {
    return undefined;
  }
}

// Resolve the OpenShell gateway Service ClusterIP (cached). Used for the per-user
// server's hostAlias so the mTLS Morty URL (*.openshell.localhost) routes to the
// gateway — nothing hardcodes the IP.
let cachedGatewayIP = '';
async function openshellGatewayIP() {
  if (cachedGatewayIP) return cachedGatewayIP;
  const res = await coreApi.readNamespacedService({ name: OPENSHELL_SERVICE, namespace: OPENSHELL_NAMESPACE });
  const ip = (res?.spec || res?.body?.spec || {}).clusterIP;
  if (!ip || ip === 'None') {
    throw new Error(`could not resolve ClusterIP for service ${OPENSHELL_SERVICE}.${OPENSHELL_NAMESPACE}`);
  }
  cachedGatewayIP = ip;
  return ip;
}

async function serverAvailable(ns) {
  // True readiness: the server answers health (migrations complete + app up), not
  // just pod-ready. Avoids AppsV1Api return-shape differences across client versions.
  const url = `http://amortized-server.${ns}.svc.cluster.local:8000/api/v1/health`;
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 2500);
    const res = await fetch(url, { signal: ctrl.signal });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}

// `helm upgrade --install` the core chart (pulled from OCI) for this user.
// Values are written as JSON (a valid YAML subset) so no YAML serializer is needed.
async function helmInstall(ns, gatewayIP) {
  if (!CHART_VERSION) throw new Error('AMORTIZED_CHART_VERSION is not set — pin the core chart version');
  const valuesFile = path.join(os.tmpdir(), `values-${ns}.json`);
  fs.writeFileSync(valuesFile, JSON.stringify(userValues(ns, { mortyEnabled: MORTY_ENABLED, gatewayIP }), null, 2));
  try {
    await run(HELM_BIN, [
      'upgrade', '--install', 'amortized', CHART_OCI,
      '--version', CHART_VERSION, '-n', ns, '-f', valuesFile, '--timeout', HELM_TIMEOUT,
    ]);
  } finally {
    try { fs.unlinkSync(valuesFile); } catch { /* best-effort cleanup */ }
  }
}

// Register the cluster OpenShell gateway with the CLI (once per process): place the
// mounted mTLS certs where the CLI expects them, then `openshell gateway add` — the
// documented flow from the RHOAI opencode kit (Step 4).
let openshellConfigured = false;
async function configureOpenshell() {
  if (openshellConfigured) return;
  const mtlsDir = path.join(OPENSHELL_CONFIG_HOME, 'gateways', OPENSHELL_GATEWAY, 'mtls');
  fs.mkdirSync(mtlsDir, { recursive: true });
  for (const f of ['ca.crt', 'tls.crt', 'tls.key']) {
    fs.copyFileSync(path.join(OPENSHELL_MTLS_DIR, f), path.join(mtlsDir, f));
  }
  // Remove any stale/wrong-type registration, then add as a --local mTLS gateway.
  // --local is REQUIRED: without it an https endpoint is treated as a cloud gateway
  // and blocks on browser authentication, which never completes in a pod.
  await run(OPENSHELL_BIN, ['gateway', 'remove', OPENSHELL_GATEWAY], { timeout: OPENSHELL_TIMEOUT_MS }).catch(() => {});
  await run(OPENSHELL_BIN, ['gateway', 'add', OPENSHELL_ENDPOINT, '--name', OPENSHELL_GATEWAY, '--local'], { timeout: OPENSHELL_TIMEOUT_MS });
  openshellConfigured = true;
}

// Create + wire the per-user Morty sandbox through the cluster OpenShell gateway.
// Idempotent: re-provisioning tolerates an existing sandbox and just re-asserts the
// egress rule + gateway route (both must survive a sandbox pod recreation).
async function ensureSandbox(ns) {
  await configureOpenshell();
  if (!OPENSHELL_PROVIDER_TYPE) {
    throw new Error(`unsupported MORTY_MODEL_PROVIDER '${MODEL_PROVIDER}' (expected vertex|openai|anthropic)`);
  }
  const name = mortyName(ns);
  const g = ['-g', OPENSHELL_GATEWAY];

  // Rewrite the baked opencode.json in-sandbox: set the per-user MCP URL (amz-<user>)
  // and the provider's model. `node` (present in the image) does a robust JSON edit
  // rather than a brittle sed; USER_NS + MORTY_MODEL are passed as sandbox env.
  const rewrite =
    'const fs=require("fs"),f="opencode.json",c=JSON.parse(fs.readFileSync(f));' +
    'c.model=process.env.MORTY_MODEL;' +
    'c.mcp.amortized.url="http://amortized-server."+process.env.USER_NS+".svc.cluster.local:8000/mcp";' +
    'fs.writeFileSync(f,JSON.stringify(c,null,2))';
  const serveCmd = `cd /workspace && node -e '${rewrite}' && HOME=/workspace opencode serve --port 4096 --hostname 0.0.0.0`;

  const createArgs = [
    ...g, 'sandbox', 'create',
    '--name', name,
    '--from', MORTY_IMAGE,
    '--policy', BASE_POLICY_FILE,
    '--provider', OPENSHELL_PROVIDER_TYPE, '--auto-providers',
    '--env', `USER_NS=${ns}`,
    '--env', `MORTY_MODEL=${MORTY_MODEL}`,
  ];
  if (MODEL_PROVIDER === 'vertex') {
    createArgs.push(
      '--env', `GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}`,
      '--env', `VERTEX_LOCATION=${VERTEX_LOCATION}`,
      '--env', 'GOOGLE_APPLICATION_CREDENTIALS=/workspace/adc.json',
    );
  }
  // openai/anthropic: the API key is auto-discovered from the gateway env by
  // `--auto-providers` and injected as sandbox runtime env — no file to upload.
  createArgs.push('--', 'sh', '-c', serveCmd);

  let created = false;
  try {
    await run(OPENSHELL_BIN, createArgs, { timeout: OPENSHELL_TIMEOUT_MS });
    created = true;
  } catch (err) {
    if (!/exist|already/i.test(err.message)) throw err;
    console.log(`  openshell sandbox ${name} already exists`);
  }
  // Vertex ADC is a file (unlike API keys) — upload it after a fresh create.
  if (created && MODEL_PROVIDER === 'vertex') {
    await run(OPENSHELL_BIN, [...g, 'sandbox', 'upload', name, MORTY_ADC_FILE, '/workspace/adc.json']);
  }
  // Per-user MCP host egress (documented `policy update`, per the RHOAI opencode kit).
  // FQDN required — the short .svc form is rejected (403) and opencode drops MCP tools.
  await run(OPENSHELL_BIN, [
    ...g, 'policy', 'update', name,
    '--add-endpoint', `amortized-server.${ns}.svc.cluster.local:8000:read-write:rest:enforce`,
    '--wait',
  ]);
  // Expose opencode :4096 via the gateway (Host-routed mTLS -> mortyHost).
  await run(OPENSHELL_BIN, [...g, 'service', 'expose', name, '4096', 'opencode']);
}

async function provision(ns, user) {
  console.log(`provisioning ${ns} for ${user}`);
  // Resolve the OpenShell gateway IP first (needed in the chart values for the
  // server's mTLS hostAlias) when Morty is enabled.
  const gatewayIP = MORTY_ENABLED ? await openshellGatewayIP() : '';
  // 1. Residual objects the chart does not create (namespace first, then ns-scoped).
  for (const obj of residualObjects(ns, user)) await applyObject(obj);
  // 2. Core stack via Helm from OCI (enterprise MLflow; opencode/studio off).
  await helmInstall(ns, gatewayIP);
  // 3. Per-user OpenShell-sandboxed Morty. Best-effort: a sandbox failure leaves the
  //    core stack (server/SDG/MLflow) usable — chat is degraded and can be retried.
  if (MORTY_ENABLED) {
    try {
      await ensureSandbox(ns);
    } catch (err) {
      console.error(`  morty sandbox for ${ns} failed (chat unavailable, retryable): ${err.message}`);
    }
  } else {
    console.log('  morty automation disabled (OpenShell mTLS certs not mounted)');
  }
  // 4. Wait for the server to answer health (migrations run as the chart's init).
  const deadline = Date.now() + 5 * 60 * 1000;
  while (Date.now() < deadline) {
    if (await serverAvailable(ns)) return;
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error(`server in ${ns} did not become ready within timeout`);
}

/**
 * Ensure the user's stack exists. Non-blocking: kicks off provisioning and returns
 * the current state immediately. Callers poll getState / retry.
 */
function ensureUserStack(user) {
  const ns = nsForUser(user);
  let entry = stacks.get(ns);
  if (!entry) {
    entry = { state: 'provisioning', error: null };
    // Fast path: if the backend is already healthy (gateway restart / returning
    // user), mark ready without re-provisioning. Otherwise provision the stack.
    entry.promise = serverAvailable(ns)
      .then((healthy) => {
        if (healthy) { entry.state = 'ready'; return; }
        return provision(ns, user).then(() => { entry.state = 'ready'; });
      })
      .catch((err) => {
        entry.state = 'error';
        entry.error = String(err.message || err);
        console.error(`provisioning failed for ${ns}:`, err?.body || err?.message || err);
      });
    stacks.set(ns, entry);
  }
  return { ns, state: entry.state, error: entry.error };
}

function getState(user) {
  const ns = nsForUser(user);
  const entry = stacks.get(ns);
  return { ns, state: entry ? entry.state : 'unprovisioned', error: entry?.error || null };
}

function markForRetry(user) {
  stacks.delete(nsForUser(user));
}

module.exports = { ensureUserStack, getState, markForRetry, nsForUser };
