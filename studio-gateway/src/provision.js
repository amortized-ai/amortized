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
// (`gateway add`, `sandbox create`, `service expose`). The Morty model provider is
// per-user (bring-your-own-key): each user picks a provider (openai|anthropic) and
// supplies its API key via the Studio splash/settings; the key is stored as a
// per-user Secret and, at sandbox-create time, given to opencode via its env while
// the provider's API host is opened in the sandbox egress policy.
//
// Idempotent + non-blocking: ensureUserStack kicks off provisioning and returns the
// current state immediately; callers poll getState / retry.

const os = require('os');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFile } = require('child_process');
const { promisify } = require('util');
const k8s = require('@kubernetes/client-node');
const { userValues, residualObjects, mortyName, MODEL_KEY_SECRET } = require('./manifests');

const execFileP = promisify(execFile);

const kc = new k8s.KubeConfig();
kc.loadFromCluster();
const objApi = k8s.KubernetesObjectApi.makeApiClient(kc);
const coreApi = kc.makeApiClient(k8s.CoreV1Api);

const FIELD_MANAGER = 'studio-gateway';

// The gateway's own namespace (from the downward-API service-account file), used to
// store per-user model-key Secrets.
function readOwnNamespace() {
  try {
    return fs.readFileSync('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'utf8').trim() || 'default';
  } catch {
    return 'default';
  }
}

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

// Model provider is per-user (bring-your-own-key). A user picks one of these and
// supplies its API key (Studio splash/settings). The key is stored as a per-user
// Secret, then delivered to opencode two ways at sandbox-create time: the key goes
// into opencode's env (opencode reads <credentialKey> to authenticate the provider),
// and the provider's API host is added to the sandbox egress policy so the request
// is allowed out. OpenShell's proxy does not inject credentials in this deployment,
// so egress-allowlist + env-key is the delivery mechanism; the tight allowlist is
// what bounds the key's exposure (Morty can only reach the model API, the user's own
// server, and opencode's model catalog — no attacker-controllable host).
//   provider -> { credential env-var opencode reads, API host to allow, opencode model }
// Model ids are overridable per deployment and track opencode's models.dev naming.
const PROVIDERS = {
  openai: {
    credentialKey: 'OPENAI_API_KEY',
    apiHost: 'api.openai.com',
    model: process.env.MORTY_MODEL_OPENAI || 'openai/gpt-4o',
  },
  anthropic: {
    credentialKey: 'ANTHROPIC_API_KEY',
    apiHost: 'api.anthropic.com',
    model: process.env.MORTY_MODEL_ANTHROPIC || 'anthropic/claude-opus-4-8',
  },
};
const SUPPORTED_PROVIDERS = Object.keys(PROVIDERS);

// The gateway's own namespace: per-user model-key Secrets live here so a key
// survives stack/sandbox recreation and is readable before amz-<user> exists.
const GATEWAY_NAMESPACE = process.env.GATEWAY_NAMESPACE || readOwnNamespace();

// Morty automation runs only when the OpenShell mTLS certs are mounted. Otherwise
// the core stack is provisioned without chat.
const MORTY_ENABLED = fs.existsSync(path.join(OPENSHELL_MTLS_DIR, 'tls.crt'));

// namespace -> { state: 'provisioning'|'ready'|'error', promise, error }
const stacks = new Map();

function nsForUser(user) {
  const s = String(user);
  const local = s.split('@')[0];
  // Collision-resistant: append a short hash of the FULL identity (incl. domain) so
  // two users whose sanitized local-parts would collide never share a namespace
  // (and thus MLflow workspace): a@x.com vs a@y.com, foo.bar vs foo-bar, or two
  // long local-parts sharing a prefix. The readable slug is capped so the derived
  // Morty host label (default--morty-<slug>-<hash>--opencode) stays within the
  // 63-char DNS label limit.
  const slug = local.toLowerCase()
    .replace(/[^a-z0-9-]/g, '-')
    .replace(/^-+/, '')
    .slice(0, 28)
    .replace(/-+$/, '') || 'anon';
  const hash = crypto.createHash('sha256').update(s).digest('hex').slice(0, 8);
  // `amz-` (not `amortized-u-`) so this integrated per-user stack does not collide
  // with pre-existing `amortized-u-*` backends. Keep in sync with manifests.js.
  return `amz-${slug}-${hash}`;
}

// Mask secret-looking `NAME=value` args (e.g. `OPENAI_API_KEY=sk-...`) so keys
// never land in the gateway logs.
function redactArg(a) {
  const m = /^([A-Za-z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|KEY))=.+/.exec(a);
  return m ? `${m[1]}=***` : a;
}

async function run(bin, args, opts = {}) {
  console.log(`  $ ${bin} ${args.map(redactArg).join(' ')}`);
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

// --- Per-user model key (bring-your-own-key) ---------------------------------
function slugForNs(ns) { return ns.replace(/^amz-/, ''); }
function keySecretName(ns) { return `morty-key-${slugForNs(ns)}`; }

// Read the per-user model key Secret from the gateway namespace.
// Returns { provider, key } or null when unset / unsupported.
async function readUserKey(ns) {
  try {
    const res = await coreApi.readNamespacedSecret({ name: keySecretName(ns), namespace: GATEWAY_NAMESPACE });
    const data = res?.data || res?.body?.data || {};
    if (!data.provider || !data.key) return null;
    const provider = Buffer.from(data.provider, 'base64').toString('utf8');
    const key = Buffer.from(data.key, 'base64').toString('utf8');
    return PROVIDERS[provider] ? { provider, key, credentialKey: PROVIDERS[provider].credentialKey } : null;
  } catch (err) {
    const code = err?.code ?? err?.statusCode ?? err?.response?.statusCode;
    if (code === 404) return null;
    throw err;
  }
}

// Persist (create or replace) the per-user model key Secret.
async function writeUserKey(ns, provider, key) {
  const body = {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: {
      name: keySecretName(ns),
      namespace: GATEWAY_NAMESPACE,
      labels: { app: 'amortized', 'app.kubernetes.io/managed-by': FIELD_MANAGER, 'amortized.ai/user-ns': ns },
    },
    type: 'Opaque',
    data: {
      provider: Buffer.from(provider).toString('base64'),
      key: Buffer.from(key).toString('base64'),
    },
  };
  try {
    await coreApi.createNamespacedSecret({ namespace: GATEWAY_NAMESPACE, body });
  } catch (err) {
    const code = err?.code ?? err?.statusCode ?? err?.response?.statusCode;
    if (code !== 409) throw err;
    await coreApi.replaceNamespacedSecret({ name: keySecretName(ns), namespace: GATEWAY_NAMESPACE, body });
  }
}

// Stamp the per-user model key into the USER namespace as the chart's teacherKeys
// secret (provider env-var name -> key), so the SAME key reaches the server env
// (model_catalog -> list_models + SDG teacher), not just the Morty sandbox.
async function ensureServerKeySecret(ns, keyInfo) {
  const body = {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: {
      name: MODEL_KEY_SECRET,
      namespace: ns,
      labels: { app: 'amortized', 'app.kubernetes.io/managed-by': FIELD_MANAGER },
    },
    type: 'Opaque',
    data: { [keyInfo.credentialKey]: Buffer.from(keyInfo.key).toString('base64') },
  };
  try {
    await coreApi.createNamespacedSecret({ namespace: ns, body });
  } catch (err) {
    const code = err?.code ?? err?.statusCode ?? err?.response?.statusCode;
    if (code !== 409) throw err;
    await coreApi.replaceNamespacedSecret({ name: MODEL_KEY_SECRET, namespace: ns, body });
  }
}

// Restart the user's server so it re-reads the model key from its env (the chart
// envFrom's the teacherKeys secret). Deletes the server pod(s); the Deployment
// recreates them. Used on key rotation, when the core stack is already up.
async function restartServer(ns) {
  await coreApi
    .deleteCollectionNamespacedPod({ namespace: ns, labelSelector: 'app=amortized,component=server' })
    .catch((err) => console.error(`  server restart (${ns}) failed: ${err.message || err}`));
}

// The OpenShell egress + landlock policy baked into the sandbox at create time.
// Static parts (filesystem, landlock, allowed binaries) mirror the opencode kit;
// the egress endpoints are per-provision: the chosen model API host, opencode's
// model catalog + npm (for opencode itself), and the user's own amortized-server
// (the MCP host). Nothing else is reachable, which is what bounds the in-env key.
function buildMortyPolicy(ns, provider) {
  const p = PROVIDERS[provider];
  return {
    version: 1,
    filesystem_policy: {
      include_workdir: true,
      read_only: ['/usr', '/lib', '/lib64', '/bin', '/sbin', '/proc', '/dev/urandom', '/app', '/etc', '/opt', '/var/log'],
      read_write: ['/sandbox', '/workspace', '/tmp', '/dev/null'],
    },
    landlock: { compatibility: 'best_effort' },
    network_policies: {
      morty_egress: {
        name: 'morty-egress',
        endpoints: [
          { host: p.apiHost, port: 443, protocol: 'rest', enforcement: 'enforce', access: 'read-write' },
          { host: 'models.opencode.ai', port: 443, protocol: 'rest', enforcement: 'enforce', access: 'read-only' },
          { host: 'models.dev', port: 443, protocol: 'rest', enforcement: 'enforce', access: 'read-only' },
          { host: 'registry.npmjs.org', port: 443, protocol: 'rest', enforcement: 'enforce', access: 'read-only' },
          { host: `amortized-server.${ns}.svc.cluster.local`, port: 8000, protocol: 'rest', enforcement: 'enforce', access: 'read-write' },
        ],
        binaries: [
          { path: '/usr/local/lib/node_modules/opencode-ai/bin/opencode.exe' },
          { path: '/usr/bin/node' },
          { path: '/usr/bin/curl' },
        ],
      },
    },
  };
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
async function helmInstall(ns, gatewayIP, hasModelKey) {
  if (!CHART_VERSION) throw new Error('AMORTIZED_CHART_VERSION is not set — pin the core chart version');
  const valuesFile = path.join(os.tmpdir(), `values-${ns}.json`);
  fs.writeFileSync(valuesFile, JSON.stringify(userValues(ns, { mortyEnabled: MORTY_ENABLED, gatewayIP, hasModelKey }), null, 2));
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
let openshellConfigured = null;
function configureOpenshell() {
  // Memoize the in-flight/completed promise so concurrent first-time provisions do
  // not both run `gateway remove` + `gateway add` — an interleaved remove can delete
  // the registration the other call just added. Reset on failure so a later provision
  // can retry.
  if (openshellConfigured) return openshellConfigured;
  openshellConfigured = (async () => {
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
  })().catch((err) => { openshellConfigured = null; throw err; });
  return openshellConfigured;
}

// Create + wire the per-user Morty sandbox through the cluster OpenShell gateway.
// The model key is delivered to opencode via its env, and egress to the model API
// (plus opencode's catalog + the user's MCP host) is opened in the create-time
// policy. Idempotent: an existing sandbox is a benign re-provision (a key change
// recreates it via setUserKey), so we just re-assert the gateway route.
async function ensureSandbox(ns, provider, key) {
  await configureOpenshell();
  const p = PROVIDERS[provider];
  if (!p) {
    throw new Error(`unsupported model provider '${provider}' (expected ${SUPPORTED_PROVIDERS.join('|')})`);
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

  // Egress/landlock policy baked at create time (JSON is a valid YAML subset).
  const policyFile = path.join(os.tmpdir(), `policy-${ns}.json`);
  fs.writeFileSync(policyFile, JSON.stringify(buildMortyPolicy(ns, provider)));
  try {
    const createArgs = [
      ...g, 'sandbox', 'create',
      '--name', name,
      '--from', MORTY_IMAGE,
      '--policy', policyFile,
      '--env', `USER_NS=${ns}`,
      '--env', `MORTY_MODEL=${p.model}`,
      // opencode reads <credentialKey> from its env to authenticate the provider;
      // redacted from logs by run(), and reachable only to the allowlisted hosts.
      '--env', `${p.credentialKey}=${key}`,
      '--', 'sh', '-c', serveCmd,
    ];
    try {
      await run(OPENSHELL_BIN, createArgs, { timeout: OPENSHELL_TIMEOUT_MS });
    } catch (err) {
      if (!/exist|already/i.test(err.message)) throw err;
      console.log(`  openshell sandbox ${name} already exists — re-asserting gateway route`);
    }
  } finally {
    try { fs.unlinkSync(policyFile); } catch { /* best-effort cleanup */ }
  }
  // Expose opencode :4096 via the gateway (Host-routed mTLS -> mortyHost).
  await run(OPENSHELL_BIN, [...g, 'service', 'expose', name, '4096', 'opencode'], { timeout: OPENSHELL_TIMEOUT_MS });
}

async function provision(ns, user, keyInfo) {
  console.log(`provisioning ${ns} for ${user}`);
  // Resolve the OpenShell gateway IP first (needed in the chart values for the
  // server's mTLS hostAlias) when Morty is enabled.
  const gatewayIP = MORTY_ENABLED ? await openshellGatewayIP() : '';
  // 1. Residual objects the chart does not create (namespace first, then ns-scoped).
  for (const obj of residualObjects(ns, user)) await applyObject(obj);
  // 1b. Stamp the per-user model key into the ns (before Helm) so the chart wires it
  //     into the server env (list_models + SDG teacher) — the same key Morty uses.
  if (keyInfo) await ensureServerKeySecret(ns, keyInfo);
  // 2. Core stack via Helm from OCI (enterprise MLflow; opencode/studio off).
  await helmInstall(ns, gatewayIP, !!keyInfo);
  // 3. Per-user OpenShell-sandboxed Morty. Best-effort: a sandbox failure leaves the
  //    core stack (server/SDG/MLflow) usable — chat is degraded and can be retried.
  if (MORTY_ENABLED && keyInfo) {
    try {
      await ensureSandbox(ns, keyInfo.provider, keyInfo.key);
    } catch (err) {
      console.error(`  morty sandbox for ${ns} failed (chat unavailable, retryable): ${err.message}`);
    }
  } else if (!MORTY_ENABLED) {
    console.log('  morty automation disabled (OpenShell mTLS certs not mounted)');
  } else {
    console.log('  no model key set — Morty chat disabled until the user provides a key');
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
    entry.promise = (async () => {
      // Per-user BYOK gate: no model key -> ask for one (splash form) before
      // provisioning. Skipped when Morty is disabled (no OpenShell certs mounted).
      const keyInfo = MORTY_ENABLED ? await readUserKey(ns) : null;
      if (MORTY_ENABLED && !keyInfo) { entry.state = 'needs_key'; return; }
      // Fast path: backend already healthy (gateway restart / returning user).
      if (await serverAvailable(ns)) { entry.state = 'ready'; return; }
      await provision(ns, user, keyInfo);
      entry.state = 'ready';
    })().catch((err) => {
      entry.state = 'error';
      entry.error = String(err.message || err);
      console.error(`provisioning failed for ${ns}:`, err?.body || err?.message || err);
    });
    stacks.set(ns, entry);
  }
  return { ns, state: entry.state, error: entry.error, providers: SUPPORTED_PROVIDERS };
}

function getState(user) {
  const ns = nsForUser(user);
  const entry = stacks.get(ns);
  return { ns, state: entry ? entry.state : 'unprovisioned', error: entry?.error || null, providers: SUPPORTED_PROVIDERS };
}

function markForRetry(user) {
  stacks.delete(nsForUser(user));
}

// Set (or rotate) the user's model key: persist it, then (re)provision Morty. First
// set triggers the full stack provision; rotation (stack already up) recreates just
// the provider + sandbox so the new key takes effect without a full re-provision.
async function setUserKey(user, provider, key) {
  if (!SUPPORTED_PROVIDERS.includes(provider)) {
    throw new Error(`unsupported provider '${provider}' (expected ${SUPPORTED_PROVIDERS.join('|')})`);
  }
  if (typeof key !== 'string' || key.trim().length < 8) {
    throw new Error('invalid or missing API key');
  }
  if (!MORTY_ENABLED) {
    throw new Error('Morty is not enabled on this deployment (OpenShell mTLS certs not mounted)');
  }
  const ns = nsForUser(user);
  key = key.trim();
  await writeUserKey(ns, provider, key);

  const existing = stacks.get(ns);
  const rotate = existing && existing.state === 'ready';
  stacks.delete(ns);

  if (rotate) {
    // Core stack is up: update the server-side key (list_models + SDG teacher) and
    // restart the server, then recreate the sandbox — so the new key applies everywhere.
    const keyInfo = { provider, key, credentialKey: PROVIDERS[provider].credentialKey };
    const entry = { state: 'provisioning', error: null };
    entry.promise = (async () => {
      await ensureServerKeySecret(ns, keyInfo);
      await restartServer(ns);
      await run(OPENSHELL_BIN, ['-g', OPENSHELL_GATEWAY, 'sandbox', 'delete', mortyName(ns)], { timeout: OPENSHELL_TIMEOUT_MS }).catch(() => {});
      await ensureSandbox(ns, provider, key);
      entry.state = 'ready';
    })().catch((err) => {
      entry.state = 'error';
      entry.error = String(err.message || err);
      console.error(`morty key rotation failed for ${ns}:`, err?.message || err);
    });
    stacks.set(ns, entry);
    return { ns, state: entry.state, error: entry.error, providers: SUPPORTED_PROVIDERS };
  }
  // First-time set: full provision now that a key exists.
  return ensureUserStack(user);
}

// Current provider status for the settings UI (never returns the key value).
async function getProviderStatus(user) {
  const ns = nsForUser(user);
  const info = await readUserKey(ns);
  return { ns, provider: info ? info.provider : null, providers: SUPPORTED_PROVIDERS };
}

module.exports = { ensureUserStack, getState, markForRetry, setUserKey, getProviderStatus, nsForUser };
