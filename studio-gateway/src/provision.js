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
// per-user (bring-your-own-key): each user picks a provider (openai|anthropic|vertex)
// and supplies its credential via the Studio splash/settings; the credential is stored
// as a per-user Secret and, at sandbox-create time, given to opencode via its env while
// the provider's API host is opened in the sandbox egress policy. Vertex is ADC-only
// (a Google credentials JSON, not a key string) and Morty-chat-only — see PROVIDERS.
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
//   key provider -> { credential env-var opencode reads, API host to allow, model }
// Model ids are overridable per deployment and track opencode's models.dev naming.
const PROVIDERS = {
  openai: {
    kind: 'key',
    credentialKey: 'OPENAI_API_KEY',
    apiHost: 'api.openai.com',
    model: process.env.MORTY_MODEL_OPENAI || 'openai/gpt-4o',
  },
  anthropic: {
    kind: 'key',
    credentialKey: 'ANTHROPIC_API_KEY',
    apiHost: 'api.anthropic.com',
    model: process.env.MORTY_MODEL_ANTHROPIC || 'anthropic/claude-opus-4-8',
  },
  // Claude via Google Vertex. ADC-only: the credential is a Google application-default-
  // credentials JSON blob (not a key string), so delivery differs from the key providers
  // (see ensureSandbox) — the JSON is written to a file in the sandbox and read via
  // GOOGLE_APPLICATION_CREDENTIALS, project/location come from required gateway env, and
  // the Google token-exchange + Vertex inference hosts are opened in the egress policy
  // (see buildMortyPolicy). Chat-only: data-designer has no vertex provider, so this
  // credential never powers the server model catalog / SDG teacher (kind 'adc' skips it).
  vertex: {
    kind: 'adc',
    model: process.env.MORTY_MODEL_VERTEX || 'google-vertex-anthropic/claude-opus-4-8@default',
    // Per-deployment (required gateway env), read under the names the rest of the
    // opencode stack uses, with the AI-SDK-native names accepted as fallbacks. No
    // default: unset means Vertex is not configured (validated in validateCredential).
    project: process.env.GOOGLE_CLOUD_PROJECT || process.env.GOOGLE_VERTEX_PROJECT || '',
    location: process.env.VERTEX_LOCATION || process.env.GOOGLE_CLOUD_LOCATION || process.env.GOOGLE_VERTEX_LOCATION || '',
    // The ADC JSON is written here in-sandbox; GOOGLE_APPLICATION_CREDENTIALS points to it.
    adcPath: '/workspace/adc.json',
    // Google token exchange (OAuth/STS/IAM); the Vertex inference host is derived from
    // location at policy-build time. Mirrors the validated prior egress recipe.
    tokenHosts: ['oauth2.googleapis.com', 'accounts.google.com', 'sts.googleapis.com', 'www.googleapis.com', 'iam.googleapis.com'],
  },
};
const SUPPORTED_PROVIDERS = Object.keys(PROVIDERS);
// Providers whose credential is a Google ADC JSON (file-delivered, Morty-chat-only)
// rather than an env-key string. Drives the delivery + egress + server-skip branches.
function isAdcProvider(provider) { return PROVIDERS[provider]?.kind === 'adc'; }

// The gateway's own namespace: per-user model-key Secrets live here so a key
// survives stack/sandbox recreation and is readable before amz-<user> exists.
const GATEWAY_NAMESPACE = process.env.GATEWAY_NAMESPACE || readOwnNamespace();

// Morty automation runs only when the OpenShell mTLS certs are mounted. Otherwise
// the core stack is provisioned without chat.
const MORTY_ENABLED = fs.existsSync(path.join(OPENSHELL_MTLS_DIR, 'tls.crt'));

// namespace -> { state: 'provisioning'|'ready'|'error', promise, error }
const stacks = new Map();

// Serialize per-namespace provider mutations (add/remove) so the read-merge-write of the
// credential Secret is atomic and no concurrent save is lost. The gateway is single-replica,
// so an in-process lock suffices. Each mutation chains onto the previous one for the ns.
const providerLocks = new Map(); // ns -> tail promise (always resolves)
function withProviderLock(ns, fn) {
  const prev = providerLocks.get(ns) || Promise.resolve();
  const result = prev.then(fn, fn); // run fn after prev regardless of prev's outcome
  const tail = result.then(() => {}, () => {}); // tail never rejects, so the chain flows
  providerLocks.set(ns, tail);
  tail.then(() => { if (providerLocks.get(ns) === tail) providerLocks.delete(ns); });
  return result;
}

// namespace -> { entry, pending: { requested, restart } } while a reconcile is in flight.
const reconcilers = new Map();

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
// never land in the gateway logs. Also masks the Vertex ADC blob (ADC_B64,
// CREDENTIALS) and the deployment-confidential Vertex project/location
// (PROJECT/LOCATION) — the only `NAME=value` args here are sandbox `--env` pairs.
function redactArg(a) {
  const m = /^([A-Za-z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|KEY|PROJECT|LOCATION|ADC_B64|CREDENTIALS))=.+/.exec(a);
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

// Read the per-user credential set from the gateway namespace. Returns a
// { <provider>: credential } map (credential = key string, or ADC JSON for vertex),
// filtered to supported providers; {} when unset. Bring-your-own-KEYS: a user can
// configure several providers at once (openai + anthropic + vertex), each usable by
// Morty; the key providers additionally power the server catalog + SDG teacher.
async function readUserProviders(ns) {
  try {
    const res = await coreApi.readNamespacedSecret({ name: keySecretName(ns), namespace: GATEWAY_NAMESPACE });
    const data = res?.data || res?.body?.data || {};
    const dec = (v) => Buffer.from(v, 'base64').toString('utf8');
    // Migrate the legacy single-provider shape { provider, key } -> { <provider>: key }.
    // (No provider is literally named 'provider'/'key', so the two together are the old
    // shape; the new shape keys the map by provider name.)
    if (data.provider && data.key) {
      const provider = dec(data.provider);
      return PROVIDERS[provider] ? { [provider]: dec(data.key) } : {};
    }
    const out = {};
    for (const [name, val] of Object.entries(data)) {
      if (PROVIDERS[name] && val) out[name] = dec(val);
    }
    return out;
  } catch (err) {
    const code = err?.code ?? err?.statusCode ?? err?.response?.statusCode;
    if (code === 404) return {};
    throw err;
  }
}

// Names of the configured providers whose credential is an env-key (openai/anthropic),
// i.e. the ones that also power the server model catalog + SDG teacher. Vertex (ADC) is
// Morty-chat-only and excluded.
function keyProviderNames(providers) {
  return Object.keys(providers).filter((n) => PROVIDERS[n] && PROVIDERS[n].kind === 'key');
}

// Create the Secret, or replace it if it already exists. A full replace (PUT) must
// carry the current metadata.resourceVersion, so read it back first; retry if the
// object changes underneath us.
async function upsertSecret(namespace, name, body) {
  try {
    await coreApi.createNamespacedSecret({ namespace, body });
    return;
  } catch (err) {
    const code = err?.code ?? err?.statusCode ?? err?.response?.statusCode;
    if (code !== 409) throw err;
  }
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    const existing = await coreApi.readNamespacedSecret({ name, namespace });
    const meta = existing?.metadata || existing?.body?.metadata || {};
    const put = { ...body, metadata: { ...body.metadata, resourceVersion: meta.resourceVersion } };
    try {
      await coreApi.replaceNamespacedSecret({ name, namespace, body: put });
      return;
    } catch (err) {
      const code = err?.code ?? err?.statusCode ?? err?.response?.statusCode;
      if (code !== 409) throw err;
      lastErr = err;
    }
  }
  throw lastErr;
}

// Persist (create or replace) the per-user credential set as a { <provider>: cred } map.
async function writeUserProviders(ns, providers) {
  const data = {};
  for (const [name, cred] of Object.entries(providers)) {
    data[name] = Buffer.from(cred).toString('base64');
  }
  const body = {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: {
      name: keySecretName(ns),
      namespace: GATEWAY_NAMESPACE,
      labels: { app: 'amortized', 'app.kubernetes.io/managed-by': FIELD_MANAGER, 'amortized.ai/user-ns': ns },
    },
    type: 'Opaque',
    data,
  };
  await upsertSecret(GATEWAY_NAMESPACE, keySecretName(ns), body);
}

// Stamp the per-user KEY-provider credentials into the USER namespace as the chart's
// teacherKeys secret (env-var name -> key), so the same keys reach the server env
// (model_catalog -> list_models + SDG teacher), not just the Morty sandbox. Reflects the
// CURRENT set: providers the user removed drop out, so a re-stamp clears stale keys.
// Vertex (ADC) is Morty-chat-only and contributes nothing here. The secret is always
// (re)written — possibly empty — so it exists for the chart's envFrom (see helmInstall).
async function ensureServerKeySecret(ns, providers) {
  const data = {};
  for (const name of keyProviderNames(providers)) {
    data[PROVIDERS[name].credentialKey] = Buffer.from(providers[name]).toString('base64');
  }
  const body = {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: {
      name: MODEL_KEY_SECRET,
      namespace: ns,
      labels: { app: 'amortized', 'app.kubernetes.io/managed-by': FIELD_MANAGER },
    },
    type: 'Opaque',
    data,
  };
  await upsertSecret(ns, MODEL_KEY_SECRET, body);
}

// Restart the user's server so it re-reads the model key from its env (the chart
// envFrom's the teacherKeys secret). Deletes the server pod(s); the Deployment
// recreates them. Used on key rotation, when the core stack is already up.
async function restartServer(ns) {
  // Let a deletion failure reject so the caller's reconcile surfaces it (logged); the
  // Deployment recreates the pod, which re-reads the teacherKeys secret on startup.
  await coreApi.deleteCollectionNamespacedPod({ namespace: ns, labelSelector: 'app=amortized,component=server' });
}

// The model-API egress endpoint(s) for a provider. Key providers reach one API host;
// the Vertex (ADC) provider needs the Google token-exchange hosts (OAuth/STS/IAM) plus
// the Vertex inference host, derived from location (global -> aiplatform.googleapis.com;
// a region -> <region>-aiplatform.googleapis.com).
function modelEgressEndpoints(p) {
  const rw = (host) => ({ host, port: 443, protocol: 'rest', enforcement: 'enforce', access: 'read-write' });
  if (p.kind === 'adc') {
    const aiplatform = !p.location || p.location === 'global'
      ? 'aiplatform.googleapis.com'
      : `${p.location}-aiplatform.googleapis.com`;
    return [aiplatform, ...p.tokenHosts].map(rw);
  }
  return [rw(p.apiHost)];
}

// The OpenShell egress + landlock policy baked into the sandbox at create time.
// Static parts (filesystem, landlock, allowed binaries) mirror the opencode kit;
// the egress endpoints are per-provision: the chosen model API host(s), opencode's
// model catalog + npm (for opencode itself), and the user's own amortized-server
// (the MCP host). Nothing else is reachable, which is what bounds the in-env credential.
function buildMortyPolicy(ns, providers) {
  // Union the egress endpoints of every configured provider (deduped by host), so a
  // multi-provider sandbox can reach each model API it has a credential for.
  const modelEndpoints = [];
  const seen = new Set();
  for (const name of Object.keys(providers)) {
    if (!PROVIDERS[name]) continue;
    for (const ep of modelEgressEndpoints(PROVIDERS[name])) {
      if (seen.has(ep.host)) continue;
      seen.add(ep.host);
      modelEndpoints.push(ep);
    }
  }
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
          ...modelEndpoints,
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

// Create + wire the per-user Morty sandbox through the cluster OpenShell gateway with the
// user's FULL provider set: every configured provider's credential is delivered via the
// sandbox env (opencode auto-loads each) and egress to each provider's model API (plus
// opencode's catalog + the user's MCP host) is opened in the create-time policy, so Morty's
// model picker offers all of them. Idempotent: an existing sandbox is a benign re-provision,
// so we just re-assert the gateway route (a credential change deletes + recreates it via
// reconcileStack).
async function ensureSandbox(ns, providers) {
  await configureOpenshell();
  const names = Object.keys(providers).filter((n) => PROVIDERS[n]);
  if (names.length === 0) {
    throw new Error('no model providers configured — nothing to run in the sandbox');
  }
  const name = mortyName(ns);
  const g = ['-g', OPENSHELL_GATEWAY];

  // opencode.json needs a default model; the chat UI selects per-request so this is only
  // the fallback. Prefer a Claude-capable provider (vertex, then anthropic) then whatever
  // is configured — matching Studio's catalog ordering.
  const preferred = ['vertex', 'anthropic', 'openai'].find((n) => names.includes(n)) || names[0];
  const defaultModel = PROVIDERS[preferred].model;

  // Rewrite the baked opencode.json in-sandbox: set the per-user MCP URL (amz-<user>)
  // and the provider's model. `node` (present in the image) does a robust JSON edit
  // rather than a brittle sed; USER_NS + MORTY_MODEL are passed as sandbox env. For the
  // ADC (Vertex) provider it also materializes the credentials JSON from its base64 env
  // (MORTY_ADC_B64) to the GOOGLE_APPLICATION_CREDENTIALS path before opencode starts —
  // env-then-write delivery (the prior `sandbox upload` path was broken). A no-op for
  // key providers (MORTY_ADC_B64 unset).
  const rewrite =
    'const fs=require("fs"),f="opencode.json",c=JSON.parse(fs.readFileSync(f));' +
    'c.model=process.env.MORTY_MODEL;' +
    'c.mcp.amortized.url="http://amortized-server."+process.env.USER_NS+".svc.cluster.local:8000/mcp";' +
    'fs.writeFileSync(f,JSON.stringify(c,null,2));' +
    'if(process.env.MORTY_ADC_B64){fs.writeFileSync(process.env.GOOGLE_APPLICATION_CREDENTIALS,Buffer.from(process.env.MORTY_ADC_B64,"base64"))}';
  const serveCmd = `cd /workspace && node -e '${rewrite}' && HOME=/workspace opencode serve --port 4096 --hostname 0.0.0.0`;

  // Egress/landlock policy baked at create time (JSON is a valid YAML subset).
  const policyFile = path.join(os.tmpdir(), `policy-${ns}.json`);
  fs.writeFileSync(policyFile, JSON.stringify(buildMortyPolicy(ns, providers)));
  try {
    // Deliver EVERY configured provider's credential via env (union). Key providers:
    // opencode reads <credentialKey>. ADC (Vertex): the JSON is delivered base64 in env and
    // written to a file by the bootstrap above; opencode reads it via
    // GOOGLE_APPLICATION_CREDENTIALS, with project/location from gateway env. opencode's
    // google-vertex-anthropic loader reads GOOGLE_CLOUD_PROJECT + VERTEX_LOCATION (and
    // ignores opencode.json provider.options), while the underlying @ai-sdk/google-vertex
    // reads GOOGLE_VERTEX_PROJECT/LOCATION — set both name families so it resolves
    // regardless of the baked opencode build. All are redacted from logs by run() and
    // reachable only to the allowlisted hosts.
    const credEnv = [];
    for (const n of names) {
      const p = PROVIDERS[n];
      if (p.kind === 'adc') {
        if (!p.project || !p.location) {
          throw new Error('Vertex provider requires the project and location to be set on the gateway (GOOGLE_CLOUD_PROJECT + VERTEX_LOCATION)');
        }
        credEnv.push(
          '--env', `GOOGLE_APPLICATION_CREDENTIALS=${p.adcPath}`,
          '--env', `GOOGLE_CLOUD_PROJECT=${p.project}`,
          '--env', `GOOGLE_VERTEX_PROJECT=${p.project}`,
          '--env', `VERTEX_LOCATION=${p.location}`,
          '--env', `GOOGLE_VERTEX_LOCATION=${p.location}`,
          '--env', `MORTY_ADC_B64=${Buffer.from(providers[n]).toString('base64')}`,
        );
      } else {
        credEnv.push('--env', `${p.credentialKey}=${providers[n]}`);
      }
    }
    const createArgs = [
      ...g, 'sandbox', 'create',
      '--name', name,
      '--from', MORTY_IMAGE,
      '--policy', policyFile,
      '--env', `USER_NS=${ns}`,
      '--env', `MORTY_MODEL=${defaultModel}`,
      ...credEnv,
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

async function provision(ns, user, providers) {
  console.log(`provisioning ${ns} for ${user}`);
  // Resolve the OpenShell gateway IP first (needed in the chart values for the
  // server's mTLS hostAlias) when Morty is enabled.
  const gatewayIP = MORTY_ENABLED ? await openshellGatewayIP() : '';
  // 1. Residual objects the chart does not create (namespace first, then ns-scoped).
  for (const obj of residualObjects(ns, user)) await applyObject(obj);
  // 1b. In the BYOK hybrid (Morty enabled) ALWAYS create the model-key secret (possibly
  //     empty) and wire teacherKeys to it, so the server env carries every key provider
  //     (list_models + SDG teacher). Doing it unconditionally makes later provider
  //     add/remove a secret update + server restart, never a re-helm, and removes the
  //     ordering gap where a vertex-first user could never wire a key added afterwards.
  //     When Morty is off, keep the old behavior (the deployment's TEACHER_KEYS_DIR path).
  const wireModelKey = MORTY_ENABLED;
  if (wireModelKey) await ensureServerKeySecret(ns, providers);
  // 2. Core stack via Helm from OCI (enterprise MLflow; opencode/studio off).
  await helmInstall(ns, gatewayIP, wireModelKey);
  // 3. Per-user OpenShell-sandboxed Morty with ALL configured providers. Best-effort: a
  //    sandbox failure leaves the core stack (server/SDG/MLflow) usable — chat is degraded
  //    and retryable.
  if (MORTY_ENABLED && Object.keys(providers).length) {
    try {
      await ensureSandbox(ns, providers);
    } catch (err) {
      console.error(`  morty sandbox for ${ns} failed (chat unavailable, retryable): ${err.message}`);
    }
  } else if (!MORTY_ENABLED) {
    console.log('  morty automation disabled (OpenShell mTLS certs not mounted)');
  } else {
    console.log('  no model providers set — Morty chat disabled until the user adds one');
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
      // Per-user BYOK gate: no providers -> ask for at least one (splash form) before
      // provisioning. Skipped when Morty is disabled (no OpenShell certs mounted).
      const providers = MORTY_ENABLED ? await readUserProviders(ns) : {};
      if (MORTY_ENABLED && Object.keys(providers).length === 0) { entry.state = 'needs_key'; return; }
      // Fast path: backend already healthy (gateway restart / returning user). Re-stamp the
      // server key secret and ensure the Morty sandbox exists (both idempotent) so a lost
      // in-memory entry can't leave a server-up/sandbox-gone stack. No restartServer here:
      // the stored keys already match the running server env (genuine changes go through
      // setUserProvider). Sandbox is best-effort so a Morty hiccup still leaves the server usable.
      if (await serverAvailable(ns)) {
        if (MORTY_ENABLED && Object.keys(providers).length) {
          await ensureServerKeySecret(ns, providers);
          try {
            await ensureSandbox(ns, providers);
          } catch (err) {
            console.error(`  morty sandbox reconcile for ${ns} failed (chat unavailable, retryable): ${err.message}`);
          }
        }
        entry.state = 'ready';
        return;
      }
      await provision(ns, user, providers);
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
  const ns = nsForUser(user);
  const entry = stacks.get(ns);
  // Only clear a settled attempt. Deleting an in-flight entry orphans its promise and
  // lets ensureUserStack start a second provision() concurrently (racing helm upgrade
  // --install + duplicate sandbox create), so retrying mid-provision is a no-op.
  if (entry && entry.state === 'provisioning') return;
  stacks.delete(ns);
}

// Validate one provider's credential (throws on invalid); returns the trimmed value.
// Vertex takes a Google ADC JSON and needs the gateway's project/location; key providers
// take a >=8 char key string.
function validateCredential(provider, credential) {
  if (!SUPPORTED_PROVIDERS.includes(provider)) {
    throw new Error(`unsupported provider '${provider}' (expected ${SUPPORTED_PROVIDERS.join('|')})`);
  }
  if (typeof credential !== 'string' || !credential.trim()) {
    throw new Error('invalid or missing credential');
  }
  const cred = credential.trim();
  if (isAdcProvider(provider)) {
    // Vertex (ADC): a Google application-default-credentials JSON blob; project/location must
    // be configured on the gateway. Fail at the form boundary, not deep in sandbox create.
    const p = PROVIDERS[provider];
    if (!p.project || !p.location) {
      throw new Error('Vertex is not configured on this deployment (GOOGLE_CLOUD_PROJECT / VERTEX_LOCATION unset)');
    }
    let adc;
    try { adc = JSON.parse(cred); } catch { throw new Error('Vertex (ADC) requires a valid JSON credentials blob'); }
    if (!adc || typeof adc !== 'object' || !adc.type) {
      throw new Error('Vertex (ADC) JSON must be a Google credentials file (missing "type")');
    }
  } else if (cred.length < 8) {
    throw new Error('invalid or missing API key');
  }
  return cred;
}

// Reconcile the running stack to `providers` (stack already up): update the server
// teacher-key secret (union of key providers; a re-stamp also clears removed ones),
// optionally restart the server so it re-reads them, then recreate the Morty sandbox with
// the full set (or delete it when none remain). No re-helm: teacherKeys is always wired
// (see provision), so a secret update + restart is all the server needs.
async function reconcileStack(ns, providers, { restart }) {
  await ensureServerKeySecret(ns, providers);
  if (restart) await restartServer(ns);
  await run(OPENSHELL_BIN, ['-g', OPENSHELL_GATEWAY, 'sandbox', 'delete', mortyName(ns)], { timeout: OPENSHELL_TIMEOUT_MS }).catch(() => {});
  if (Object.keys(providers).length) await ensureSandbox(ns, providers);
}

// Kick off a best-effort reconcile, tracked as the in-memory entry. A reconcile (server
// restart + sandbox recreate) runs async; if another save lands while one is in flight we do
// NOT start a competing reconcile — we flag a follow-up so that when the current one finishes
// it reconciles again with the LATEST persisted set. This guarantees the stack converges to
// the current credential set (without it, a save during a restart is persisted but never
// applied). The set is always re-read from the Secret, so it reflects every serialized save.
// Crucially, a reconcile failure that leaves the SERVER up keeps the stack 'ready' (only
// Morty/chat degraded) — it must NOT flip the whole stack to 'error' (that would 500 every
// /api call, not just chat). Only a genuinely-down server yields 'error'.
function startReconcile(ns, { restart }) {
  const active = reconcilers.get(ns);
  if (active) {
    active.pending.requested = true;
    active.pending.restart = active.pending.restart || restart;
    return active.entry;
  }
  const entry = { state: 'provisioning', error: null };
  const pending = { requested: false, restart: false };
  const drive = async () => {
    let doRestart = restart;
    for (;;) {
      const providers = await readUserProviders(ns); // always reconcile the latest persisted set
      try {
        await reconcileStack(ns, providers, { restart: doRestart });
        entry.state = 'ready';
        entry.error = null;
      } catch (err) {
        const up = await serverAvailable(ns).catch(() => false);
        entry.state = up ? 'ready' : 'error';
        entry.error = String(err.message || err);
        console.error(`stack reconcile for ${ns} failed (${up ? 'server up, chat degraded' : 'server down'}):`, err?.message || err);
      }
      if (!pending.requested) break;
      pending.requested = false;
      doRestart = pending.restart;
      pending.restart = false;
    }
    reconcilers.delete(ns);
  };
  entry.promise = drive();
  reconcilers.set(ns, { entry, pending });
  stacks.set(ns, entry);
  return entry;
}

// Add or update ONE provider in the user's set (bring-your-own-keys). Non-destructive: the
// other providers are preserved. When the stack is already up, only the server keys (if a
// key provider changed) + the Morty sandbox are updated — no full re-provision. On first run
// (stack not up) it PERSISTS ONLY and defers provisioning to the /gateway/ready trigger (the
// splash's "Continue"), so the user can add several providers first and the stack is
// provisioned once with the full set — avoiding a provision-then-reconcile per extra key.
async function setUserProvider(user, provider, credential) {
  if (!MORTY_ENABLED) {
    throw new Error('Morty is not enabled on this deployment (OpenShell mTLS certs not mounted)');
  }
  const cred = validateCredential(provider, credential);
  const ns = nsForUser(user);

  // Serialize per-ns so the read-merge-write is atomic (a concurrent save can't clobber this
  // provider's credential or be lost to a stale reconcile snapshot).
  return withProviderLock(ns, async () => {
    const current = await readUserProviders(ns);
    const providers = { ...current, [provider]: cred };
    await writeUserProviders(ns, providers);

    // Rotate = the core stack is already up OR a reconcile is in flight. Checking the cluster
    // (serverAvailable) keeps changes applying after a gateway restart cleared the map; the
    // reconcilers check ensures a save that lands mid-restart queues a follow-up reconcile
    // instead of taking the first-run persist-only path (which would never apply it).
    const existing = stacks.get(ns);
    const rotate = reconcilers.has(ns) || (existing && existing.state === 'ready') || (await serverAvailable(ns));

    if (rotate) {
      // Restart the server only when a KEY provider changed (it feeds the server env); a
      // vertex-only add never touches the server, so just recreate the sandbox. startReconcile
      // re-reads the latest persisted set, so it applies this write even if it coalesces.
      const entry = startReconcile(ns, { restart: PROVIDERS[provider].kind === 'key' });
      return { ns, state: entry.state, error: entry.error, providers: SUPPORTED_PROVIDERS, configured: Object.keys(providers) };
    }
    // First run: persist only — the credential is stored, but provisioning waits for the
    // explicit /gateway/ready trigger (Continue), which provisions with the full set.
    return { ns, state: 'needs_key', error: null, providers: SUPPORTED_PROVIDERS, configured: Object.keys(providers) };
  });
}

// Remove ONE provider from the user's set. Re-stamps the server keys (dropping the removed
// key provider) and recreates the sandbox with the remaining set (or deletes it when none remain).
async function removeUserProvider(user, provider) {
  if (!MORTY_ENABLED) {
    throw new Error('Morty is not enabled on this deployment (OpenShell mTLS certs not mounted)');
  }
  const ns = nsForUser(user);
  // Serialize per-ns with the add path so the read-merge-write is atomic.
  return withProviderLock(ns, async () => {
    const current = await readUserProviders(ns);
    if (!(provider in current)) {
      return { ns, state: stacks.get(ns)?.state || 'unprovisioned', error: null, providers: SUPPORTED_PROVIDERS };
    }
    const removedKeyProvider = !!(PROVIDERS[provider] && PROVIDERS[provider].kind === 'key');
    const providers = { ...current };
    delete providers[provider];
    await writeUserProviders(ns, providers);

    const up = reconcilers.has(ns) || stacks.get(ns)?.state === 'ready' || (await serverAvailable(ns));
    if (up) {
      const entry = startReconcile(ns, { restart: removedKeyProvider });
      return { ns, state: entry.state, error: entry.error, providers: SUPPORTED_PROVIDERS };
    }
    return { ns, state: 'unprovisioned', error: null, providers: SUPPORTED_PROVIDERS };
  });
}

// Current provider status for the settings UI: which providers are supported and which the
// user has configured (never returns the credential values).
async function getProviderStatus(user) {
  const ns = nsForUser(user);
  const providers = await readUserProviders(ns);
  return { ns, supported: SUPPORTED_PROVIDERS, configured: Object.keys(providers) };
}

module.exports = { ensureUserStack, getState, markForRetry, setUserProvider, removeUserProvider, getProviderStatus, nsForUser };
