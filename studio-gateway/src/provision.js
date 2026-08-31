// On-demand provisioning of a user's backend stack via the Kubernetes API.
// Idempotent: applies each manifest with server-side apply, then reports when
// the server Deployment is available. Per-user state is cached in memory so we
// only reconcile once per gateway lifetime (or until a retry is requested).

const k8s = require('@kubernetes/client-node');
const { userStackManifests } = require('./manifests');

const kc = new k8s.KubeConfig();
kc.loadFromCluster();
const objApi = k8s.KubernetesObjectApi.makeApiClient(kc);
const appsApi = kc.makeApiClient(k8s.AppsV1Api);

const FIELD_MANAGER = 'studio-gateway';

// namespace -> { state: 'provisioning'|'ready'|'error', promise, error }
const stacks = new Map();

function nsForUser(user) {
  const local = String(user).split('@')[0];
  const slug = local.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'anon';
  // `amz-` (not `amortized-u-`) so this integrated per-user stack does not collide
  // with pre-existing `amortized-u-*` backends. Keep in sync with manifests.js slug.
  return `amz-${slug}`;
}

async function applyManifest(obj) {
  // Create-if-not-exists (idempotent). We provision once per user, so we don't
  // need update semantics — and this avoids the apply-patch content-type
  // handling differences across @kubernetes/client-node versions.
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

async function serverAvailable(ns) {
  // True readiness: the server answers health. Avoids AppsV1Api return-shape
  // differences and reflects migrations-complete + app-up, not just pod-ready.
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

async function provision(ns, user) {
  console.log(`provisioning stack for ${user} in ${ns}`);
  const manifests = userStackManifests(ns, user);
  for (const m of manifests) {
    await applyManifest(m);
  }
  console.log(`manifests applied for ${ns}; waiting for server`);
  // Wait for the server to become available (migrations run as init container).
  const deadline = Date.now() + 5 * 60 * 1000;
  // Note: Date.now() is fine here — this runs in the gateway service, not a workflow.
  while (Date.now() < deadline) {
    if (await serverAvailable(ns)) return;
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error(`server in ${ns} did not become ready within timeout`);
}

/**
 * Ensure the user's stack exists. Non-blocking: kicks off provisioning and
 * returns the current state immediately. Callers poll getState / retry.
 */
function ensureUserStack(user) {
  const ns = nsForUser(user);
  let entry = stacks.get(ns);
  if (!entry) {
    entry = { state: 'provisioning', error: null };
    // Fast path: if the backend is already healthy (e.g. after a gateway
    // restart, or a returning user), mark ready without re-provisioning.
    // Otherwise provision the full stack.
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
