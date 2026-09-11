// Per-user provisioning artifacts for the RHOAI hybrid deployment.
//
// The core stack (server + Postgres + enterprise-MLflow wiring) is installed by
// pulling the amortized Helm chart from OCI (see provision.js) — the chart is the
// single source of truth. This module builds only the pieces the chart cannot
// express:
//   - the per-user Helm values (including the sandboxed-Morty mTLS upstream, wired
//     through the chart's server.extra* hooks);
//   - the residual K8s objects the gateway applies directly (the namespace, the
//     openshell client-cert Secret the server mounts, an optional SDG teacher-keys
//     Secret, a per-user GPU quota).
// The per-user OpenShell egress rule (the user's MCP host) is applied by the
// provisioner via `openshell policy update` (provision.js), not templated here.

const fs = require('fs');
const path = require('path');

// Enterprise MLflow (kubernetes-namespaced auth): workspace == namespace, so the
// chart derives it from .Values.namespace; we only pass the tracking URI here.
const MLFLOW_TRACKING_URI = process.env.SHARED_MLFLOW_TRACKING_URI || '';

// The dir where the gateway mounts the openshell client cert (used both to stamp the
// per-user secret and as the in-server mount path for the mTLS upstream).
const OPENSHELL_MTLS_DIR = process.env.OPENSHELL_MTLS_DIR || '/etc/openshell-mtls';

// Optional overrides passed through to the chart.
const SERVER_IMAGE_TAG = process.env.AMORTIZED_SERVER_IMAGE_TAG || '';
// SDG teacher keys: a dir of provider-env-name files (e.g. OPENAI_API_KEY) the
// gateway mounts and stamps into each user ns as `amortized-teacher-keys`, then
// references via the chart's teacherKeys.existingSecret. Empty => no teacher keys.
const TEACHER_KEYS_DIR = process.env.TEACHER_KEYS_DIR || '';
const TEACHER_KEYS_SECRET = 'amortized-teacher-keys';
const GPU_PER_USER = process.env.GPU_PER_USER || '1';

const labels = { app: 'amortized', 'app.kubernetes.io/managed-by': 'studio-gateway' };

function slugFor(ns) {
  return ns.replace(/^amz-/, '');
}
function mortyName(ns) {
  return `morty-${slugFor(ns)}`;
}
// The Host the OpenShell gateway routes to this sandbox's opencode :4096.
function mortyHost(ns) {
  return `default--${mortyName(ns)}--opencode.openshell.localhost`;
}
function sanitizeLabel(v) {
  return String(v).toLowerCase().replace(/[^a-z0-9._-]/g, '-').slice(0, 63);
}
function mtlsCertsPresent() {
  return fs.existsSync(path.join(OPENSHELL_MTLS_DIR, 'tls.crt'));
}

/**
 * Helm values for a per-user enterprise install into `ns`
 * (namespace == workspace == jobsNamespace). The gateway creates the namespace
 * itself (createNamespaces:false) — the chart would otherwise render two identical
 * Namespace objects when namespace == jobsNamespace.
 * @param {object} opts
 * @param {boolean} opts.mortyEnabled  wire the sandboxed-Morty mTLS upstream
 * @param {string}  opts.gatewayIP     OpenShell gateway ClusterIP (resolved at runtime)
 */
function userValues(ns, { mortyEnabled = false, gatewayIP = '' } = {}) {
  const values = {
    namespace: ns,
    jobsNamespace: ns,
    createNamespaces: false,
    security: { runAsNonRoot: true },
    // Per-user Postgres (app DB) stays bundled; app S3/MinIO is unused under
    // enterprise MLflow (jobs log through the MLflow artifact proxy, AD-3).
    dataStores: { bundled: true },
    minio: { bundled: false },
    mlflow: { enterprise: { enabled: true }, trackingUri: MLFLOW_TRACKING_URI },
    // Morty is the OpenShell sandbox; Studio is served by the gateway itself.
    opencode: { enabled: false },
    studio: { enabled: false },
  };
  if (mortyEnabled) {
    // Sandboxed-Morty mTLS upstream, injected via the chart's server.extra* hooks.
    const host = mortyHost(ns);
    values.server = {
      hostAliases: [{ ip: gatewayIP, hostnames: [host] }],
      extraEnv: [
        { name: 'AMORTIZED_AGENT_UPSTREAM_URL', value: `https://${host}:8080` },
        { name: 'AMORTIZED_AGENT_UPSTREAM_CLIENT_CERT', value: `${OPENSHELL_MTLS_DIR}/tls.crt` },
        { name: 'AMORTIZED_AGENT_UPSTREAM_CLIENT_KEY', value: `${OPENSHELL_MTLS_DIR}/tls.key` },
        { name: 'AMORTIZED_AGENT_UPSTREAM_CA_BUNDLE', value: `${OPENSHELL_MTLS_DIR}/ca.crt` },
      ],
      extraVolumes: [{ name: 'openshell-mtls', secret: { secretName: 'openshell-client-tls' } }],
      extraVolumeMounts: [{ name: 'openshell-mtls', mountPath: OPENSHELL_MTLS_DIR, readOnly: true }],
    };
  }
  if (TEACHER_KEYS_DIR) values.teacherKeys = { existingSecret: TEACHER_KEYS_SECRET };
  if (SERVER_IMAGE_TAG) values.images = { server: { tag: SERVER_IMAGE_TAG } };
  return values;
}

function namespaceManifest(ns, user) {
  return {
    apiVersion: 'v1',
    kind: 'Namespace',
    metadata: {
      name: ns,
      labels: { ...labels, 'amortized.ai/owner': sanitizeLabel(user) },
      annotations: { 'amortized.ai/owner': user },
    },
  };
}

// The openshell client cert Secret, stamped into the user ns from the gateway's
// own mounted copy, so the server can present it for mTLS to the sandbox gateway.
function openshellTlsSecret(ns) {
  const read = (f) => fs.readFileSync(path.join(OPENSHELL_MTLS_DIR, f)).toString('base64');
  return {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: { name: 'openshell-client-tls', namespace: ns, labels },
    type: 'Opaque',
    data: { 'tls.crt': read('tls.crt'), 'tls.key': read('tls.key'), 'ca.crt': read('ca.crt') },
  };
}

// Optional: stamp the SDG teacher-keys Secret from the gateway's mounted dir.
function teacherKeysSecret(ns) {
  if (!TEACHER_KEYS_DIR) return null;
  const data = {};
  for (const f of fs.readdirSync(TEACHER_KEYS_DIR)) {
    if (f.startsWith('.')) continue;
    data[f] = fs.readFileSync(path.join(TEACHER_KEYS_DIR, f)).toString('base64');
  }
  return {
    apiVersion: 'v1',
    kind: 'Secret',
    metadata: { name: TEACHER_KEYS_SECRET, namespace: ns, labels },
    type: 'Opaque',
    data,
  };
}

// One GPU per user by default; tune per policy via GPU_PER_USER.
function gpuQuota(ns) {
  return {
    apiVersion: 'v1',
    kind: 'ResourceQuota',
    metadata: { name: 'gpu-quota', namespace: ns, labels },
    spec: { hard: { 'requests.nvidia.com/gpu': GPU_PER_USER, 'limits.nvidia.com/gpu': GPU_PER_USER } },
  };
}

// Objects the gateway applies directly, in dependency order (namespace first). The
// openshell client-cert secret is stamped only when the gateway has the certs mounted.
function residualObjects(ns, user) {
  return [
    namespaceManifest(ns, user),
    mtlsCertsPresent() ? openshellTlsSecret(ns) : null,
    gpuQuota(ns),
    teacherKeysSecret(ns),
  ].filter(Boolean);
}

module.exports = { userValues, residualObjects, mortyName };
