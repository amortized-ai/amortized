// Per-user backend stack manifests (server + Postgres), parameterized by
// namespace. The gateway applies these on demand so each dashboard user gets an
// isolated control plane + compute namespace. MLflow/S3 are shared upstreams
// injected via env, so nothing here assumes a per-user MLflow.

const SERVER_IMAGE = process.env.AMORTIZED_SERVER_IMAGE || 'ghcr.io/amortized-ai/amortized:latest';
const POSTGRES_IMAGE = process.env.POSTGRES_IMAGE || 'quay.io/sclorg/postgresql-16-c9s:latest';
const IMAGE_REGISTRY = process.env.AMORTIZED_IMAGE_REGISTRY || 'ghcr.io/amortized-ai';

// Shared upstreams (per the hybrid model: everything per-user except these).
const MLFLOW_TRACKING_URI = process.env.SHARED_MLFLOW_TRACKING_URI || '';
const S3 = {
  AWS_ACCESS_KEY_ID: process.env.S3_ACCESS_KEY_ID || 'minioadmin',
  AWS_SECRET_ACCESS_KEY: process.env.S3_SECRET_ACCESS_KEY || 'minioadmin',
  AWS_S3_ENDPOINT: process.env.S3_ENDPOINT || 'http://minio:9000',
  MLFLOW_S3_ENDPOINT_URL: process.env.S3_ENDPOINT || 'http://minio:9000',
  FSSPEC_S3_ENDPOINT_URL: process.env.S3_ENDPOINT || 'http://minio:9000',
  FSSPEC_S3_KEY: process.env.S3_ACCESS_KEY_ID || 'minioadmin',
  FSSPEC_S3_SECRET: process.env.S3_SECRET_ACCESS_KEY || 'minioadmin',
};

const labels = { app: 'amortized', 'app.kubernetes.io/managed-by': 'studio-gateway' };

/**
 * Returns the ordered list of manifests for a user's backend stack.
 * @param {string} ns  the user's namespace (e.g. amortized-u-meyceoz)
 * @param {string} user  the authenticated username (for annotations)
 */
function userStackManifests(ns, user) {
  return [
    {
      apiVersion: 'v1',
      kind: 'Namespace',
      metadata: {
        name: ns,
        labels: { ...labels, 'amortized.ai/owner': sanitizeLabel(user) },
        annotations: { 'amortized.ai/owner': user },
      },
    },
    // GPU quota — one GPU per user by default; tune per policy.
    {
      apiVersion: 'v1',
      kind: 'ResourceQuota',
      metadata: { name: 'gpu-quota', namespace: ns, labels },
      spec: { hard: { 'requests.nvidia.com/gpu': '1', 'limits.nvidia.com/gpu': '1' } },
    },
    {
      apiVersion: 'v1',
      kind: 'ConfigMap',
      metadata: { name: 'amortized-config', namespace: ns, labels },
      data: {
        AMORTIZED_DATABASE_URL: 'postgresql://amortized:amortized@postgres:5432/amortized',
        AMORTIZED_DATA_DIR: '/data',
        AMORTIZED_COMPUTE_BACKEND: 'kubernetes',
        AMORTIZED_COMPUTE_NAMESPACE: ns, // jobs run in the user's own namespace
        AMORTIZED_IMAGE_REGISTRY: IMAGE_REGISTRY,
        AMORTIZED_EXTERNAL_URL: 'http://amortized-server:8000',
        AMORTIZED_AGENT_SERVER_URL: 'http://amortized-server:8000',
        AMORTIZED_MLFLOW_TRACKING_URI: MLFLOW_TRACKING_URI,
        AMORTIZED_GATEWAY_URL: '',
        AMORTIZED_S3_BUCKET: process.env.S3_BUCKET || 'amortized',
      },
    },
    {
      apiVersion: 'v1',
      kind: 'Secret',
      metadata: { name: 'amortized-s3', namespace: ns, labels },
      type: 'Opaque',
      stringData: S3,
    },
    {
      apiVersion: 'v1',
      kind: 'ServiceAccount',
      metadata: { name: 'amortized-server', namespace: ns, labels },
    },
    // The server launches training/SDG jobs into its own namespace.
    {
      apiVersion: 'rbac.authorization.k8s.io/v1',
      kind: 'Role',
      metadata: { name: 'amortized-job-manager', namespace: ns, labels },
      rules: [
        { apiGroups: ['batch'], resources: ['jobs'], verbs: ['create', 'get', 'list', 'watch', 'delete'] },
        { apiGroups: ['apps'], resources: ['deployments'], verbs: ['create', 'get', 'list', 'watch', 'delete'] },
        { apiGroups: [''], resources: ['services'], verbs: ['create', 'get', 'list', 'delete'] },
        { apiGroups: [''], resources: ['pods', 'pods/log'], verbs: ['get', 'list', 'watch'] },
        { apiGroups: [''], resources: ['secrets', 'configmaps'], verbs: ['create', 'get', 'patch', 'delete'] },
      ],
    },
    {
      apiVersion: 'rbac.authorization.k8s.io/v1',
      kind: 'RoleBinding',
      metadata: { name: 'amortized-job-manager', namespace: ns, labels },
      roleRef: { apiGroup: 'rbac.authorization.k8s.io', kind: 'Role', name: 'amortized-job-manager' },
      subjects: [{ kind: 'ServiceAccount', name: 'amortized-server', namespace: ns }],
    },
    // Postgres (OpenShift-native SCL image; runs under restricted-v2).
    {
      apiVersion: 'apps/v1',
      kind: 'StatefulSet',
      metadata: { name: 'postgres', namespace: ns, labels },
      spec: {
        replicas: 1,
        serviceName: 'postgres',
        selector: { matchLabels: { app: 'amortized', component: 'postgres' } },
        template: {
          metadata: { labels: { app: 'amortized', component: 'postgres' } },
          spec: {
            containers: [{
              name: 'postgres',
              image: POSTGRES_IMAGE,
              ports: [{ name: 'pg', containerPort: 5432 }],
              env: [
                { name: 'POSTGRESQL_DATABASE', value: 'amortized' },
                { name: 'POSTGRESQL_USER', value: 'amortized' },
                { name: 'POSTGRESQL_PASSWORD', value: 'amortized' },
              ],
              resources: { requests: { cpu: '250m', memory: '256Mi' }, limits: { cpu: '1', memory: '1Gi' } },
              volumeMounts: [{ name: 'data', mountPath: '/var/lib/pgsql/data' }],
              readinessProbe: { exec: { command: ['pg_isready', '-U', 'amortized'] }, initialDelaySeconds: 5, periodSeconds: 10 },
            }],
          },
        },
        volumeClaimTemplates: [{
          metadata: { name: 'data', labels: { app: 'amortized', component: 'postgres' } },
          spec: { accessModes: ['ReadWriteOnce'], resources: { requests: { storage: '10Gi' } } },
        }],
      },
    },
    {
      apiVersion: 'v1',
      kind: 'Service',
      metadata: { name: 'postgres', namespace: ns, labels },
      spec: { selector: { app: 'amortized', component: 'postgres' }, ports: [{ name: 'pg', port: 5432, targetPort: 'pg' }] },
    },
    {
      apiVersion: 'v1',
      kind: 'PersistentVolumeClaim',
      metadata: { name: 'amortized-server-data', namespace: ns, labels },
      spec: { accessModes: ['ReadWriteOnce'], resources: { requests: { storage: '20Gi' } } },
    },
    // Server: init containers wait for DB then run migrations, so provisioning
    // is atomic (no separate migration Job to coordinate).
    {
      apiVersion: 'apps/v1',
      kind: 'Deployment',
      metadata: { name: 'amortized-server', namespace: ns, labels: { ...labels, component: 'server' } },
      spec: {
        replicas: 1,
        selector: { matchLabels: { app: 'amortized', component: 'server' } },
        strategy: { type: 'Recreate' },
        template: {
          metadata: { labels: { app: 'amortized', component: 'server' } },
          spec: {
            serviceAccountName: 'amortized-server',
            initContainers: [
              {
                name: 'wait-for-db',
                image: POSTGRES_IMAGE,
                command: ['sh', '-c', 'until pg_isready -h postgres -p 5432 -U amortized; do echo waiting for db; sleep 2; done'],
                securityContext: initSec(),
              },
              {
                name: 'migrate',
                image: SERVER_IMAGE,
                command: ['sh', '-c', 'cd /app && alembic upgrade head'],
                envFrom: [{ configMapRef: { name: 'amortized-config' } }, { secretRef: { name: 'amortized-s3' } }],
                securityContext: initSec(),
              },
            ],
            containers: [{
              name: 'server',
              image: SERVER_IMAGE,
              ports: [{ name: 'http', containerPort: 8000 }],
              envFrom: [{ configMapRef: { name: 'amortized-config' } }, { secretRef: { name: 'amortized-s3' } }],
              volumeMounts: [{ name: 'data', mountPath: '/data' }],
              securityContext: initSec(),
              readinessProbe: { httpGet: { path: '/api/v1/health', port: 'http' }, initialDelaySeconds: 10, periodSeconds: 10 },
              livenessProbe: { httpGet: { path: '/api/v1/health', port: 'http' }, initialDelaySeconds: 15, periodSeconds: 10 },
            }],
            volumes: [{ name: 'data', persistentVolumeClaim: { claimName: 'amortized-server-data' } }],
          },
        },
      },
    },
    {
      apiVersion: 'v1',
      kind: 'Service',
      metadata: { name: 'amortized-server', namespace: ns, labels: { ...labels, component: 'server' } },
      spec: { selector: { app: 'amortized', component: 'server' }, ports: [{ name: 'http', port: 8000, targetPort: 'http' }] },
    },
  ];
}

function initSec() {
  return { allowPrivilegeEscalation: false, capabilities: { drop: ['ALL'] } };
}

function sanitizeLabel(v) {
  return String(v).toLowerCase().replace(/[^a-z0-9._-]/g, '-').slice(0, 63);
}

module.exports = { userStackManifests };
