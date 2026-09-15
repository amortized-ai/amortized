# Dashboard Backend APIs and Cluster Interaction

This document covers the dashboard backend features available to plugins and how to interact with the Kubernetes cluster from your plugin's frontend code.

---

## Integration Patterns

This reference plugin demonstrates three integration patterns. Each one covers a different way for your plugin to interact with the cluster and external services. Choose the pattern that fits your use case — most plugins will use a combination of them.

### Pattern 1: Dashboard API — for dashboard-managed data

Call the dashboard's own backend endpoints (`/api/status`, `/api/config`, `/api/dashboardConfig`, etc.) to get user info, feature flags, and dashboard configuration. These endpoints are managed by the dashboard and return pre-processed data.

**When to use:** You need user identity, dashboard settings, or information the dashboard already aggregates.

**Example page:** [User Info](../../src/app/pages/UserInfoPage.tsx) — calls `/api/status` to display the authenticated user's details.

**How it works:** Your frontend code calls `fetch('/api/status')` directly. The dashboard backend handles authentication and returns JSON. No additional backend service needed.

See [section 1.2](#12-dashboard-backend-apis-callable-from-your-plugins-frontend) for the full API reference.

### Pattern 2: K8s API pass-through — for direct Kubernetes operations

Use the dashboard's `/api/k8s/*` proxy to read, create, update, and delete any Kubernetes resource. Requests are forwarded to the cluster API server with the user's RBAC permissions — no cluster-admin needed, users only see and modify what their roles allow.

**When to use:** You need to interact with Kubernetes resources (pods, deployments, services, custom resources, etc.) and the standard K8s API is sufficient.

**Example page:** [Cluster Resources](../../src/app/pages/ClusterResourcesPage.tsx) — creates and lists Deployments and Services in the user's namespaces.

**How it works:** Your frontend code calls `fetch('/api/k8s/apis/apps/v1/namespaces/my-ns/deployments')`. The dashboard backend proxies the request to the K8s API server, forwarding the user's token. The response is the standard Kubernetes API response.

See [section 2](#2-interacting-with-the-cluster-from-your-plugin) for code examples (listing resources, creating deployments, checking RBAC permissions).

### Pattern 3: BFF (Backend For Frontend) — for server-side logic

Deploy your own backend service alongside the plugin. The dashboard proxies requests to it and forwards the user's authentication token. Your BFF can make multiple K8s API calls, talk to external services, keep credentials server-side, or perform any logic that doesn't belong in the browser.

**When to use:**

- You need to aggregate multiple API calls into one response (reduce frontend round-trips)
- You need to call external services (ML platforms, databases, third-party APIs)
- You need to keep API keys or credentials server-side
- You need server-side business logic or heavy data processing

**Example page:** [Namespace Summary](../../src/app/pages/NamespaceSummaryPage.tsx) — calls the plugin's BFF, which lists the user's projects and counts pods per namespace server-side, returning a single aggregated response.

**How it works:** Your frontend calls `fetch('/amortized-studio/api/namespace-summary')`. The dashboard matches the path against the `proxyService` configuration, rewrites it to `/api/namespace-summary`, and forwards the request to your BFF service with the user's Bearer token. Your BFF uses the token to make K8s API calls as the user.

See [section 1.3](#13-using-your-own-backend-bff-pattern) below and the full [BFF Pattern guide](../architecture/BFF_PATTERN.md) for setup details, token flow, and deployment configuration.

### Choosing the right pattern

| I need to... | Pattern |
|---|---|
| Get user info or dashboard settings | Dashboard API |
| List, create, update, or delete K8s resources | K8s pass-through |
| Aggregate multiple K8s calls into one response | BFF |
| Call an external API with server-side credentials | BFF |
| Check user RBAC permissions | K8s pass-through (SelfSubjectAccessReview) |
| Read dashboard feature flags | Dashboard API |
| Perform server-side data processing | BFF |

Most plugins start with patterns 1 and 2, which require no additional backend service. Add a BFF (pattern 3) when you need server-side logic.

---

## 1. Dashboard Backend Features Available to Your Plugin

Since your plugin's frontend code runs inside the same browser page as the dashboard, it shares the same origin and can call all dashboard backend APIs directly via `fetch()`. Your plugin can also optionally have its own backend (BFF) that receives the user's auth token.

### 1.1 Authentication and the Token Flow

The ODH Dashboard sits behind an OAuth proxy (or kube-rbac-proxy) that authenticates users. The flow:

1. User authenticates via OpenShift OAuth
2. The OAuth proxy sets the `x-forwarded-access-token` header with the user's OpenShift Bearer token on every request to the dashboard backend
3. The dashboard backend extracts this token via `getDirectCallOptions()` (`backend/src/utils/directCallUtils.ts`) and uses it for Kubernetes API calls on the user's behalf
4. When the dashboard backend proxies to your plugin's service and `authorize: true` is set, it calls `setAuthorizationHeader()` (`backend/src/utils/proxy.ts`) which converts the user's `x-forwarded-access-token` into an `Authorization: Bearer <token>` header on the proxied request

**What your plugin's BFF receives** (when `authorize: true`):

- `Authorization: Bearer <user-openshift-token>` -- the actual user's OpenShift token, not a service account token
- Any custom headers configured in the `headers` field of the ConfigMap entry

**What your plugin's BFF can do with the token**:

- Make Kubernetes API calls as the authenticated user (respecting their RBAC)
- Call `GET /apis/user.openshift.io/v1/users/~` to get user details
- Perform SelfSubjectAccessReview to check user permissions

### 1.2 Dashboard Backend APIs (callable from your plugin's frontend)

Your plugin's React components run in the browser at the same origin as the dashboard. This means they can call any `/api/*` endpoint directly. Key endpoints:

#### User and Status

| Endpoint | Method | Returns |
|---|---|---|
| `/api/status` | GET | `{ kube: { userName, userID, isAdmin, isAllowed, clusterID, clusterBranding, namespace, serverURL, currentContext, currentUser } }` |
| `/api/config` | GET | Dashboard configuration |
| `/api/dashboardConfig` | GET | `OdhDashboardConfig` CR (feature flags, settings) |

#### Kubernetes Pass-Through

| Endpoint | Method | Returns |
|---|---|---|
| `/api/k8s/*` | GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS | Direct proxy to the Kubernetes API. Actions are performed with the user's RBAC permissions. |

Example: `fetch('/api/k8s/apis/project.openshift.io/v1/projects')` lists the user's projects.

#### Other APIs

| Endpoint | Purpose |
|---|---|
| `/api/components` | Dashboard components (OdhApplication CRs) |
| `/api/namespaces` | Accessible namespaces |
| `/api/notebooks` | Notebook resources |
| `/api/servingRuntimes` | KServe serving runtimes |
| `/api/route` | OpenShift routes |
| `/api/prometheus` | Prometheus metrics |
| `/api/cluster-settings` | Cluster config |
| `/api/connection-types` | Connection type definitions |
| `/api/health` | Backend health check |
| `/api/dsc` | DataScienceCluster CR |
| `/api/dsci` | DSCInitialization CR |
| `/api/integrations` | Integration configurations |
| `/api/modelRegistries` | Model Registry instances |
| `/api/service/pipelines` | ML Pipelines service proxy |
| `/api/service/modelregistry` | Model Registry service proxy |
| `/api/service/trustyai` | TrustyAI service proxy |

### 1.3 Using Your Own Backend (BFF Pattern)

If your plugin needs its own API backend (e.g., to talk to a third-party service), add a `proxyService` entry in the ConfigMap. The dashboard backend will then proxy requests from a URL path you choose to your backend service.

**Example**: Your plugin makes `fetch('/my-plugin/api/data')` -> dashboard backend rewrites to `/api/data` -> forwards to `my-plugin-api.redhat-ods-applications.svc.cluster.local:3000/api/data` with the user's Bearer token (if `authorize: true`).

To set this up, add a `proxyService` entry alongside your `backend` in the federation ConfigMap:

```json
{
  "name": "myPlugin",
  "backend": {
    "remoteEntry": "/remoteEntry.js",
    "authorize": false,
    "tls": false,
    "service": {
      "name": "my-plugin",
      "namespace": "redhat-ods-applications",
      "port": 8080
    }
  },
  "proxyService": [
    {
      "path": "/my-plugin/api",
      "pathRewrite": "/api",
      "authorize": true,
      "tls": false,
      "service": {
        "name": "my-plugin-api",
        "namespace": "redhat-ods-applications",
        "port": 3000
      }
    }
  ]
}
```

Your BFF container can then:

- Extract the user token from the `Authorization` header
- Use it to call the Kubernetes API on the user's behalf
- Call external services (ML platforms, databases, etc.)

### 1.4 What React Hooks and Packages Are Available

**Available via Module Federation shared dependencies**: If a dependency is listed as `shared: { singleton: true }` in the host's webpack config, your plugin gets the host's instance at runtime. This includes React, react-router, PatternFly, and `@openshift/dynamic-plugin-sdk`.

The host also shares all `@odh-dashboard/*` runtime packages as singletons (collected automatically by `frontend/config/getRuntimeOdhPackages.js`). At runtime, if your plugin requests any of these packages, the host provides its copy. However, since these packages are not published to npm, a standalone plugin cannot import them at **build time** for type checking.

**Not directly available from outside the monorepo**:

- `useUser()`, `useAccessReview()`, `useNotification()`, `useAppContext()` -- these live in `@odh-dashboard/internal`, which is not published
- Any hooks or components from `@odh-dashboard/internal`

**Workarounds for a standalone plugin**:

1. **Call `/api/status` directly** -- Instead of `useUser()`, do `fetch('/api/status')` in your component to get user info
2. **Use the K8s pass-through** -- Instead of dedicated hooks, call `/api/k8s/...` endpoints (see section 2 below)
3. **Use PatternFly directly** -- For notifications, use PatternFly's `Alert` or `AlertGroup` components instead of the dashboard's `useNotification()` hook

---

## 2. Interacting with the Cluster from Your Plugin

Your plugin's frontend runs inside the dashboard's browser page and can use the dashboard's backend as a proxy to the Kubernetes API. All requests through `/api/k8s/*` are authenticated as the logged-in user -- the dashboard backend automatically forwards the user's OpenShift token.

### 2.1 Getting User Information

Call `/api/status` to get information about the currently logged-in user:

```typescript
type KubeStatus = {
  currentContext: string;
  currentUser: string;
  namespace: string;
  userName: string;
  userID: string;
  clusterID: string;
  clusterBranding: string;
  isAdmin: boolean;
  isAllowed: boolean;
  serverURL: string;
};

async function getCurrentUser(): Promise<KubeStatus> {
  const response = await fetch('/api/status');
  if (!response.ok) {
    throw new Error(`Failed to fetch status: ${response.status}`);
  }
  const data = await response.json();
  return data.kube;
}
```

As a React hook:

```typescript
import { useState, useEffect } from 'react';

function useCurrentUser() {
  const [user, setUser] = useState<KubeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => { setUser(data.kube); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, []);

  return { user, loading, error };
}
```

### 2.2 Reading Kubernetes Resources

The `/api/k8s/*` endpoint is a pass-through proxy to the Kubernetes API. It supports all HTTP methods and the user's RBAC permissions are enforced by the cluster.

The URL pattern is: `/api/k8s/{kubernetes-api-path}`

Where `{kubernetes-api-path}` is the standard Kubernetes API path you would use with `kubectl` or `curl`.

**Examples:**

```typescript
// List the user's projects
const projects = await fetch('/api/k8s/apis/project.openshift.io/v1/projects')
  .then((res) => res.json());

// List pods in a namespace
const pods = await fetch('/api/k8s/api/v1/namespaces/my-project/pods')
  .then((res) => res.json());

// Get a specific deployment
const deployment = await fetch(
  '/api/k8s/apis/apps/v1/namespaces/my-project/deployments/my-app'
).then((res) => res.json());

// List custom resources (e.g., Kueue ClusterQueues)
const queues = await fetch(
  '/api/k8s/apis/kueue.x-k8s.io/v1beta1/clusterqueues'
).then((res) => res.json());
```

As a reusable React hook (pattern from the [kueue-visualizer](https://github.com/rh-ai-community-plugins/kueue-visualizer) plugin):

```typescript
import { useState, useEffect, useCallback } from 'react';

function useK8sResource<T>(apiPath: string) {
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    fetch(`/api/k8s${apiPath}`)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((list) => { setItems(list.items ?? []); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [apiPath]);

  useEffect(() => { refresh(); }, [refresh]);

  return { items, loading, error, refresh };
}

// Usage:
const { items: pods, loading } = useK8sResource<Pod>(
  '/api/v1/namespaces/my-project/pods'
);
```

### 2.3 Creating and Modifying Kubernetes Resources

The same `/api/k8s/*` proxy supports POST, PUT, PATCH, and DELETE. The user must have the appropriate RBAC permissions for the operation.

**Creating a Deployment:**

```typescript
async function createDeployment(namespace: string, name: string, image: string) {
  const deployment = {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: { name, namespace },
    spec: {
      replicas: 1,
      selector: { matchLabels: { app: name } },
      template: {
        metadata: { labels: { app: name } },
        spec: {
          containers: [{
            name,
            image,
            ports: [{ containerPort: 8080 }],
          }],
        },
      },
    },
  };

  const response = await fetch(
    `/api/k8s/apis/apps/v1/namespaces/${namespace}/deployments`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(deployment),
    },
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || `Failed to create deployment: ${response.status}`);
  }
  return response.json();
}
```

**Creating a Service:**

```typescript
async function createService(namespace: string, name: string, port: number) {
  const service = {
    apiVersion: 'v1',
    kind: 'Service',
    metadata: { name, namespace },
    spec: {
      selector: { app: name },
      ports: [{ port, targetPort: port, protocol: 'TCP' }],
    },
  };

  const response = await fetch(
    `/api/k8s/api/v1/namespaces/${namespace}/services`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(service),
    },
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || `Failed to create service: ${response.status}`);
  }
  return response.json();
}
```

**Deleting a resource:**

```typescript
async function deleteDeployment(namespace: string, name: string) {
  const response = await fetch(
    `/api/k8s/apis/apps/v1/namespaces/${namespace}/deployments/${name}`,
    { method: 'DELETE' },
  );
  if (!response.ok) {
    throw new Error(`Failed to delete deployment: ${response.status}`);
  }
}
```

### 2.4 Checking User Permissions (RBAC)

Before creating resources, check whether the user has the required permissions using a SelfSubjectAccessReview:

```typescript
async function canUserCreate(
  namespace: string,
  group: string,
  resource: string,
): Promise<boolean> {
  const review = {
    apiVersion: 'authorization.k8s.io/v1',
    kind: 'SelfSubjectAccessReview',
    spec: {
      resourceAttributes: {
        namespace,
        verb: 'create',
        group,
        resource,
      },
    },
  };

  const response = await fetch(
    '/api/k8s/apis/authorization.k8s.io/v1/selfsubjectaccessreviews',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(review),
    },
  );

  const result = await response.json();
  return result.status?.allowed === true;
}

// Check if user can create deployments in a namespace
const canDeploy = await canUserCreate('my-project', 'apps', 'deployments');
```

As a React hook:

```typescript
function useCanCreate(namespace: string, group: string, resource: string) {
  const [allowed, setAllowed] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    canUserCreate(namespace, group, resource)
      .then((result) => { setAllowed(result); setLoading(false); })
      .catch(() => { setAllowed(false); setLoading(false); });
  }, [namespace, group, resource]);

  return { allowed, loading };
}
```

### 2.5 Role-Based UI Visibility

Use the `isAdmin` field from `/api/status` to conditionally render content that only RHOAI administrators should see. This is straightforward conditional rendering — the component fetches the user's role and shows or hides a section based on it.

```tsx
import { Card, CardTitle, CardBody } from '@patternfly/react-core';
import { LockIcon } from '@patternfly/react-icons';
import { useCurrentUser } from '~/app/hooks/useCurrentUser';

const MyPage: React.FC = () => {
  const { user } = useCurrentUser();

  return (
    <>
      {/* Content visible to all users */}

      {user?.isAdmin && (
        <Card>
          <CardTitle><LockIcon /> Admin Settings</CardTitle>
          <CardBody>This section is only visible to RHOAI admins.</CardBody>
        </Card>
      )}
    </>
  );
};
```

The same approach works with `isAllowed` or with fine-grained RBAC checks via `useAccessReview` (see [section 2.4](#24-checking-user-permissions-rbac)) — for example, hiding a "Create" button when the user lacks `create` permission on a specific resource.

**Live example:** The [User Info page](../../src/app/pages/UserInfoPage.tsx) includes an admin-only card that is only rendered when `user.isAdmin` is `true`.

### 2.6 Reading Dashboard Configuration

```typescript
// Get dashboard feature flags and settings
const dashboardConfig = await fetch('/api/dashboardConfig').then((res) => res.json());

// Get basic dashboard config
const config = await fetch('/api/config').then((res) => res.json());
```
