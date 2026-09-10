# RHOAI Dashboard Plugin System

This document explains how the RHOAI Dashboard's modular architecture works and how to build your own plugin.

---

## 1. How the System Works

### What It Is

The ODH Dashboard implements a **Webpack Module Federation** micro-frontend architecture. Features can run in **separate containers** and be dynamically loaded into the main dashboard at runtime. Each plugin contributes navigation items, routes, and UI components without modifying the host application code.

### The Extension Contract

Every plugin exposes an `./extensions` module that exports a default array of typed extension objects. The host dashboard consumes these to build its navigation, routes, and feature areas dynamically.

Available extension types (defined in `packages/plugin-core/src/extension-points/`):

| Type | Purpose |
|---|---|
| `app.area` | Declares a feature area gated by feature flags |
| `app.navigation/href` | Adds a sidebar navigation link |
| `app.navigation/section` | Groups navigation links into a collapsible section |
| `app.route` | Registers a URL route with a lazy-loaded component |
| `app.status-provider` | Provides a status hook for navigation items |
| `app.project-details/tab` | Adds a tab to project details views |
| `app.project-details/overview-section` | Adds a section to the project overview page |
| `app.project-details/settings-card` | Adds a settings card to project details |
| `app.tab-route/page` | Creates a tabbed page that appears as a navigation item |
| `app.tab-route/tab` | Adds a tab to a tabbed page |
| `app.task/group` | Adds a task group to the Task Assistant |
| `app.task/item` | Adds a task item to a task group |
| `app.masthead/brand` | Provides masthead branding (RHAII distributions only) |
| `app.masthead/toolbar-item` | Adds a masthead toolbar item (RHAII only) |
| `app.masthead/user-menu` | Provides user menu content (RHAII only) |
| `app.masthead/about` | Provides about modal content (RHAII only) |

Each extension can declare feature flags (`flags: { required: [...], disallowed: [...] }`) so it only appears when certain features are enabled (or disabled) on the cluster.

Packages can also define their **own** extension points in `frontend/src/odh/extension-points/`. The naming convention is `namespace.section[/sub-section]`.

### How the Backend Discovers Plugins

The configuration is **injected at deploy time**, not baked into the container image.

1. A Kubernetes **ConfigMap** named `federation-config` stores a JSON array of module definitions under the key `module-federation-config.json`
   - File: `manifests/modular-architecture/federation-configmap.yaml`

2. A **kustomize JSON patch** (`manifests/modular-architecture/deployment.yaml`) injects this ConfigMap value as the environment variable `MODULE_FEDERATION_CONFIG` into the dashboard pod

3. At startup, the backend (`packages/app-config/src/module-federation.ts`) reads `process.env.MODULE_FEDERATION_CONFIG`, parses the JSON, and normalizes each entry to the current config format. It then:
   - Registers HTTP reverse proxies at `/_mf/[name]/*` for each module's static assets (`backend/src/routes/module-federation.ts`)
   - Registers API proxies for each module's `proxyService` paths
   - Injects a `<script id="mf-remotes-json">` tag into the HTML with the list of remotes (`backend/src/routes/root.ts`)

4. The frontend (`frontend/src/plugins/useAppExtensions.ts`) reads the injected JSON, initializes `@module-federation/runtime`, and calls `loadRemote('[name]/extensions')` for each module. Loaded extensions are merged with any statically bundled extensions.

5. Components like `ExtensibleNav` (`frontend/src/app/navigation/ExtensibleNav.tsx`) and `AppRoutes` (`frontend/src/app/AppRoutes.tsx`) use `useExtensions()` hooks to render navigation items and routes from the merged extension set.

**In dev mode**, if `MODULE_FEDERATION_CONFIG` is not set, the backend falls back to scanning workspace `package.json` files for `module-federation` properties.

### Module Federation Config Format

The config supports two formats. The **new format** is recommended; the old format is deprecated but still accepted and auto-converted.

**New format** (defined in `packages/app-config/src/types.ts`):

```json
{
  "name": "myPlugin",
  "backend": {
    "remoteEntry": "/remoteEntry.js",
    "authorize": true,
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

Key points:

- `backend` controls how the frontend remote is loaded. The dashboard proxies `/_mf/{name}/*` to the `backend.service`.
- Each `proxyService` entry has its **own** `service`, `authorize`, and `tls` settings -- they are independent of `backend`.
- A module can have `proxyService` without `backend` (API-only proxy, no frontend remote).
- `headers` (optional `Record<string, string>`) can be set on either `backend` or individual `proxyService` entries for custom header forwarding.

**Old format** (deprecated, still supported):

```json
{
  "name": "myPlugin",
  "remoteEntry": "/remoteEntry.js",
  "authorize": true,
  "tls": false,
  "service": { "name": "my-plugin", "namespace": "redhat-ods-applications", "port": 8080 },
  "proxy": [{ "path": "/my-plugin/api", "pathRewrite": "/api" }]
}
```

The backend automatically detects old-format entries (by checking for top-level `remoteEntry`) and converts them to the new format.

### Deployment Models for Plugins

Looking at the current federation ConfigMap, plugins run in one of three ways:

**A. Sidecar containers** -- added to the same pod as the dashboard. Used by most internal plugins. The dashboard's Kubernetes Service exposes additional ports. In the federation config, the `service.name` points to `odh-dashboard`:

| Plugin | Port |
|---|---|
| model-registry | 8043 |
| gen-ai | 8143 |
| maas | 8243 |
| mlflow | 8343 |
| eval-hub | 8543 |
| automl | 8643 |
| autorag | 8743 |
| agent-ops | 8843 |

**B. Separate Kubernetes Services** -- independent Deployments with their own Services. Used by `mlflowEmbedded` (`service.name: "mlflow"`, port 8443). The dashboard backend proxies to them via Kubernetes internal DNS.

**C. Proxy-only** -- no frontend remote, only API proxying. Used by Perses (`service.name: "data-science-perses"`, port 8080). These entries have `proxyService` but no `backend`.

### Who Manages the Config

The `federation-config` ConfigMap is maintained in the dashboard's own Git manifests and applied by kustomize during deployment. The ODH operator deploys the dashboard using these manifests, so it transitively applies the federation config.

To modify which plugins are loaded, you update the ConfigMap either by:

- Editing the manifest YAML in Git and redeploying
- Directly editing the ConfigMap on the cluster with `oc edit configmap federation-config`
- Setting `MODULE_FEDERATION_CONFIG` as an environment variable on the dashboard Deployment (useful when the operator reconciles the ConfigMap)

---

## 2. Recipe: How to Add Your Own Extension

> **ODH vs RHOAI:** The examples below use the RHOAI dashboard namespace `redhat-ods-applications` and deployment name `rhods-dashboard`. If you are running the Open Data Hub (ODH) upstream distribution instead, substitute `opendatahub` for the namespace and `odh-dashboard` for the deployment name throughout.

### Option A: Add to the monorepo (build-time + runtime)

This is the standard approach used by all current plugins. Your module lives in `packages/` and is both bundled at build time and loadable at runtime.

**Step 1**: Scaffold the module from the repo root:

```bash
cd packages
npx mod-arch-installer -n my-module
```

**Step 2**: Update identifiers -- replace placeholder names with your module name in:

- `packages/my-module/package.json` (name, module-federation config)
- `packages/my-module/frontend/src/odh/extensions.ts`
- `packages/my-module/frontend/config/moduleFederation.js`

**Step 3**: Configure `module-federation` in `packages/my-module/package.json`:

```json
"module-federation": {
  "name": "myModule",
  "backend": {
    "remoteEntry": "/remoteEntry.js",
    "authorize": true,
    "tls": true,
    "service": {
      "name": "odh-dashboard",
      "namespace": "opendatahub",
      "port": 8943
    },
    "localService": {
      "host": "localhost",
      "port": 9200
    }
  },
  "proxyService": [
    {
      "path": "/my-module/api",
      "pathRewrite": "/api",
      "authorize": true,
      "tls": true,
      "service": {
        "name": "odh-dashboard",
        "namespace": "opendatahub",
        "port": 8943
      }
    }
  ]
}
```

**Step 4**: Define your extensions in `packages/my-module/extensions.ts`:

```typescript
import type { AreaExtension, HrefNavItemExtension, RouteExtension } from '@odh-dashboard/plugin-core/extension-points';

const extensions: (AreaExtension | HrefNavItemExtension | RouteExtension)[] = [
  {
    type: 'app.area',
    properties: {
      id: 'my-module',
      featureFlags: ['myModuleFlag'],
    },
  },
  {
    type: 'app.navigation/href',
    flags: { required: ['my-module'] },
    properties: {
      id: 'my-module-nav',
      title: 'My Module',
      href: '/my-module',
      section: 'develop-and-train',
      path: '/my-module/*',
    },
  },
  {
    type: 'app.route',
    flags: { required: ['my-module'] },
    properties: {
      path: '/my-module/*',
      component: () => import('./MyModuleRoot'),
    },
  },
];
export default extensions;
```

**Step 5**: Add a feature flag in `frontend/src/concepts/areas/const.ts` if gating is needed.

**Step 6**: Run locally:

```bash
# Terminal 1
cd frontend && npm run start:dev
# Terminal 2
cd backend && npm run start:dev
# Terminal 3
cd packages/my-module && make dev-start-federated
```

**Step 7**: Build the container image for your module and push it to a registry.

**Step 8**: Deploy -- add your module to the cluster:

- Update the `federation-config` ConfigMap to include your module entry
- Either add a sidecar container to the dashboard Deployment, or create a separate Deployment + Service for your module
- If sidecar: add a port to the `odh-dashboard` Service
- Restart/redeploy the dashboard pod

### Option B: Standalone plugin (runtime only, no monorepo changes)

If you want to extend an already-deployed ODH Dashboard without touching the monorepo code. Your plugin runs as its own container and is discovered at runtime.

For working examples of this approach, see the [Community Plugins](COMMUNITY_PLUGINS.md) document.

#### Critical constraint: unpublished type packages

`@odh-dashboard/plugin-core` and `@odh-dashboard/internal` are both marked `"private": true` and **not published to npm**. This means a truly standalone plugin cannot `npm install` them. You have three workarounds:

1. **Copy the type definitions** -- The extension contract is a simple JSON shape. You can define the types yourself (see below). At runtime, the dashboard only checks `extension.type` string values and `extension.properties` shape -- there is no class instantiation or instanceof check.
2. **Reference the monorepo as a git dependency** -- `npm install github:opendatahub-io/odh-dashboard#main` and import from the packages directory. Fragile but works.
3. **Work within the monorepo** (Option A) -- This is the path of least resistance and what all current plugins do.

#### Extension type shapes (for standalone use)

The dashboard loads your `./extensions` module and expects a default export of an array of objects with this shape. No special classes or SDK calls -- just plain objects:

```typescript
// Minimal type definitions you can copy into your standalone project.
// These match what the dashboard host expects at runtime.

type CodeRef<T = any> = () => Promise<{ default: T }>;

interface ExtensionBase<Type extends string, Props> {
  type: Type;
  properties: Props;
  flags?: {
    required?: string[];
    disallowed?: string[];
  };
}

// Registers a route in the dashboard
type RouteExtension = ExtensionBase<'app.route', {
  path: string;
  component: CodeRef<React.ComponentType<any>>;
}>;

// Adds a link to the sidebar navigation
type HrefNavItemExtension = ExtensionBase<'app.navigation/href', {
  id: string;
  title: string;
  href: string;
  section?: string;
  path?: string;
  group?: string;
  label?: string;
  iconRef?: CodeRef;
  statusProviderId?: string;
  dataAttributes?: Record<string, string>;
}>;

// Adds a collapsible section to the sidebar
type NavSectionExtension = ExtensionBase<'app.navigation/section', {
  id: string;
  title: string;
  section?: string;
  group?: string;
  iconRef?: CodeRef;
  label?: string;
  dataAttributes?: Record<string, string>;
}>;

// Declares a feature area (used for feature-flag gating)
type AreaExtension = ExtensionBase<'app.area', {
  id: string;
  featureFlags?: string[];
  reliantAreas?: string[];
}>;

// Creates a tabbed page that appears as a navigation item
type TabRoutePageExtension = ExtensionBase<'app.tab-route/page', {
  id: string;
  title: string;
  href: string;
  path: string;
  section?: string;
  group?: string;
  objectType?: string;
  alwaysShowTabBar?: boolean;
}>;

// Adds a tab to a tabbed page
type TabRouteTabExtension = ExtensionBase<'app.tab-route/tab', {
  pageId: string;
  id: string;
  title: string;
  component: CodeRef;
  group?: string;
  singleTabTitle?: string;
  objectType?: string;
}>;

type Extension = RouteExtension | HrefNavItemExtension | NavSectionExtension
  | AreaExtension | TabRoutePageExtension | TabRouteTabExtension;
```

The existing dashboard navigation sections you can slot into (the `section` property) include:

- `home`
- `ai-hub`
- `develop-and-train`
- `deploy-and-monitor`
- `observe-and-monitor`
- `applications`
- `settings`

You can also create your own section using a `NavSectionExtension` (see the kueue-visualizer example which creates a `community-plugins` section).

#### Step-by-step guide

#### Step 1 -- Scaffold the project

Create a minimal webpack + React + TypeScript project. The critical file structure:

```text
my-plugin/
├── src/
│   ├── rhoai/
│   │   └── extensions.ts      # Extension declarations (default export)
│   ├── app/
│   │   └── MyPluginPage.tsx   # Your main React component
│   ├── bootstrap.tsx          # Async bootstrap (required for MF)
│   └── index.ts               # Webpack entry (imports bootstrap)
├── config/
│   ├── webpack.common.js
│   ├── webpack.dev.js
│   └── webpack.prod.js
├── package.json
├── tsconfig.json
├── Containerfile
├── chart/                     # Helm chart for deployment
└── plugin.yaml                # Plugin metadata
```

#### Step 2 -- Set up package.json with matching dependency versions

You must match the versions used by the dashboard host for shared singleton dependencies. Check the dashboard's `frontend/package.json` for exact versions. As of this codebase:

```json
{
  "name": "my-plugin",
  "private": true,
  "module-federation": {
    "name": "myPlugin",
    "remoteEntry": "remoteEntry.js",
    "authorize": false,
    "tls": false,
    "local": { "host": "localhost", "port": 9111 },
    "service": { "name": "odh-dashboard", "port": 8080 }
  },
  "scripts": {
    "build": "webpack --config config/webpack.prod.js",
    "start:dev": "webpack serve --config config/webpack.dev.js",
    "test": "jest"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^7.17.0",
    "@patternfly/react-core": "^6.4.1",
    "@patternfly/react-icons": "^6.4.0",
    "@openshift/dynamic-plugin-sdk": "^5.0.1"
  },
  "devDependencies": {
    "webpack": "^5.x",
    "webpack-cli": "^5.x",
    "webpack-dev-server": "^5.x",
    "ts-loader": "^9.x",
    "typescript": "^5.x",
    "style-loader": "^4.x",
    "css-loader": "^7.x",
    "html-webpack-plugin": "^5.x"
  }
}
```

#### Step 3 -- Configure webpack

Standalone plugins can use webpack's **built-in** `ModuleFederationPlugin` (from `webpack.container`) -- you do not need `@module-federation/enhanced`. This is simpler and compatible with the dashboard host.

```javascript
// config/webpack.common.js
const { ModuleFederationPlugin } = require('webpack').container;
const HtmlWebpackPlugin = require('html-webpack-plugin');
const path = require('path');

module.exports = {
  entry: './src/index.ts',
  output: {
    publicPath: 'auto',
    filename: '[name].[contenthash].js',
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js'],
  },
  module: {
    rules: [
      { test: /\.tsx?$/, use: 'ts-loader', exclude: /node_modules/ },
      { test: /\.css$/, use: ['style-loader', 'css-loader'] },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: path.resolve(__dirname, '../src/index.html'),
    }),
    new ModuleFederationPlugin({
      name: 'myPlugin',             // Must match the "name" in the ConfigMap
      filename: 'remoteEntry.js',
      exposes: {
        './extensions': './src/rhoai/extensions.ts',  // Required -- the host loads this
      },
      shared: {
        // These are provided by the host at runtime -- your plugin does NOT
        // bundle them. The host shares them as singletons.
        react:                  { singleton: true, requiredVersion: '^18' },
        'react-dom':            { singleton: true, requiredVersion: '^18' },
        'react-router-dom':     { singleton: true, requiredVersion: '^7' },
        '@patternfly/react-core': { singleton: true, requiredVersion: '^6' },
        '@openshift/dynamic-plugin-sdk': {
          singleton: true, requiredVersion: '^5',
        },
      },
    }),
  ],
};
```

Additional shared dependencies to add if your plugin uses them:

- `@patternfly/react-icons` -- shared as singleton by the host
- `@patternfly/react-table` -- if using PatternFly tables
- `@patternfly/react-topology` -- if using topology visualization
- `react-router` -- shared separately from `react-router-dom` by the host

#### Step 4 -- Set up the entry point (required for Module Federation)

Module Federation requires an async bootstrap pattern. Your entry point must dynamically import the actual application:

```typescript
// src/index.ts -- thin entry
import('./bootstrap');
```

```tsx
// src/bootstrap.tsx -- actual app mount
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './app/App';

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
```

#### Step 5 -- Define your extensions

```typescript
// src/rhoai/extensions.ts
const extensions = [
  // 1. Declare a feature area (optional -- only if you need flag gating)
  {
    type: 'app.area' as const,
    properties: {
      id: 'my-plugin-area',
      featureFlags: [] as string[],
    },
  },

  // 2. Add a sidebar navigation link
  {
    type: 'app.navigation/href' as const,
    properties: {
      id: 'my-plugin-nav',
      title: 'My Plugin',
      href: '/my-plugin',
      path: '/my-plugin/*',
      section: 'develop-and-train',
    },
  },

  // 3. Register a route so the component renders when the user navigates there
  {
    type: 'app.route' as const,
    properties: {
      path: '/my-plugin/*',
      component: () => import('../app/MyPluginPage'),   // Lazy-loaded
    },
  },
];

export default extensions;
```

```tsx
// src/app/MyPluginPage.tsx
import React from 'react';
import {
  Page, PageSection, Title, Content,
} from '@patternfly/react-core';

const MyPluginPage: React.FC = () => (
  <Page>
    <PageSection variant="light">
      <Content>
        <Title headingLevel="h1">My Plugin</Title>
        <p>This page is served from a separate container.</p>
      </Content>
    </PageSection>
  </Page>
);

export default MyPluginPage;
```

#### Step 6 -- Containerize

Your container needs to serve the built static files (especially `remoteEntry.js` and the JS chunks). Use nginx with non-root user for OpenShift compatibility:

```dockerfile
# Containerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
RUN echo 'server { \
    listen 8080; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    location /remoteEntry.js { \
        add_header Access-Control-Allow-Origin *; \
    } \
}' > /etc/nginx/conf.d/default.conf

RUN chown -R 1001:0 /var/cache/nginx /var/run /var/log/nginx /usr/share/nginx/html \
    && chmod -R g=u /var/cache/nginx /var/run /var/log/nginx

EXPOSE 8080
USER 1001
CMD ["nginx", "-g", "daemon off;"]
```

Build and push:

```bash
podman build -t quay.io/myorg/my-plugin:latest -f Containerfile .
podman push quay.io/myorg/my-plugin:latest
```

#### Step 7 -- Deploy to the cluster

You can use raw Kubernetes manifests or a Helm chart. Both example community plugins use Helm charts -- see their `chart/` directories for templates.

With raw manifests:

```yaml
# my-plugin-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-plugin
  namespace: my-plugin
  labels:
    app: my-plugin
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-plugin
  template:
    metadata:
      labels:
        app: my-plugin
    spec:
      containers:
        - name: my-plugin
          image: quay.io/myorg/my-plugin:latest
          ports:
            - containerPort: 8080
          securityContext:
            runAsNonRoot: true
            runAsUser: 1001
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            seccompProfile:
              type: RuntimeDefault
          livenessProbe:
            httpGet:
              path: /
              port: 8080
          readinessProbe:
            httpGet:
              path: /
              port: 8080
          resources:
            requests:
              cpu: 50m
              memory: 64Mi
            limits:
              cpu: 100m
              memory: 128Mi
---
apiVersion: v1
kind: Service
metadata:
  name: my-plugin
  namespace: my-plugin
spec:
  selector:
    app: my-plugin
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
```

```bash
oc apply -f my-plugin-deployment.yaml
```

#### Step 8 -- Register in the federation config

The dashboard discovers plugins via the `MODULE_FEDERATION_CONFIG` environment variable. If the RHOAI operator reconciles the `federation-config` ConfigMap, you may need to set the env var directly on the Deployment instead.

**Option A: Edit the ConfigMap** (if the operator doesn't reconcile it):

```bash
oc edit configmap federation-config -n redhat-ods-applications
```

Add to the JSON array in `module-federation-config.json` (new format):

```json
{
  "name": "myPlugin",
  "backend": {
    "remoteEntry": "/remoteEntry.js",
    "authorize": false,
    "tls": false,
    "service": {
      "name": "my-plugin",
      "namespace": "my-plugin",
      "port": 8080
    }
  }
}
```

**Option B: Extend the `MODULE_FEDERATION_CONFIG` env var** (if the operator reconciles the ConfigMap):

```bash
# Get current config
CURRENT=$(oc get deploy rhods-dashboard -n redhat-ods-applications \
  -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MODULE_FEDERATION_CONFIG")].value}')

# Append your plugin (using Python to merge JSON arrays)
NEW=$(python3 -c "
import json, sys
current = json.loads('$CURRENT') if '$CURRENT' else []
current.append({
  'name': 'myPlugin',
  'backend': {
    'remoteEntry': '/remoteEntry.js',
    'authorize': False,
    'tls': False,
    'service': {'name': 'my-plugin', 'namespace': 'my-plugin', 'port': 8080}
  }
})
print(json.dumps(current))
")

# Set the env var
oc set env deploy/rhods-dashboard -n redhat-ods-applications "MODULE_FEDERATION_CONFIG=$NEW"
```

#### Step 9 -- Restart the dashboard

```bash
oc rollout restart deployment/rhods-dashboard -n redhat-ods-applications
```

After restart, the dashboard backend will:

- Proxy `/_mf/myPlugin/*` to `my-plugin.my-plugin.svc.cluster.local:8080`
- Load `remoteEntry.js` from your service
- Dynamically import your `./extensions` module
- Merge your nav items and routes into the dashboard UI

#### What happens at runtime (request flow)

```text
Browser                     Dashboard Backend              Your Plugin Service
  |                              |                              |
  |  GET /                       |                              |
  |  --------------------------> |                              |
  |  <-- HTML with               |                              |
  |      <script id="mf-remotes-json">                         |
  |      [{"name":"myPlugin","remoteEntry":"/remoteEntry.js"}] |
  |                              |                              |
  |  GET /_mf/myPlugin/remoteEntry.js                          |
  |  --------------------------> |  GET /remoteEntry.js         |
  |                              |  --------------------------> |
  |                              |  <-- remoteEntry.js          |
  |  <-- remoteEntry.js          |                              |
  |                              |                              |
  |  (MF runtime resolves        |                              |
  |   ./extensions, loads chunks)|                              |
  |  GET /_mf/myPlugin/chunk.js  |  GET /chunk.js               |
  |  --------------------------> |  --------------------------> |
  |  <-- chunk.js                |  <-- chunk.js                |
  |                              |                              |
  |  Navigation + routes now     |                              |
  |  include your plugin         |                              |
```

#### Important considerations

- **Dependency version alignment**: Shared singleton dependencies (React, PatternFly, react-router) must be version-compatible with the deployed dashboard. Mismatches cause runtime errors. Check the dashboard's `frontend/package.json` for versions.
- **Feature flags**: If your extensions use `flags: { required: ['someFlag'] }`, that flag must exist in the dashboard's area system. If you omit the `flags` property entirely, your extensions are always active (simplest approach for standalone plugins).
- **TLS**: In production ODH/RHOAI deployments, inter-service communication typically uses TLS with cluster-internal certificates. Set `"tls": true` and ensure your container serves over HTTPS if required. For initial testing, `"tls": false` with plain HTTP works if the cluster NetworkPolicy allows it.
- **Authorization**: Set `"authorize": true` if your plugin's backend needs the user's auth token. The dashboard backend will forward the user's OpenShift token in the `Authorization: Bearer <token>` header of the proxied request.
- **No feature flag registration from outside**: A standalone plugin cannot add new entries to the dashboard's `SupportedArea` enum or `dashboardConfig`. If you need a feature flag, either don't use one (extensions are always shown), or coordinate with the dashboard team to add the flag upstream.
- **OpenShift security**: Run containers as non-root (user 1001), drop all capabilities, and set `seccompProfile: RuntimeDefault` to comply with OpenShift's `restricted-v2` SCC.
