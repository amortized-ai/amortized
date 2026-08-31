# Amortized Studio — Project Plan

> Captured from project planning session, 2026-07-07.

## 1. Project Vision

**amortized-studio** serves a dual purpose for the RHOAI (Red Hat OpenShift AI) community:

1. **Reference implementation** — A working plugin that demonstrates real integration patterns with the RHOAI Dashboard: authentication, project/namespace access, Kubernetes resource management via the dashboard's backend APIs. Not just a skeleton, but a source of practical examples.

2. **Scaffold/seed** — A project that developers can clone and use as a starting point for their own RHOAI community plugins. The structure, configuration, and patterns are designed to be extended or trimmed down.

Beyond code, the project should **fully document** how to set up a local development environment, including running the RHOAI Dashboard itself in dev mode so plugin authors can see their plugin integrating in real time with hot reload.

## 2. Plugin UI Structure

### Navigation

- **Section**: Reuse the `community-plugins` section (id: `community-plugins`, group: `9_plugins`) with the same pattern as the [kueue-visualizer plugin](https://github.com/rh-ai-community-plugins/kueue-visualizer). This groups community plugins together in the dashboard sidebar.
- **Label**: Each nav item carries `label: 'Community'` flair, matching the kueue-visualizer convention.
- **Icon**: Keep the existing AmortizedStudio nav icon or adapt it.

### Pages

The current single "Amortized Studio" page (click counter + welcome alert) is **replaced entirely** by two pages:

#### Page 1 — User & Projects

- **Purpose**: The "read" side of cluster interaction. Welcoming landing page with a "Amortized Studio" feel.
- **Content**:
  - Welcome message ("Hello, welcome to your plugin example")
  - Display the logged-in user's information (username, admin status, cluster info) retrieved via `/api/status`
  - A project/namespace dropdown populated from `/api/k8s/apis/project.openshift.io/v1/projects`, showing only projects the authenticated user has access to
  - RBAC permission checks using `SelfSubjectAccessReview` to show what the user can do in the selected project

#### Page 2 — Cluster Resources

- **Purpose**: The "write" side of cluster interaction. Demonstrates creating and managing Kubernetes resources.
- **Content**:
  - Namespace selector (reusing the project dropdown pattern from Page 1)
  - Create a standard Kubernetes object (e.g., Deployment, Service) as the authenticated user via `/api/k8s/*` pass-through
  - List existing resources in the selected namespace
  - Delete resources
  - All operations respect the authenticated user's RBAC — the plugin uses the dashboard's K8s pass-through proxy which enforces user permissions, not the dashboard service account's permissions

### Key Principle

All cluster interactions should go through the dashboard's existing backend APIs (`/api/status`, `/api/k8s/*`, etc.) rather than implementing custom backends. This teaches plugin authors to reuse the dashboard's infrastructure, including its RBAC enforcement via the user's OpenShift token.

## 3. Extension Points

The plugin will define these extensions (in `src/rhoai/extensions.ts`):

1. **`app.area`** — Registers the `amortized-studio` feature area
2. **`app.navigation/section`** — Defines the `community-plugins` sidebar section (same id as kueue-visualizer so they group together)
3. **`app.navigation/href`** (x2) — Two nav items under the community section, each with `label: 'Community'`
4. **`app.route`** (x2) — Routes for the two pages

## 4. Documentation Reorganization

### Structure

All documentation lives under `docs/` with semantic subfolders. Every document is referenced in a README — no orphans.

```text
docs/
├── README.md                        # Top-level index with pointers to subfolders
├── architecture/
│   └── README.md                    # Plugin system architecture, extension points, Module Federation
├── development/
│   └── README.md                    # Local dev setup, dashboard integration, dev workflow
├── deployment/
│   └── README.md                    # Container build, Helm chart, cluster registration
└── archives/
    ├── README.md                    # Historical/planning documents
    ├── PROJECT_PLAN.md              # This file
    └── BUILD_PLAN.md                # Migration plan from Red Hat base images (moved from docs/)
```

### Content Redistribution

- **Root `README.md`** — Becomes a concise project intro with quick start and pointers to the docs subfolders. No longer the full guide.
- **`docs/plugin-guide.md`** — Content gets split across architecture and development folders (it covers both). The file itself is removed once content is redistributed.
- **`docs/BUILD_AND_PUSH.md`** — Moves under `docs/deployment/`
- **`docs/BUILD_PLAN.md`** — Moves to `docs/archives/` (internal planning artifact, kept for reference)

### New Content

- **Development guide** — Full local development setup including:
  - Cloning and setting up the dashboard (`opendatahub-io/odh-dashboard`)
  - Switching to the right version/tag (e.g., 3.4.4)
  - Configuring `env.local` with `MODULE_FEDERATION_CONFIG`
  - Running dashboard backend and frontend in separate terminals
  - Running the plugin in dev mode on a specific port
  - Seeing live changes with hot reload

## 5. Dashboard Dev Environment Setup (Reference)

This is the development workflow that needs to be properly documented:

### Dashboard Setup

1. Clone `https://github.com/opendatahub-io/odh-dashboard`
2. Switch to the desired version/tag (latest: 3.4.4)
3. Copy `env.local.example` to `env.local`
4. Run `npm install`
5. Log onto the cluster as cluster-admin: `oc login ...`
6. Open two terminals:
   - `cd backend && npm run start:dev`
   - `cd frontend && npm run start:dev`
7. Dashboard available at `http://localhost:4010`

### Plugin Development

1. Run the plugin in dev mode: `npm run start:dev` (listens on configured port)
2. In the dashboard's `env.local`, add the plugin to `MODULE_FEDERATION_CONFIG`:

   ```json
   {
     "name": "amortizedStudio",
     "backend": {
       "remoteEntry": "/remoteEntry.js",
       "tls": false,
       "localService": { "host": "localhost", "port": 9112 },
       "service": { "name": "placeholder", "namespace": "opendatahub", "port": 8080 }
     }
   }
   ```

3. Restart the dashboard backend, then frontend
4. Plugin appears in the dashboard with live reload

### Port Conventions

The plugin's dev port is configurable. Default in this project is 9112. Standard RHOAI plugin ports (avoid conflicts):

- 9100: modelRegistry
- 9102: genAi
- 9104: maas
- 9105: notebooks
- 9106: evalHub
- 9107: autorag
- 9108: automl
- 9110: mlflow
- 9111: agentOps

## 6. Technical Reference

### Dashboard Backend APIs Used by This Plugin

| API | Purpose | Used In |
|---|---|---|
| `GET /api/status` | Get authenticated user info (userName, userID, isAdmin, clusterID, etc.) | Page 1 |
| `GET /api/k8s/apis/project.openshift.io/v1/projects` | List user's accessible projects/namespaces | Page 1, Page 2 |
| `POST /api/k8s/apis/authorization.k8s.io/v1/selfsubjectaccessreviews` | Check user RBAC permissions | Page 1, Page 2 |
| `GET /api/k8s/apis/apps/v1/namespaces/{ns}/deployments` | List deployments in a namespace | Page 2 |
| `POST /api/k8s/apis/apps/v1/namespaces/{ns}/deployments` | Create a deployment | Page 2 |
| `DELETE /api/k8s/apis/apps/v1/namespaces/{ns}/deployments/{name}` | Delete a deployment | Page 2 |
| `GET /api/k8s/api/v1/namespaces/{ns}/services` | List services | Page 2 |
| `POST /api/k8s/api/v1/namespaces/{ns}/services` | Create a service | Page 2 |
| `DELETE /api/k8s/api/v1/namespaces/{ns}/services/{name}` | Delete a service | Page 2 |

### Key Dependency Versions (must match dashboard host)

- React 18
- react-router-dom 7
- PatternFly 6 (@patternfly/react-core, @patternfly/react-icons)
- @openshift/dynamic-plugin-sdk 5

### Existing Reference Documentation

- `docs/plugin-guide.md` — Comprehensive guide to the dashboard plugin system (to be reorganized)
- [kueue-visualizer](https://github.com/rh-ai-community-plugins/kueue-visualizer) — Example community plugin with K8s API integration
- [odh-dashboard](https://github.com/opendatahub-io/odh-dashboard) — The host dashboard codebase
