# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `amortized-studio`, a community plugin for the **Red Hat OpenShift AI (RHOAI) Dashboard**. It uses Webpack 5 Module Federation to expose remote modules that the RHOAI dashboard host application loads at runtime.

## Build & Development Commands

```bash
npm run start:dev     # Dev server on port 9500 with HMR
npm run build         # Production build to dist/
npm test              # Run all tests (Jest + jsdom)
npm run test:watch    # Watch mode
npm run test:coverage # Tests with coverage report
npm run lint          # ESLint on src/ + markdownlint on **/*.md
```

To run a single test file:

```bash
npx jest src/app/hooks/useCurrentUser.test.ts
```

### BFF Service Commands

```bash
cd bff
K8S_API_BASE=$(oc whoami --show-server) npm run start:dev  # Dev server on port 3000 (K8S_API_BASE required for local dev)
K8S_TLS_INSECURE=true K8S_API_BASE=$(oc whoami --show-server) npm run start:dev  # With self-signed cert
npm run build         # Compile TypeScript to dist/
npm start             # Run compiled server (in-cluster, K8S_API_BASE not needed)
npm test              # Run BFF tests (Jest + node)
npm run lint          # ESLint on bff/src/
```

## Architecture

### Module Federation Plugin System

The plugin exposes two remote modules to the RHOAI dashboard host via Webpack Module Federation (configured inline in `config/webpack.common.js`):

- **`./extensions`** (`src/rhoai/extensions.ts`) — Defines seven extension points:
  - `app.area` — registers the `amortized-studio` feature area
  - `app.navigation/section` (x2) — `community-plugins` shared parent section (with `CommunityNavIcon`) and `amortized-studio` plugin subsection (with `AmortizedStudioNavIcon`)
  - `app.navigation/href` (x3) — "User Info", "Cluster Resources", and "Namespace Summary" nav items under the `amortized-studio` section
  - `app.route` — mounts the App component with wildcard routing at `/amortized-studio/*`
- **`./Icon`** (`src/app/components/AmortizedStudioNavIcon.tsx`) — SVG icon for the plugin's nav subsection. A separate `CommunityNavIcon.tsx` provides the icon for the shared `community-plugins` parent section.

Shared singletons (react, react-dom, react-router-dom, @patternfly/react-core, @openshift/dynamic-plugin-sdk) are provided by the host and not bundled into the plugin.

### Pages

The plugin has three pages, routed under `/amortized-studio/*`, each demonstrating a different integration pattern:

- **User Info page** (`src/app/pages/UserInfoPage.tsx`) — Displays the authenticated user's information via `/api/status` (dashboard API pattern).
- **Cluster Resources page** (`src/app/pages/ClusterResourcesPage.tsx`) — Create and list Deployments and Services via the dashboard's K8s API pass-through (`/api/k8s/*` pattern).
- **Namespace Summary page** (`src/app/pages/NamespaceSummaryPage.tsx`) — Displays aggregated namespace and pod data via the plugin's own BFF service (BFF pattern).

### Custom Hooks

Six hooks in `src/app/hooks/` provide data fetching and API integration:

- `useCurrentUser` — Fetches authenticated user info from `/api/status`.
- `useProjects` — Fetches accessible projects from the OpenShift projects API.
- `useFavoriteProjects` — Manages localStorage-backed project favorites.
- `useK8sResources` — Generic hook for listing K8s resources with create/delete helpers.
- `useAccessReview` — Checks RBAC permissions via SelfSubjectAccessReview.
- `useNamespaceSummary` — Fetches aggregated namespace and pod summary from the BFF endpoint.

### BFF Service

The `bff/` directory contains a standalone Express.js + TypeScript backend service that demonstrates the BFF pattern. The dashboard proxies requests from `/amortized-studio/api/*` to this service, forwarding the user's Bearer token. The BFF exposes a `GET /api/config` endpoint returning `{ bffNamespace }` (sourced from the `POD_NAMESPACE` env var, injected via Kubernetes downward API). See `docs/architecture/BFF_PATTERN.md` for details.

### Entry Point Chain

`src/index.ts` → dynamic import → `src/bootstrap.tsx` (React 18 root render). The dynamic import is required for Module Federation to resolve shared dependencies before the app renders.

### Plugin Registration

`plugin.yaml` at the repo root is a unified flat manifest that serves both as the Module Federation runtime config (consumed by the RHOAI dashboard) and the community plugin catalog metadata (consumed by the charter registry). It declares plugin identity, maintainer, RHOAI version compatibility, deployment model, container image, install method (automatic/assisted/manual with Helm registry and prerequisites), Module Federation remote entry and routes, RBAC requirements, and support links.

### Webpack Configs

- `config/webpack.common.js` — Shared config: entry point, loaders, Module Federation, path alias `~` → `./src`
- `config/webpack.dev.js` — Dev server on port 9500, proxies `/amortized-studio/api` to BFF at `localhost:3000` and `/amortized-studio` to dashboard at `localhost:8443`
- `config/webpack.prod.js` — Output to `dist/`, CSS extraction, vendor chunk splitting

### Test Setup

Jest with `ts-jest` preset and `jsdom` environment (`jest.config.js`). `jest.setup.tsx` mocks `react-router-dom` (useNavigate, useParams, useLocation, Outlet, Routes, Route, Navigate) and polyfills TextEncoder/TextDecoder. CSS modules are proxied to return property names as class names (`jest.style-mock.js`).

### Scripts

- `scripts/build-push.sh` — Builds and pushes container images (frontend, BFF, or both) to Quay.io. Auto-computes the next version from git tags if not provided.
- `scripts/scan-image.sh` — Builds container images locally and scans them for vulnerabilities using Trivy.
- `scripts/rename-plugin.js` — Interactive script to rename all plugin identifiers when forking this seed project into a new plugin. Prompts for a display name and updates all files.
- `scripts/sync-chart-version.js` — Syncs the version from root `package.json` into `chart/Chart.yaml`, `bff/package.json`, and `plugin.yaml` (both `version` and `image.tag`). Runs automatically via npm's `version` lifecycle hook.

### Deployment

- **Frontend container**: Multi-stage build in `Containerfile` — UBI9 Node 22 builder → UBI9 Nginx 1.24 serving `dist/` on port 8080 as UID 1001. Nginx adds CORS header on `remoteEntry.js`.
- **BFF container**: Multi-stage build in `bff/Containerfile` — UBI9 Node 22 builder → UBI9 Node 22 runtime on port 3000 as UID 1001.
- **Helm chart**: `chart/` deploys to Kubernetes with Deployment + Service for both frontend and BFF into the `cp-amortized-studio` namespace by default (configurable via `values.yaml`). Frontend defaults to `quay.io/rh-ai-community-plugins/amortized-studio:latest`, BFF to `quay.io/rh-ai-community-plugins/amortized-studio-bff:latest`.

### CI/CD Workflows

- `.github/workflows/ci.yml` — Runs tests and lint for both frontend and BFF on push/PR to main.
- `.github/workflows/build-push.yml` — Builds and pushes both container images to Quay.io. Manually triggered via `workflow_dispatch` with a version input.

## Documentation

Project documentation lives under `docs/` in semantic subfolders:

```text
docs/architecture/   — Plugin system internals and extension contract
docs/development/    — Local dev setup and dashboard API reference
docs/deployment/     — OpenShift deployment with Helm and dashboard registration
docs/archives/       — Project plan and historical documents
```

## Key Conventions

- Path alias: `~` maps to `./src` (webpack) and `@` maps to `./src` (jest). Use `~` in source code imports.
- UI components use **PatternFly 6** (`@patternfly/react-core`, `@patternfly/react-icons`).
- TypeScript strict mode is enabled. Target is ES2020 with ESNext modules and `react-jsx` transform.
- No standalone ESLint config file — uses `@typescript-eslint` defaults via dev dependencies.
- Plugin-specific identifiers are annotated with `[PLUGIN-SPECIFIC]` comments; shared conventions use `[SHARED]`. See `docs/development/CUSTOMIZATION.md` for the full reference.
