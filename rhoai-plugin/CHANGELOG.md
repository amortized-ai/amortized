# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Default installation namespace `cp-hello-world` — Helm chart now creates and targets a dedicated namespace via `values.yaml` and a `namespace.yaml` template
- `install.namespace` field in `plugin.yaml` for the community plugin catalog
- BFF `GET /api/config` endpoint exposing the BFF's namespace (`POD_NAMESPACE` env var) for runtime discovery
- `POD_NAMESPACE` env var injected into the BFF container via Kubernetes downward API

### Fixed

- ProjectSelector test `act(...)` warning silenced by wrapping `window.blur` dispatch in `act()`

## [0.4.2] - 2026-07-30

### Added

- Create Project modal in ProjectSelector dropdown — users can create OpenShift projects directly from the selector and auto-land on them
- Optimistic local state update after project creation via `addProject()` in `useProjects` hook
- Window blur handler to close ProjectSelector dropdown when the browser loses focus
- ProjectSelector CSS with border styling matching dashboard conventions
- Resource name auto-generation from display name with DNS validation and manual override
- Namespace labeling (`opendatahub.io/dashboard: 'true'`) on newly created projects

### Changed

- BFF TLS skip now requires explicit `K8S_TLS_INSECURE=true` env var instead of applying unconditionally when `K8S_API_BASE` is set
- `useProjects.refresh()` now returns `Promise<Project[]>` for downstream consumers

### Internal

- Rename-plugin script now resets version to `0.1.0` during seed cleanup (via `npm version` + existing `sync-chart-version` hook)
- Added `K8S_TLS_INSECURE=true` documentation for self-signed cluster certificates across LOCAL_SETUP.md, BFF_PATTERN.md, README.md, and AGENTS.md

## [0.4.1] - 2026-07-15

### Fixed

- Intermittent `ChunkLoadError` on plugin load caused by vendor chunk splitting in the production webpack build ([#24](https://github.com/rh-ai-community-plugins/hello-world/issues/24))
- Last selected project not persisted across page navigations — ProjectSelector now saves the selection to localStorage ([#25](https://github.com/rh-ai-community-plugins/hello-world/issues/25))
- Plugin loading resilience: disabled vendor chunk splitting and switched to eager webpack imports for icons and App component, reducing HTTP requests from 5 to 2
- Container logging: nginx access/error logs routed to stdout/stderr, BFF Express server now includes request logging middleware
- Rename-plugin script: added `src/index.html` title to the change plan, removed incorrect `rhoai-` prefix from generated names, added npm install step for lock file regeneration ([#22](https://github.com/rh-ai-community-plugins/hello-world/issues/22))
- Version sync script now updates all `tag:` entries in `plugin.yaml` (previously missed `bff_image.tag`)

## [0.4.0] - 2026-07-10

### Added

- Three demo pages — User Info, Cluster Resources, and Namespace Summary — each demonstrating a different RHOAI Dashboard integration pattern (dashboard API, K8s pass-through, and BFF)
- Backend-for-Frontend (BFF) service: standalone Express.js + TypeScript backend with namespace and pod aggregation endpoint
- Community plugins shared navigation section with custom SVG icons, designed for multi-plugin ecosystems
- Community Plugin banner across all plugin pages for consistent branding
- Admin-only section on the User Info page with role-based visibility
- ProjectSelector component with localStorage-backed favorites
- Interactive `rename-plugin` script for forking the seed project into a new plugin
- Claude Code skills for plugin development workflows (add/remove pages, add BFF endpoints, rename plugin)
- Makefile with build targets for common development tasks
- Component tests for page components
- Comprehensive documentation: architecture guides, BFF pattern, local dev setup, deployment, and customization reference
- `CHANGELOG.md` and `CONTRIBUTING.md`
- Apache License 2.0 (`LICENSE`)

### Changed

- Unified `plugin.yaml` as a single flat manifest for both Module Federation config and community plugin catalog metadata
- Switched container base images from Alpine to Red Hat UBI9
- Renamed Helm chart to `hello-world-chart` to avoid OCI registry collisions
- Renamed plugin from `hello-plugin-world` to `hello-world`
- Reorganized source files to align with odh-dashboard conventions
- Streamlined CI/CD build-push workflow with BFF support in build and scan scripts
- Automated version sync across `package.json`, `Chart.yaml`, `bff/package.json`, and `plugin.yaml`
- Renamed `CLAUDE.md` to `AGENTS.md` for agent-harness portability (`CLAUDE.md` kept as symlink)

### Fixed

- Relative route handling so the plugin renders correctly inside the dashboard routing context
- Helm chart OpenShift compatibility (security context constraints, route TLS)
- `useK8sResources` loading state stuck true when deselecting a project
- BFF namespace summary now surfaces partial failures in the response instead of silently dropping them
- `useAccessReview` wired into Cluster Resources page to disable create buttons when RBAC denies access
- `useAccessReview` switched from `Promise.all` to `Promise.allSettled` to preserve successful checks
- Added AbortController cleanup to `useCurrentUser`, `useProjects`, and `useNamespaceSummary` hooks
- BFF CA certificate cached at module load instead of reading from disk on every request
- Renamed `useFavoriteProjects.test.ts` to `.spec.ts` to match `jest.config.js` testMatch pattern
- Added missing `[PLUGIN-SPECIFIC]` comments to `namespaceSummaryNavExtension`

## [0.3.0] - 2026-06-25

Initial release of the hello-world community plugin seed project.

### Added

- Webpack 5 Module Federation plugin scaffold for the RHOAI Dashboard
- Hello World demo page with basic extension point registration
- `plugin.yaml` manifest and Helm chart for Kubernetes deployment
- GitHub Actions CI/CD workflows for testing, linting, and container image publishing
- Container image build (`build-push.sh`) and vulnerability scan (`scan-image.sh`) scripts
- Multi-stage `Containerfile` for production builds

[Unreleased]: https://github.com/rh-ai-community-plugins/hello-world/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/rh-ai-community-plugins/hello-world/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/rh-ai-community-plugins/hello-world/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/rh-ai-community-plugins/hello-world/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/rh-ai-community-plugins/hello-world/releases/tag/v0.3.0
