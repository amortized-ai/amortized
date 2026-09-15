# Example Community Plugins

Two community plugins demonstrate the standalone approach for building RHOAI Dashboard plugins.

---

## amortized-studio

**Repository**: [github.com/rh-ai-community-plugins/amortized-studio](https://github.com/rh-ai-community-plugins/amortized-studio)

A minimal "Amortized Studio" plugin that demonstrates:

- Project structure with `src/rhoai/extensions.ts` for extension declarations
- Webpack Module Federation config using `webpack.container.ModuleFederationPlugin`
- Seven extensions: `app.area`, `app.navigation/section` (x2), `app.navigation/href` (x3), `app.route`
- `Containerfile` with Nginx serving static files on port 8080 as non-root user 1001
- Helm chart in `chart/` for deployment
- Three dashboard API integration patterns: `/api/status` (dashboard API), `/api/k8s/*` (K8s pass-through), and BFF service (server-side aggregation)

## kueue-visualizer

**Repository**: [github.com/rh-ai-community-plugins/kueue-visualizer](https://github.com/rh-ai-community-plugins/kueue-visualizer)

A more substantial plugin that demonstrates:

- Multiple pages with separate routes (Queue Infrastructure, Workloads)
- Creating a custom navigation section (`app.navigation/section`)
- **Reading Kubernetes resources** via the `/api/k8s/*` pass-through proxy -- fetches Kueue CRDs (ClusterQueues, LocalQueues, Workloads, ResourceFlavors) and core API resources (Namespaces)
- React hooks pattern for K8s data fetching (`useKueueResources.ts`)
- PatternFly Topology visualization
- Helm chart with ClusterRole for read-only RBAC
- OpenShift security best practices (restricted-v2 SCC compliance)

The kueue-visualizer is the best reference for plugins that need to read cluster data. Its `src/app/hooks/useKueueResources.ts` shows the recommended pattern for calling `/api/k8s/*` endpoints.

---

## Key Reference Files

These files in the [odh-dashboard](https://github.com/opendatahub-io/odh-dashboard) codebase are the most relevant when building plugins:

| File | Purpose |
|---|---|
| `manifests/modular-architecture/federation-configmap.yaml` | ConfigMap defining which modules are loaded at runtime |
| `manifests/modular-architecture/deployment.yaml` | Kustomize patch injecting `MODULE_FEDERATION_CONFIG` env var + sidecar containers |
| `packages/app-config/src/types.ts` | TypeScript types for `ModuleFederationConfig` and `ProxyService` |
| `packages/app-config/src/module-federation.ts` | Config parsing and old-to-new format conversion |
| `backend/src/routes/module-federation.ts` | Backend proxy registration for remotes and proxy services |
| `backend/src/routes/root.ts` | HTML injection of remote list (`mf-remotes-json` script tag) |
| `backend/src/utils/proxy.ts` | Proxy logic and auth header forwarding (`registerProxy`, `setAuthorizationHeader`) |
| `backend/src/utils/directCallUtils.ts` | User token extraction from `x-forwarded-access-token` |
| `backend/src/routes/api/k8s/index.ts` | Kubernetes API pass-through proxy |
| `backend/src/routes/api/status/statusUtils.ts` | User status endpoint implementation |
| `frontend/src/plugins/useAppExtensions.ts` | Frontend runtime module loading |
| `frontend/config/moduleFederation.js` | Host Module Federation webpack config with shared singletons |
| `frontend/config/getRuntimeOdhPackages.js` | Auto-collection of `@odh-dashboard/*` shared packages |
| `frontend/src/app/navigation/ExtensibleNav.tsx` | Dynamic navigation rendering |
| `frontend/src/app/AppRoutes.tsx` | Dynamic route rendering |
| `packages/plugin-core/src/extension-points/` | Extension type definitions |
| `docs/module-federation.md` | Official module federation docs |
| `docs/extensibility.md` | Extension points, code refs, hooks, best practices |
| `docs/onboard-modular-architecture.md` | Official onboarding guide for monorepo modules |
