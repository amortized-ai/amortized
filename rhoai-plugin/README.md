# Amortized Studio (RHOAI dashboard plugin)

The Amortized Studio frontend, packaged as a **Webpack Module Federation remote** for the
Red Hat OpenShift AI (RHOAI) dashboard. Webpack builds `remoteEntry.js`; the dashboard
loads it at runtime and mounts Studio as a sidebar item.

**Deployment and dashboard registration are owned by the `amortized-rhoai` umbrella** — the
`studio-gateway` serves this bundle and registers the nav entry. This directory only builds
the frontend image; there is no standalone chart or backend here.

## Image

Built by `.github/workflows/plugin-image.yml` from `Containerfile` and published to
`ghcr.io/amortized-ai/amortized-studio` (public). The umbrella pulls that image.

Build locally:

```bash
podman build -f Containerfile -t amortized-studio .
```

## Local development

Requires Node.js 20+ and a running RHOAI dashboard to load the remote (the plugin runs
inside the dashboard, which proxies its API calls to the cluster).

```bash
npm ci
npm run start:dev     # dev server on :9500
npm run build         # production build to dist/
npm test              # unit tests
npm run typecheck     # tsc --noEmit
npm run lint          # eslint (src/) + markdownlint
```

## Layout

- `src/rhoai/extensions.ts` — the Module Federation extensions the dashboard reads (nav
  section + route). Studio appears under **Community plugins → Amortized Studio → Studio**
  in the sidebar.
- `src/app/` — the embedded Studio app (`StudioEmbed`) + nav icon.
- `config/` — Webpack + Module Federation configuration.

## License

Apache-2.0
