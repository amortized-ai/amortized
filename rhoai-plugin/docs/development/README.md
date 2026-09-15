# Development

Guides for setting up a local development environment and working with the dashboard backend APIs.

## Documents

- [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md) -- Start here. Maps the directory structure, explains what each piece does, and identifies shared vs plugin-specific files.
- [DASHBOARD_APIS.md](DASHBOARD_APIS.md) -- Covers the three integration patterns (Dashboard API, K8s pass-through, BFF), with a decision guide for choosing the right one. Also includes the full API reference, authentication flow, and code examples.
- [LOCAL_SETUP.md](LOCAL_SETUP.md) -- Complete guide to setting up the RHOAI Dashboard and plugin dev server locally, including prerequisites, dashboard configuration, BFF setup, and hot reload workflow.
- [CUSTOMIZATION.md](CUSTOMIZATION.md) -- Start here when creating a new plugin. Lists the required and optional deliverables (container images, Helm chart, plugin.yaml), their naming conventions, and how to rename identifiers.
- [BUILD_AND_PUSH.md](BUILD_AND_PUSH.md) -- Building and pushing container images (frontend and BFF) to Quay.io, scanning for vulnerabilities with Trivy, and the CI build workflow. Also covers the Makefile that wraps all common build tasks.
- [BEST_PRACTICES.md](BEST_PRACTICES.md) -- Development patterns, common pitfalls, and a pre-PR checklist distilled from real plugin development experience.
- [TESTING.md](TESTING.md) -- Running tests, contract testing principles, and integration test scenarios to verify before shipping.
