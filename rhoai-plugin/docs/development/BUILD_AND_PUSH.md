# Build, Push, and Scan Container Images

## Makefile

The project includes a `Makefile` at the repository root that provides a single entry point for all common tasks. Run `make help` to see every target and configurable variable with its default value.

### Quick Reference

| Target | Description |
|---|---|
| `make install` | Install dependencies (frontend + BFF) |
| `make lint` | Lint source code (frontend + BFF) |
| `make typecheck` | TypeScript type checking (frontend + BFF) |
| `make test` | Run tests (frontend + BFF) |
| `make validate` | Full validation: typecheck + lint + test |
| `make build` | Production build (frontend + BFF) |
| `make dev` | Start frontend dev server (port 9500) |
| `make dev-bff` | Start BFF dev server (port 3000) |
| `make image-build` | Build container images locally |
| `make image-push` | Build and push container images |
| `make image-scan` | Build and scan images for vulnerabilities |
| `make chart-package` | Package Helm chart into a `.tgz` archive |
| `make chart-push` | Package and push Helm chart to OCI registry |
| `make clean` | Remove build artifacts |

Most targets have a `-frontend` and `-bff` variant (e.g. `make lint-frontend`, `make test-bff`) for running against a single component.

### Configurable Variables

Override any variable on the command line:

```bash
make image-build BUILDER=docker IMAGE_TAG=dev
make image-push VERSION=0.5.0
make image-scan SEVERITY=MEDIUM
```

| Variable | Description | Default |
|---|---|---|
| `REGISTRY` | Container image registry | `quay.io/rh-ai-community-plugins` |
| `FRONTEND_IMAGE` | Frontend image name | `amortized-studio` |
| `BFF_IMAGE` | BFF image name | `amortized-studio-bff` |
| `CHART_NAME` | Helm chart name | `amortized-studio-chart` |
| `VERSION` | Release version for `image-push` | Auto-computed from git tags |
| `BUILDER` | Container build tool | `podman` |
| `IMAGE_TAG` | Tag for `image-build` / `image-scan` | `latest` |
| `SEVERITY` | Trivy severity filter for `image-scan` | `HIGH,CRITICAL` |

The sections below document the underlying scripts that the Makefile wraps.

---

## Image Details

- **Registry**: `quay.io`
- **Frontend image**: `quay.io/rh-ai-community-plugins/amortized-studio`
- **BFF image**: `quay.io/rh-ai-community-plugins/amortized-studio-bff`

## Prerequisites

- [Podman](https://podman.io/) installed
- [Helm](https://helm.sh/) 3.8+ installed (required for `helm push` to OCI registries)
- [Quay.io](https://quay.io/) account with push access to `rh-ai-community-plugins`

---

## Build and Push

The `scripts/build-push.sh` script builds container images and pushes them to Quay.io.

### Usage

```bash
./scripts/build-push.sh [TARGET] [VERSION]
```

| Argument  | Values                        | Default |
|-----------|-------------------------------|---------|
| `TARGET`  | `frontend`, `bff`, or `all`   | `all`   |
| `VERSION` | Version tag (e.g. `0.5.0`, `0.5.0-rc1`) | Auto-computed from git tags |

When `VERSION` is omitted, the script computes the next minor version from existing git tags and prompts for confirmation before proceeding.

### Examples

```bash
./scripts/build-push.sh                  # Build+push both, auto-version with confirmation
./scripts/build-push.sh frontend         # Build+push frontend only
./scripts/build-push.sh bff 0.5.0-rc1    # Build+push BFF with explicit version
./scripts/build-push.sh all 0.5.0        # Build+push both with explicit version
```

### Manual build and push

To build and push images manually without the script:

```bash
podman login quay.io
podman build -t quay.io/rh-ai-community-plugins/amortized-studio:0.5.0 -f Containerfile .
podman push quay.io/rh-ai-community-plugins/amortized-studio:0.5.0

podman build -t quay.io/rh-ai-community-plugins/amortized-studio-bff:0.5.0 -f bff/Containerfile bff/
podman push quay.io/rh-ai-community-plugins/amortized-studio-bff:0.5.0
```

Buildah and Docker work the same way — substitute `buildah build` or `docker build`/`docker push`.

---

## Vulnerability Scanning

The `scripts/scan-image.sh` script builds container images and scans them for vulnerabilities using [Trivy](https://github.com/aquasecurity/trivy).

### Prerequisites

- [Podman](https://podman.io/) (or Docker)
- [Trivy](https://aquasecurity.github.io/trivy/) — install with `brew install aquasecurity/trivy/trivy`

### Usage

```bash
./scripts/scan-image.sh [TARGET] [SEVERITY]
```

| Argument   | Values                        | Default          |
|------------|-------------------------------|------------------|
| `TARGET`   | `frontend`, `bff`, or `all`   | `all`            |
| `SEVERITY` | Trivy severity levels         | `HIGH,CRITICAL`  |

### Examples

```bash
./scripts/scan-image.sh                  # Scan both frontend and BFF
./scripts/scan-image.sh frontend         # Scan frontend only
./scripts/scan-image.sh bff              # Scan BFF only
./scripts/scan-image.sh all MEDIUM       # Scan both with MEDIUM+ severity
BUILDER=docker ./scripts/scan-image.sh   # Use Docker instead of Podman
```

### Environment Variables

| Variable    | Description                          | Default   |
|-------------|--------------------------------------|-----------|
| `IMAGE_TAG` | Tag applied to built images          | `latest`  |
| `BUILDER`   | Container build tool (`podman`/`docker`) | `podman`  |

---

## Helm Chart

The `chart/` directory contains a Helm chart for deploying both the frontend and BFF to Kubernetes/OpenShift. The chart is published to the same Quay.io registry as an OCI artifact.

- **Chart name**: `amortized-studio-chart`
- **OCI registry**: `oci://quay.io/rh-ai-community-plugins/amortized-studio-chart`

The chart version is kept in sync with the project version in `package.json` by the `scripts/sync-chart-version.js` script, which runs automatically on `npm version`.

### Package

```bash
helm package chart/
```

This produces a `amortized-studio-chart-<version>.tgz` file in the current directory, using the version from `chart/Chart.yaml`.

### Push to OCI Registry

Pushing Helm charts to an OCI registry requires **Helm 3.8+** (`helm push` was added in that release). Check your version with `helm version --short`.

```bash
helm registry login quay.io
helm push amortized-studio-chart-<version>.tgz oci://quay.io/rh-ai-community-plugins
```

Or use the Makefile to package and push in one step:

```bash
make chart-push
```

### Install from OCI Registry

```bash
helm install amortized-studio oci://quay.io/rh-ai-community-plugins/amortized-studio-chart \
  --version 0.4.2 \
  --namespace cp-amortized-studio \
  --create-namespace
```

See [OPENSHIFT_DEPLOY.md](../deployment/OPENSHIFT_DEPLOY.md) for the full deployment guide including dashboard registration.

---

## CI Workflow

The `.github/workflows/build-push.yml` workflow can build and push both images from CI. It is manually triggered via GitHub's `workflow_dispatch` and requires a `version` input. It uses Buildah for builds and pushes to the same Quay.io registry.
