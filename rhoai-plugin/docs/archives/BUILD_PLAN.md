# Build and Push Plan for quay.io

## Current State

- **Project**: RHOAI Amortized Studio Plugin (React/TypeScript with Module Federation)
- **Current Containerfile**: Uses `registry.redhat.io/ubi9/nodejs-20` and `registry.redhat.io/ubi9/nginx-122` base images
- **Build tool**: Webpack (production build via `npm run build`)
- **Target registry**: `quay.io`
- **Container runtime**: Podman (no Docker)

## Challenges

1. **Red Hat registry images** require subscription/authentication and may not be available via podman directly
2. **quay.io repository** needs to be created (or identified) if it doesn't exist yet
3. **Podman login** to quay.io is required before pushing

## Plan

### Step 1: Update Containerfile for Podman Compatibility

Replace Red Hat-specific base images with freely available alternatives:

| Current Image | Replacement |
|---|---|
| `registry.redhat.io/ubi9/nodejs-20:latest` | `registry.access.redhat.com/ubi9/nodejs-20:latest` or `node:20-alpine` |
| `registry.redhat.io/ubi9/nginx-122:latest` | `nginx:alpine` |

**Recommended approach**: Use `node:20-alpine` for the builder stage and `nginx:alpine` for the production stage. These are freely available on Docker Hub and work seamlessly with podman.

### Step 2: Login to Quay.io (do this first)

```bash
podman login quay.io
```

- Enter your quay.io username (under `rh-ai-community-plugins` organization)
- Use an **Access Token** (not password) — generate one at quay.io > Account Settings > Security > Access Tokens

### Step 3: Build the Image with Podman

```bash
# Build the image (tag for quay.io/rh-ai-community-plugins)
podman build -t quay.io/rh-ai-community-plugins/amortized-studio:latest .
```

### Step 4: Push the Image

```bash
podman push quay.io/rh-ai-community-plugins/amortized-studio:latest
```

### Step 5: (Optional) Push with Semantic Version Tag

```bash
podman tag quay.io/rh-ai-community-plugins/amortized-studio:latest quay.io/rh-ai-community-plugins/amortized-studio:0.1.0
podman push quay.io/rh-ai-community-plugins/amortized-studio:0.1.0
```

## Files to Modify

| File | Change |
|---|---|
| [`Containerfile`](Containerfile) | Replace `registry.redhat.io/ubi9/nodejs-20` with `node:20-alpine` and `registry.redhat.io/ubi9/nginx-122` with `nginx:alpine` |

## Pre-requisites Checklist

- [ ] quay.io account created
- [ ] Repository created (e.g., `amortized-studio`)
- [ ] Quay.io Access Token generated
- [ ] Podman installed (confirmed)

## Notes

- The `USER 1001` directive in the Containerfile is fine for both `nginx:alpine` and OpenShift — nginx alpine runs as user `nginx` (UID 101), so we may want to adjust to `USER 1001` which is still non-root and satisfies OpenShift's security constraints.
- The `.env.development` file is not used during the container build (it's for local development). No changes needed there.
