# Morty image (sandboxed)

The **sandboxed** build of the Morty agent — `opencode` + the Morty persona — meant to run
**inside an OpenShell sandbox** (see the OpenShell platform install in `amortized-deploy`,
`helm/openshell-platform`). This is distinct from the plain Morty the core Helm chart ships
(which mounts the persona via ConfigMap and runs as a normal Deployment).

Published to `ghcr.io/amortized-ai/morty` by `.github/workflows/morty-image.yml`.

## What's baked vs. provided at sandbox-create

The image is a **generic, multi-tenant template** — nothing per-user is baked in.

| Baked | Provided when the sandbox is created (the provisioner) |
|---|---|
| aipcc base + node + `opencode-ai` | the serve command (`opencode serve --port 4096 …`) |
| persona (`morty/sdg/training.md`) + `skills/` from `make prompt` | the egress **policy** (`--policy`), incl. the per-user MCP host |
| `opencode.json` (default model + MCP url + agent disables) | model **creds** (Vertex ADC / project / location, or an OpenAI key) |
| | the **per-user MCP url** (overrides the baked default) |

## Build

Persona/skills are generated from `agents/` (never hand-edited here).

```bash
# assemble the context (runs `make prompt`) — prints the context dir
containers/morty/prepare-context.sh /tmp/morty-ctx
# build amd64 (REQUIRED — an arm64 layer fails at pod init with "exec format error")
docker buildx build --platform linux/amd64 -f /tmp/morty-ctx/Dockerfile -t morty:dev /tmp/morty-ctx
```

CI does this on push to `main` (paths `agents/**`, `containers/morty/**`). To build a
**testable image from a feature branch**, dispatch the workflow on that ref:

```bash
gh workflow run morty-image.yml --ref <branch>    # pushes ghcr.io/amortized-ai/morty:sha-<commit>
```

## Knobs

| Knob | Where | Default | Notes |
|---|---|---|---|
| `BASE_IMAGE` | Dockerfile ARG | `quay.io/aipcc/agentic-ci/openshell:0.3.27` | bump on base upgrades |
| `OPENCODE_VERSION` | Dockerfile ARG | `1.17.1` | pinned for reproducibility |
| default model | `opencode.json` | `google-vertex-anthropic/claude-opus-4-8@default` | matches the core chart; overridable per-sandbox |
| MCP url | `opencode.json` | `amortized-server.amortized.svc…:8000/mcp` | **single-tenant default**; the provisioner overrides it per user (`amz-<user>`) |
| registry/repo | workflow | `ghcr.io/amortized-ai/morty` | tags `:latest` (main) + `:sha-<commit>` |

## Gotchas

- **amd64 only** — a Mac `podman/docker build` produces arm64 → `exec format error` at pod init.
  Use CI or `buildx --platform linux/amd64`.
- **Pin sandboxes by digest**, not tag — nodes cache tags and can serve a stale arch/layer.
- **aipcc base specifics** (already handled in the Dockerfile): no node (installed here), no
  `/sandbox` (persona at `/workspace`), `opencode.exe` under `/usr/local/lib/node_modules/…`
  (the egress **policy** binary path must match — the policy lives with the provisioner).
- **Persona is baked**, so an `agents/` change means a rebuild (inherent to the sandbox model;
  CI makes it cheap). The plain (ConfigMap) Morty updates without a rebuild.

## Follow-up (not in this image)

- **MCP url via env** — cleaner than the provisioner rewriting `opencode.json`; pending
  confirmation that opencode supports env interpolation in config.
- **Provisioning (E3/E4)** — the `Sandbox` CR + policy + creds + MCP-url substitution live with
  the studio-gateway provisioner.
