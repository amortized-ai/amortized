# Sandboxed Morty image — opencode + the Morty persona, run inside an OpenShell sandbox.
# Build context is assembled by prepare-context.sh (persona/skills from `make prompt`).
#
# Build args (pinned defaults; bump only on upgrades):
#   BASE_IMAGE       RHOAI aipcc OpenShell base
#   OPENCODE_VERSION opencode-ai npm version
ARG BASE_IMAGE=quay.io/aipcc/agentic-ci/openshell:0.3.27
FROM ${BASE_IMAGE}
USER 0

# The aipcc base has no node — add it (mirrors the RHOAI Containerfile.openshell).
RUN dnf install -y --nodocs nodejs npm && dnf clean all
ARG OPENCODE_VERSION=1.18.33
RUN npm install -g opencode-ai@${OPENCODE_VERSION} && opencode --version

# opencode's skill tool shells out to ripgrep and otherwise fetches it from GitHub at
# runtime — which the OpenShell egress policy blocks, so the first skill load fails.
# Pre-install rg on PATH; opencode uses it and skips the runtime download + dep-lock step.
ARG RIPGREP_VERSION=15.1.0
RUN curl -fsSL "https://github.com/BurntSushi/ripgrep/releases/download/${RIPGREP_VERSION}/ripgrep-${RIPGREP_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
      | tar -xz -C /tmp \
    && install -m0755 "/tmp/ripgrep-${RIPGREP_VERSION}-x86_64-unknown-linux-musl/rg" /usr/local/bin/rg \
    && rg --version

# Persona baked under /workspace (this base has no /sandbox and no fixed sandbox user).
# /workspace is made group-0 writable for OpenShift's random runtime UID (gid 0).
RUN mkdir -p /workspace/.opencode/agents /workspace/.opencode/skills
COPY morty.md      /workspace/.opencode/agents/morty.md
COPY sdg.md        /workspace/.opencode/agents/sdg.md
COPY training.md   /workspace/.opencode/agents/training.md
COPY eval.md       /workspace/.opencode/agents/eval.md
COPY skills/       /workspace/.opencode/skills/
COPY opencode.json /workspace/opencode.json
RUN chown -R 1001:0 /workspace && chmod -R g+rwX /workspace

USER 1001
WORKDIR /workspace
ENV AGENT_NAME=opencode HOME=/workspace
