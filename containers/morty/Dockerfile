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
ARG OPENCODE_VERSION=1.17.1
RUN npm install -g opencode-ai@${OPENCODE_VERSION} && opencode --version

# Persona baked under /workspace (this base has no /sandbox and no fixed sandbox user).
# /workspace is made group-0 writable for OpenShift's random runtime UID (gid 0).
RUN mkdir -p /workspace/.opencode/agents /workspace/skills
COPY morty.md      /workspace/.opencode/agents/morty.md
COPY sdg.md        /workspace/.opencode/agents/sdg.md
COPY training.md   /workspace/.opencode/agents/training.md
COPY skills/       /workspace/skills/
COPY opencode.json /workspace/opencode.json
RUN chown -R 1001:0 /workspace && chmod -R g+rwX /workspace

USER 1001
WORKDIR /workspace
ENV AGENT_NAME=opencode HOME=/workspace
