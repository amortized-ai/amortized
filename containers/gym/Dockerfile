# NeMo Gym verifier sidecar — runs one Gym resources server (e.g. mcqa) next to a GRPO
# training container in the same pod; the reward function calls localhost:8000/verify.
# CPU-only. Per-server venvs are pre-built (`gym env prefetch`) so the pod needs no network.
#
#   docker build -t ghcr.io/amortized-ai/gym:dev containers/gym   (CI: build-gym in containers.yml)
#   docker run --rm -p 8000:8000 ghcr.io/amortized-ai/gym:<tag>                       # default: mcqa
#   docker run --rm -p 8000:8000 -e GYM_SERVER=<name> ghcr.io/amortized-ai/gym:<tag>   # configs/<name>.yaml
#
# The server is started by run_verifier.sh, not `gym env start`: see the note in configs/mcqa.yaml.
FROM debian:trixie-slim

ARG GYM_GIT_REF=v0.7.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    PATH=/opt/gym/.venv/bin:$PATH

# Gym itself (needs Python >= 3.13.14 — uv downloads a matching interpreter).
RUN git clone --depth 1 --branch "${GYM_GIT_REF}" https://github.com/NVIDIA-NeMo/Gym.git /opt/gym \
    && cd /opt/gym && uv sync --locked --no-dev

COPY configs /opt/amortized-gym/configs

# Pre-build each verifier's isolated venv at image build time, then drop the download cache.
WORKDIR /opt/gym
RUN for cfg in /opt/amortized-gym/configs/*.yaml; do \
        gym env prefetch "+config_paths=[${cfg}]" || exit 1; \
    done \
    && rm -rf /opt/uv-cache

COPY run_verifier.sh /usr/local/bin/run_verifier.sh

ENV GYM_SERVER=mcqa
EXPOSE 8000
CMD ["run_verifier.sh"]
