# NeMo Gym verifier sidecar — the GRPO reward pool. Runs ONE Gym resources server (verifier)
# next to a training container in the same pod; the reward function calls localhost:8000/verify.
# CPU-only. Every verifier in verifiers.txt ships pre-installed, so the pod needs no network.
#
#   docker build -t ghcr.io/amortized-ai/gym:dev containers/gym   (CI: build-gym in containers.yml)
#   docker run --rm -p 8000:8000 -e GYM_SERVER=<name> ghcr.io/amortized-ai/gym:<tag>
#   docker run --rm --entrypoint python ghcr.io/amortized-ai/gym:<tag> \
#       /opt/amortized-gym/test_verifiers.py          # smoke-test every verifier
#
# Servers are started by run_verifier.sh, not `gym env start`: in Gym v0.7.0 that always boots
# a Ray cluster (one idle worker per visible CPU, >15 GB on a large node) these servers never use.
FROM debian:trixie-slim

ARG GYM_GIT_REF=v0.7.0
# 1: fail the build if any listed verifier cannot be installed or described.
ARG STRICT=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH=/opt/gym/.venv/bin:$PATH

# Gym itself (needs Python >= 3.13.14 — uv downloads a matching interpreter).
RUN git clone --depth 1 --branch "${GYM_GIT_REF}" https://github.com/NVIDIA-NeMo/Gym.git /opt/gym \
    && cd /opt/gym && UV_LINK_MODE=copy uv sync --locked --no-dev --no-cache

COPY verifiers.txt runtime_defaults.yaml make_config.py describe_verifiers.py test_verifiers.py \
    /opt/amortized-gym/
COPY run_verifier.sh /usr/local/bin/run_verifier.sh

WORKDIR /opt/gym
# One layer: generate each runtime config, then install every verifier's isolated venv with
# hardlinks into a shared uv cache, so packages common to all verifiers are stored once
# (dropping the cache afterwards keeps the linked files).
RUN set -u; cd /opt/amortized-gym && mkdir -p configs && failed="" \
    && names=$(grep -vE '^\s*(#|$)' verifiers.txt) \
    && for n in $names; do \
        if ! python make_config.py /opt/gym "$n" runtime_defaults.yaml configs \
            || ! (cd /opt/gym && UV_CACHE_DIR=/opt/uv-cache UV_LINK_MODE=hardlink \
                gym env prefetch "+config_paths=[/opt/amortized-gym/configs/$n.yaml]"); then \
            failed="$failed $n"; rm -f "configs/$n.yaml"; \
        fi; \
    done \
    && rm -rf /opt/uv-cache \
    && echo "$failed" > install_failed.txt \
    && if [ -n "$failed" ]; then echo "INSTALL FAILED:$failed"; [ "$STRICT" = 0 ] || exit 1; fi

# Manifest for amortized's registry: description + request fields of each verifier.
RUN set -u; cd /opt/amortized-gym && failed="" && echo "[" > verifiers.json && sep="" \
    && for cfg in configs/*.yaml; do \
        n=$(basename "$cfg" .yaml); \
        if line=$(cd "/opt/gym/resources_servers/$n" \
                && .venv/bin/python /opt/amortized-gym/describe_verifiers.py "$n" "/opt/amortized-gym/$cfg" 2>/tmp/describe-$n.log); then \
            printf '%s%s\n' "$sep" "$line" >> verifiers.json; sep=","; \
        else failed="$failed $n"; fi; \
    done \
    && echo "]" >> verifiers.json \
    && if [ -n "$failed" ]; then echo "DESCRIBE FAILED:$failed"; [ "$STRICT" = 0 ] || exit 1; fi

ENV GYM_SERVER=mcqa
EXPOSE 8000
CMD ["run_verifier.sh"]
