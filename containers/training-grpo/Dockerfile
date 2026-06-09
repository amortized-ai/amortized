# Training container — GRPO (CUDA-enabled)
# Multi-stage build: CUDA devel (builder) → CUDA runtime (slim)
#
# Build: docker build -f containers/training-grpo/Dockerfile -t ghcr.io/amortized-ai/training-grpo .
# Run:   docker run --gpus all ghcr.io/amortized-ai/training-grpo

# =============================================================================
# Stage 1: Builder — install Python deps with full build toolchain
# =============================================================================
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg2 \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python3.11 -m pip install --no-cache-dir "training-hub[grpo]"
RUN python3.11 -m pip install --no-cache-dir --no-build-isolation "training-hub[cuda]"

# =============================================================================
# Stage 2: Runtime — slim image with only runtime dependencies
# =============================================================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
LABEL org.opencontainers.image.source=https://github.com/amortized-ai/amortized

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg2 \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY containers/shared/ /app/shared/

WORKDIR /app

CMD ["python3.11", "shared/training_runner.py"]
