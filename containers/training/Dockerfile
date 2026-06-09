# Training container — universal (all algorithms, CUDA-enabled)
# Supports: lora_sft, sft, osft, grpo, lora_grpo, gepa
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg2 curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install all algorithm extras following Training Hub's recommended order.
# [grpo] caps transformers<5.0, but [cuda] step can upgrade it via
# instructlab-training — enforce with PIP_CONSTRAINT.
RUN echo 'transformers<5.0' > /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt
RUN python3.11 -m pip install --no-cache-dir "training-hub[grpo,lora,gepa]"
RUN python3.11 -m pip install --no-cache-dir --no-build-isolation "training-hub[cuda]"

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
LABEL org.opencontainers.image.source=https://github.com/amortized-ai/amortized

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg2 curl gcc g++ \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY containers/shared/ /app/shared/

WORKDIR /app

CMD ["python3.11", "shared/training_runner.py"]
