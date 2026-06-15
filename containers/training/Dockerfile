# Training container — universal (all algorithms, CUDA-enabled)
# Supports: lora_sft, sft, osft, dpo, kto, gkd, gepa
# Includes vLLM for future GRPO support (lora_grpo requires verl fixes — see issue #XXX)
FROM docker.io/nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

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
RUN echo 'transformers<5.0' > /tmp/constraints.txt && echo 'verl<0.8' >> /tmp/constraints.txt && echo 'prometheus-fastapi-instrumentator<8' >> /tmp/constraints.txt && echo 'apache-tvm-ffi<=0.1.9' >> /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt
RUN python3.11 -m pip install --no-cache-dir --ignore-installed "training-hub[grpo,lora,gepa] @ git+https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git@main"
RUN python3.11 -m pip install --no-cache-dir --no-build-isolation --ignore-installed "training-hub[grpo,lora,gepa,cuda] @ git+https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git@main"
RUN python3.11 -m pip install --no-cache-dir --ignore-installed "vllm>=0.19,<0.20"

# Ensure numpy is consistent after all installs (vLLM may upgrade, unsloth needs <2)
RUN python3.11 -m pip install --no-cache-dir --force-reinstall "numpy<2" \
    && python3.11 -c "import numpy; print(f'numpy: {numpy.__version__}')"

FROM docker.io/nvidia/cuda:12.6.0-runtime-ubuntu22.04
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
