# Training container — universal (all algorithms, CUDA-enabled)
# Supports: lora_sft, lora_grpo, sft, osft, dpo, kto, gkd, gepa
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

# Constraints: pin verl<0.8 (0.8 has broken experimental imports),
# transformers<5.0 (keeps torch at 2.10 which is stable with CUDA 12.6),
# huggingface-hub<1.0 (hub 1.x introduces keras_nlp backend errors)
RUN echo 'transformers<5.0' > /tmp/constraints.txt \
    && echo 'verl<0.8' >> /tmp/constraints.txt \
    && echo 'huggingface-hub<1.0' >> /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

RUN python3.11 -m pip install --no-cache-dir --ignore-installed \
    "training-hub[grpo,lora,gepa] @ git+https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git@main"
RUN python3.11 -m pip install --no-cache-dir --no-build-isolation --ignore-installed \
    "training-hub[grpo,lora,gepa,cuda] @ git+https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git@main"

# vLLM 0.19 for GRPO rollout generation (compatible with torch 2.10)
RUN python3.11 -m pip install --no-cache-dir --ignore-installed "vllm>=0.19,<0.20"

# Fix numpy: vLLM may leave numpy 2.x .so files alongside 1.x metadata.
# Force-reinstall to ensure consistency.
RUN python3.11 -m pip install --no-cache-dir --force-reinstall "numpy<2"

# Disable Ray's uv_runtime_env_hook (crashes on pyproject.toml/working_dir detection)
# Patch: insert 'return runtime_env' as first line of the hook function body
RUN python3.11 -c "import re; path='/usr/local/lib/python3.11/dist-packages/ray/_private/runtime_env/uv_runtime_env_hook.py'; code=open(path).read(); code=re.sub(r'(def hook\([^)]*\)[^:]*:)\n', r'\1\n    return runtime_env  # disabled\n', code, count=1); open(path,'w').write(code); print('Patched Ray uv hook')"

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

WORKDIR /app
CMD ["python3.11"]
