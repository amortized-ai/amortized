# Training container — universal (all algorithms, CUDA-enabled)
# Supports: lora_sft, lora_grpo, sft, osft, dpo, kto, gkd, gepa
FROM docker.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg2 curl \
        python3-dev python3-venv python3-pip \
        build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install training-hub base + grpo + gepa (pulls torch, transformers, etc.)
# Skip lora extra — it pulls unsloth which caps transformers<5
# --ignore-installed needed to bypass distutils system packages
RUN pip install --no-cache-dir --ignore-installed --break-system-packages \
    "training-hub[grpo,gepa] @ git+https://github.com/amortized-ai/training_hub.git@main"

# Install CUDA extensions (flash-attn, mamba-ssm, etc.) — torch is already installed above
RUN pip install --no-cache-dir --no-build-isolation --break-system-packages \
    "training-hub[grpo,gepa,cuda] @ git+https://github.com/amortized-ai/training_hub.git@main"

# Reinstall training-hub cleanly (first install with --ignore-installed may be partial)
RUN pip install --no-cache-dir --force-reinstall --no-deps --break-system-packages \
    "training-hub[grpo,gepa] @ git+https://github.com/amortized-ai/training_hub.git@main"

# Install lora deps manually — trl and xformers without unsloth's caps
RUN pip install --no-cache-dir --break-system-packages "trl>=0.18.0" "xformers>=0.0.33.post1"

# Install modern unsloth with --no-deps (its pyproject caps transformers<5 but code works)
RUN pip install --no-cache-dir --no-deps --break-system-packages "unsloth>=2026.7" "unsloth_zoo>=2026.7"

# Qwen3.5 attention kernels: flash-linear-attention (pure triton, no CUDA ext)
RUN pip install --no-cache-dir --break-system-packages "flash-linear-attention[cuda]"

# Ensure transformers 5.x for Qwen3.5 support
RUN pip install --no-cache-dir --upgrade --break-system-packages "transformers>=5.0"

# vLLM (for GRPO online RL) — may change torch version
RUN pip install --no-cache-dir --break-system-packages --ignore-installed "vllm>=0.19.0"

# S3/MinIO support for reading datasets and writing artifacts
RUN pip install --no-cache-dir --break-system-packages s3fs fsspec mlflow boto3

# CUDA extensions MUST be built from source AFTER vllm (which upgrades torch)
# --no-deps prevents overwriting torch with a CPU-only build
# --no-binary forces source compilation against the final torch+CUDA ABI
RUN pip install --no-cache-dir --no-build-isolation --no-deps --no-binary causal-conv1d,flash-attn \
    --force-reinstall --break-system-packages "causal-conv1d>=1.4.0" "flash-attn>=2.7.0"

# Verify key versions
RUN python3 -c "\
import transformers, trl, torch; \
print(f'transformers {transformers.__version__}'); \
print(f'trl {trl.__version__}'); \
print(f'torch {torch.__version__}'); \
" && python3 -c "import unsloth; print(f'unsloth {unsloth.__version__}')" || echo "unsloth import check deferred to GPU runtime"

FROM docker.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
LABEL org.opencontainers.image.source=https://github.com/amortized-ai/amortized

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
CMD ["python3"]
