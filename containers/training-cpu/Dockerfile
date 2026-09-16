# Training container — CPU-only (tiny models, smoke tests, low-cost experimentation)
# Supports the CPU-safe algorithms: sft, osft, lora_sft (via trl).
# GRPO/lora_grpo/GEPA/QLoRA are rejected by validation for CPU (vLLM/bitsandbytes are CUDA-only).
#
# Intended image: {settings.image_registry}/training-cpu:{settings.training_cpu_image_tag}
# (composed in src/amortized/jobs/training.py; registry/tag come from settings — defaults:
# ghcr.io/amortized-ai/training-cpu:latest). The Dockerfile itself does not encode
# the registry or tag.
#
# Deliberately EXCLUDED (CUDA-only or transformers-capping): vllm, flash-attn,
# causal-conv1d, unsloth, xformers, flash-linear-attention, liger, bitsandbytes.
FROM docker.io/library/ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg2 curl \
        python3-dev python3-venv python3-pip \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install the CPU-only torch wheel FIRST (~184MB) from the pytorch CPU index so
# training-hub's torch>=2.6 dependency is already satisfied. torchvision is
# installed from the same CPU index — letting timm (a training-hub dep) pull it
# from PyPI yields a CUDA-built binary that fails to load against CPU torch
# ("operator torchvision::nms does not exist"). Do NOT run plain
# `pip install torch` from PyPI — it silently pulls the nvidia-* CUDA packages.
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# training-hub requires wheel>=0.43 but ubuntu's debian-managed wheel is 0.42.0
# (no RECORD file — pip cannot uninstall it). Shadow it with a pip-installed copy
# in /usr/local. NOTE: we deliberately do NOT use --ignore-installed on the
# training-hub install below — that flag makes pip ignore the CPU torch wheel
# above and silently re-resolve torch from PyPI, pulling the nvidia-* CUDA stack.
RUN pip install --no-cache-dir --ignore-installed --break-system-packages "wheel>=0.43"

# Install training-hub BASE only (no [cuda], no [grpo], no lora/unsloth extras) —
# same thub pin as the CUDA Dockerfile. The CPU torch wheel installed above
# satisfies training-hub's torch>=2.6 requirement (verified: "Requirement already
# satisfied ... 2.14.0+cpu" — no re-download from PyPI).
RUN pip install --no-cache-dir --break-system-packages \
    "training-hub @ git+https://github.com/amortized-ai/training_hub.git@main"

# Install lora deps manually — trl directly for lora_sft, mirroring the CUDA
# Dockerfile's trl-direct workaround, but WITHOUT unsloth (caps transformers<5)
# and WITHOUT xformers (CUDA-only).
RUN pip install --no-cache-dir --break-system-packages "trl>=0.18.0"

# Ensure transformers 5.x for Qwen3.5 support (same pin as the CUDA Dockerfile)
RUN pip install --no-cache-dir --upgrade --break-system-packages "transformers>=5.0"

# S3/MinIO support for reading datasets and writing artifacts (same set as the CUDA
# Dockerfile). --ignore-installed is safe HERE because nothing in this dependency
# graph (s3fs/fsspec/mlflow/boto3) depends on torch — it sidesteps ubuntu's
# debian-managed blinker/cryptography packages, which have no RECORD file and
# cannot be uninstalled by pip. (On the training-hub step above, by contrast,
# --ignore-installed would make pip re-resolve torch from PyPI and pull the
# nvidia-* CUDA stack — do not add it there.)
RUN pip install --no-cache-dir --ignore-installed --break-system-packages s3fs fsspec mlflow boto3

# The CPU runner the builder dispatches (src/amortized/jobs/training.py CPU branch).
# It lives at /usr/local/bin — NOT under /amortized — because the K8s job mounts a
# read-only ConfigMap at /amortized that would shadow any image file placed there.
COPY cpu_sft.py /usr/local/bin/cpu_sft.py

# Build-time smoke checks — every supported CPU algorithm path must import cleanly.
# These FAIL the build on error (no `|| echo deferred` fallbacks): CPU problems must
# surface in CI, not in a user's first job.
RUN python3 -c "\
import torch; \
assert not torch.cuda.is_available(), 'CPU image must not expose CUDA'; \
import transformers, trl, accelerate, peft; \
import training_hub; \
print(f'torch {torch.__version__} (cuda_available={torch.cuda.is_available()})'); \
print(f'transformers {transformers.__version__}'); \
print(f'trl {trl.__version__}'); \
print(f'accelerate {accelerate.__version__}'); \
print(f'peft {peft.__version__}'); \
" && thub --help > /dev/null && python3 -m py_compile /usr/local/bin/cpu_sft.py && echo "thub CLI + cpu_sft.py OK"

FROM docker.io/library/ubuntu:24.04
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
# The dispatched CPU runner — same path the builder emits in its command.
COPY cpu_sft.py /usr/local/bin/cpu_sft.py

WORKDIR /app
CMD ["python3"]
