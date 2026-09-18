# Eval container — evaluates a model on a dataset, serving the model itself.
#
# Serve-only stack: run_eval.py talks HTTP to the model endpoint, and the
# vLLM server is launched via serve_vllm.py (delivered as a job config
# file, not baked into the image) which delegates to the vLLM CLI. torch
# wheels bundle their own CUDA runtime libs, so a plain python base works
# and no nvidia/cuda base (or training stack) is needed — 9.2 GB vs the
# 18.8 GB the old training-based image weighed. External-endpoint evals
# run the same image CPU-only.
# httpx and pyarrow are the eval client's deps; mlflow + boto3 cover the
# job script's dataset download / result upload; versions pinned to match
# the training image so evals behave identically.
FROM python:3.12-slim

# nvidia/cu13 (pulled in by nvidia-cutlass-dsl-libs-cu13) ships nvcc +
# CUDA headers in site-packages; flashinfer runtime JIT needs them, so
# expose them as CUDA_HOME / on PATH instead of a full CUDA toolkit.
# NVCC_PREPEND_FLAGS: the cu13 wheel pairs an nvcc 13.4 binary with CUDA
# 13.0 headers; flashinfer's bundled CCCL rejects that mix unless the
# compatibility check is disabled (newer compiler + older toolkit is a
# supported combination).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13 \
    PATH=/usr/local/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
    NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK

# gcc/g++ + libc6-dev: triton/inductor and flashinfer JIT-compile kernels
# at serve time (a CUDA devel base provided these implicitly before).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# vllm pulls torch, triton, and the nvidia runtime wheels it needs.
# flash-linear-attention: Qwen3.5 hybrid attention kernels (pure triton).
# peft: merging lora_sft adapters into the base model before serving.
# mlflow: the job script shells out to the mlflow CLI for dataset
# download / result upload (S3/MinIO artifacts → boto3).
RUN pip install --no-cache-dir \
        "vllm==0.26.0" \
        "transformers==5.15.0" \
        "flash-linear-attention==0.5.2" \
        "peft==0.20.0" \
        "httpx" \
        "pyarrow" \
        "mlflow==3.15.1" \
        "boto3"

# flashinfer's JIT linker expects the toolkit lib64 layout with the
# unversioned libcudart.so dev name; the pip cu13 tree ships lib/ with
# only the versioned soname — bridge both.
RUN cu13=/usr/local/lib/python3.12/site-packages/nvidia/cu13 \
    && ln -s lib "$cu13/lib64" \
    && ln -s libcudart.so.13 "$cu13/lib/libcudart.so"

COPY run_eval.py /app/run_eval.py

# Run as a non-root user (review hardening). /amortized/work is an
# emptyDir volume the job mounts; the job spec's fsGroup (or the
# platform's default) must allow this uid to write there — kind and
# OpenShift both do. HOME must be writable for vllm/flashinfer JIT
# caches, so point it at the work dir rather than /root. The USER must
# be NUMERIC: kubelet cannot verify runAsNonRoot against a named user.
RUN useradd --create-home --uid 1000 evaluser
ENV HOME=/amortized/work
USER 1000

WORKDIR /amortized/work

ENTRYPOINT ["python3", "/app/run_eval.py"]
