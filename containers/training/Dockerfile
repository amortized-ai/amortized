# Training container — universal (all algorithms, CUDA-enabled)
# Supports: lora_sft, lora_grpo (verl), sft, osft, dpo, kto, gkd, gepa, embedding_sft, classifier_sft
#
# Built from the official training-hub v0.12.0 release (PyPI) — ships the CLI (#103),
# parquet/CSV support (#151), the embedding-sft subcommand (#152), the embedding_sft
# parquet int-label fix (#154), sft parquet/CSV support via the shared
# ensure_messages_jsonl conversion (#155), and the classifier_sft decision/classification-head
# algorithm + `thub classifier-sft` subcommand (#160). Pinned (not @main) for reproducible builds.
# The [embedding] extra pulls sentence-transformers, which both `thub embedding-sft`
# training and the classification eval (classify_eval.py) import at runtime.
#
# The torch/CUDA stack is pinned to a reproducible, CUDA-13-aligned set (torch 2.11.0+cu130,
# the versions vLLM 0.24.0 pins) via a pip constraints file + the cu130 extra index applied to
# EVERY pip call below (see PIP_CONSTRAINT / PIP_EXTRA_INDEX_URL). Without this, `torch` is
# unbounded (training-hub needs only >=2.6.0) and drifts to the newest PyPI wheel — currently
# 2.14.x, whose c10 headers require C++20 (std::strong_ordering), which breaks the flash-attn /
# mamba-ssm / causal-conv1d source compiles (2.11 is a C++17 line, so they build as-is).
#
# vLLM / torch / verl / trl match verl 0.9.1's own tested [vllm] stack (vLLM 0.24, torch 2.11;
# transformers stays 5.17.0): verl's FSDP->vLLM weight sync is tied to vLLM internals, and
# on vLLM 0.32 it broke (removed FusedMoE import; Qwen3.5 tied lm_head/embed_tokens rejected by
# 0.32's _check_skipped_aliases). Eval serving uses the separate eval image, not this vLLM.
FROM docker.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg2 curl \
        python3-dev python3-venv python3-pip \
        build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Pin the torch/CUDA stack for every pip call in this stage: the constraints file forces the
# exact CUDA-13-aligned wheels (so nothing — not even --ignore-installed / --force-reinstall —
# drifts torch off 2.11.0), and the extra index makes the +cu130 local builds resolvable.
RUN printf 'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\ntorchaudio==2.11.0+cu130\n' \
    > /build/constraints.txt
ENV PIP_CONSTRAINT=/build/constraints.txt
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130

# Install training-hub base + grpo + gepa + embedding (pulls the pinned torch, transformers,
# sentence-transformers, etc.)
# Skip lora extra — it pulls unsloth which caps transformers<5
# --ignore-installed needed to bypass distutils system packages
RUN pip install --no-cache-dir --ignore-installed --break-system-packages \
    "training-hub[grpo,gepa,embedding]==0.12.0"

# Install CUDA extensions (flash-attn, mamba-ssm, causal-conv1d) — torch is already installed above
RUN pip install --no-cache-dir --no-build-isolation --break-system-packages \
    "training-hub[grpo,gepa,cuda,embedding]==0.12.0"

# Reinstall training-hub cleanly (first install with --ignore-installed may be partial)
RUN pip install --no-cache-dir --force-reinstall --no-deps --break-system-packages \
    "training-hub[grpo,gepa,embedding]==0.12.0"

# Extend the constraints file for the HF/NLP stack. vLLM below installs with --ignore-installed,
# which re-resolves transformers/huggingface-hub against vLLM's own (looser) ranges; without these
# constraints it can pull a newer transformers → huggingface-hub 2.x → breaks sentence-transformers
# (needs hub<2.0.0). Appended here (after the cached CUDA-extensions compile) so it doesn't bust
# that layer.
# peft is pinned to 0.20.0: with 0.21.x, get_peft_model_state_dict() on an FSDP-wrapped PeftModel
# (inside summon_full_params) returns no tensors, so verl syncs an empty LoRA adapter to vLLM and
# rollouts crash in add_lora (StopIteration). 0.20.0 returns them (and satisfies unsloth >=0.18).
RUN printf 'transformers==5.17.0\nhuggingface-hub<2.0\npeft==0.20.0\n' >> /build/constraints.txt

# Install lora deps manually — xformers (unsloth's caps don't apply at --no-deps below).
# trl is pinned to 1.12.0: the last release with trl.experimental.ppo's value-head models, which
# verl (lora_grpo) imports whenever trl is installed — trl>=1.13 removed them and verl crashes at
# model init. It satisfies unsloth 2026.10.7's trl<=1.15.0 cap, and its GRPOTrainer imports on
# vLLM 0.24 (vllm...nccl_engine.NCCLTrainerSendWeightsArgs), which unsloth patches on import.
RUN pip install --no-cache-dir --break-system-packages "trl==1.12.0" "xformers==0.0.35"

# Install modern unsloth with --no-deps — its pyproject artificially caps `datasets` (and
# transformers<5), but the code works fine with the current datasets>=5 / transformers 5.x,
# and --no-deps keeps unsloth from dragging those caps in.
RUN pip install --no-cache-dir --no-deps --break-system-packages "unsloth==2026.10.7" "unsloth_zoo>=2026.9"

# Qwen3.5 attention kernels: flash-linear-attention (pure triton, no CUDA ext)
RUN pip install --no-cache-dir --break-system-packages "flash-linear-attention[cuda]"

# Pin transformers to 5.17.0 (the training-hub GA lockfile version, also what v0.12.0's
# classifier_sft was validated on): satisfies vLLM 0.24.0 (>=5.5.3), unsloth 2026.10.7 (<=5.17.0)
# and flash-linear-attention (>=5.6.2), and requires huggingface-hub<2.0 — newer transformers allow
# hub 2.x, which breaks sentence-transformers (the embedding-router path). Covers Qwen3.5.
RUN pip install --no-cache-dir --upgrade --break-system-packages "transformers==5.17.0" "peft==0.20.0"

# vLLM (for GRPO online RL) — 0.24.0 hard-pins torch==2.11.0, matched by our constraints file
RUN pip install --no-cache-dir --break-system-packages --ignore-installed "vllm==0.24.0"

# verl + ray for `thub lora-grpo` (training-hub's default GRPO backend; the [grpo-verl] extra).
# verl 0.9.1 declares transformers<5.11, but we pin 5.17.0, so verl (pure Python) is installed
# --no-deps and its runtime deps are added explicitly under the constraints: its base deps plus
# its [verl-core] runtime extras (minus pytest plugins) — the v1 PPO trainer imports
# transfer_queue (TransferQueue) unconditionally.
RUN printf 'vllm==0.24.0\nnumpy>=2\n' >> /build/constraints.txt \
    && pip install --no-cache-dir --break-system-packages --no-deps "verl==0.9.1" \
    && pip install --no-cache-dir --break-system-packages \
       codetiming dill hydra-core pybind11 pylatexenc "ray[default]>=2.41.0" torchdata \
       "tensordict>=0.8.0,<=0.10.0,!=0.9.0" tensorboard wandb \
       "TransferQueue>=0.1.10" pyzmq msgspec fastapi uvicorn mathruler qwen_vl_utils \
       cachetools nvtx

# S3/MinIO support for reading datasets and writing artifacts
RUN pip install --no-cache-dir --break-system-packages s3fs fsspec mlflow boto3

# Rebuild the CUDA extensions from source against the final torch+CUDA ABI (vllm/other installs
# may have swapped wheels). --no-deps keeps the pinned torch; --no-binary forces source compile.
RUN pip install --no-cache-dir --no-build-isolation --no-deps --no-binary causal-conv1d,flash-attn \
    --force-reinstall --break-system-packages "causal-conv1d==1.7.0" "flash-attn==2.8.3.post2"

# Verify key versions — hard gate (build fails if any of these are missing,
# incl. sentence-transformers from the [embedding] extra)
RUN python3 -c "\
import transformers, trl, torch, sentence_transformers, training_hub, huggingface_hub, verl, ray, pyarrow, datasets, transfer_queue; \
import verl.trainer.main_ppo, verl.trainer.ppo.v1, verl.workers.engine_workers, verl.workers.rollout.vllm_rollout; \
from trl.experimental.ppo import AutoModelForCausalLMWithValueHead; \
from importlib.metadata import version; \
thv = version('training-hub'); \
print(f'training-hub {thv}'); \
print(f'transformers {transformers.__version__}'); \
print(f'trl {trl.__version__}'); \
print(f'torch {torch.__version__}'); \
print(f'vllm {version('vllm')}'); \
print(f'verl {verl.__version__}'); \
print(f'sentence-transformers {sentence_transformers.__version__}'); \
print(f'huggingface-hub {huggingface_hub.__version__}'); \
assert thv == '0.12.0', f'expected training-hub 0.12.0, got {thv}'; \
assert torch.__version__.startswith('2.11.0'), f'expected torch 2.11.0, got {torch.__version__}'; \
assert version('vllm') == '0.24.0', f'expected vllm 0.24.0, got {version('vllm')}'; \
assert transformers.__version__ == '5.17.0', f'expected transformers 5.17.0, got {transformers.__version__}'; \
assert version('peft') == '0.20.0', f'expected peft 0.20.0 (0.21 breaks verl LoRA sync under FSDP), got {version('peft')}'; \
assert '+cu13' in torch.__version__, f'expected a cu13 torch build, got {torch.__version__}'; \
assert int(huggingface_hub.__version__.split('.')[0]) < 2, f'sentence-transformers needs huggingface-hub<2.0, got {huggingface_hub.__version__}'; \
assert version('litellm') != '1.107.0', 'litellm 1.107.0 is compromised (credential stealer + backdoor)'; \
"

# Verify the embedding-sft (#152) and classifier-sft (#160) CLI subcommands are present — hard gates
RUN thub embedding-sft --help > /dev/null && echo "thub embedding-sft OK"
RUN thub classifier-sft --help > /dev/null && echo "thub classifier-sft OK"

# Soft check — unsloth may only import on a GPU runtime
RUN python3 -c "import unsloth; print(f'unsloth {unsloth.__version__}')" || echo "unsloth import check deferred to GPU runtime"

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
