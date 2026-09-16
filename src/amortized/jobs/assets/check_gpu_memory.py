"""Fail fast when a model will not fit in the GPU memory available.

vLLM only reports out-of-memory minutes into engine init (after weights
load and KV cache profiling). This pre-flight sizes the model from its
safetensors files (local dir or HF hub id), compares against free GPU
memory (nvidia-smi), and exits non-zero with a clear message when the
weights alone exceed the budget, so the serve job fails in seconds
instead of after a long init.

Usage: python3 check_gpu_memory.py <model_dir_or_hf_id> <gpus> <utilization>
"""

from __future__ import annotations

import subprocess
import sys


def _model_bytes(model_ref: str) -> int | None:
    """Total weight-file bytes for a local dir or HF hub id, if known."""
    import os

    if os.path.isdir(model_ref):
        total = 0
        for root, _dirs, files in os.walk(model_ref):
            for name in files:
                if name.endswith((".safetensors", ".bin", ".pt")):
                    total += os.path.getsize(os.path.join(root, name))
        return total or None
    # HF hub id — ask the hub for file sizes without downloading weights
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_ref, files_metadata=True)
        total = 0
        for sibling in info.siblings or []:
            if sibling.rfilename.endswith((".safetensors", ".bin")):
                total += sibling.size or 0
        return total or None
    except Exception:
        return None


def _gpu_free_bytes() -> list[int] | None:
    """Per-GPU free memory in bytes, via nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:
        return None
    values = []
    for line in out.strip().splitlines():
        try:
            values.append(int(float(line)) * 1024 * 1024)
        except ValueError:
            return None
    return values or None


def main() -> int:
    model_ref = sys.argv[1] if len(sys.argv) > 1 else ""
    gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    utilization = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9

    model_bytes = _model_bytes(model_ref) if model_ref else None
    free = _gpu_free_bytes()
    if model_bytes is None or free is None:
        # Cannot size the model or query GPUs (e.g. no nvidia-smi in a
        # debug pod) — let vLLM's own checks be the arbiter.
        print("check_gpu_memory: skipping (model size or GPU status unknown)")
        return 0

    per_gpu_needed = model_bytes / max(gpus, 1)
    usable = min(free[:gpus]) if len(free) >= gpus else sum(free) / gpus
    budget = usable * utilization
    if per_gpu_needed > budget:
        print(
            f"check_gpu_memory: model needs ~{per_gpu_needed / 1e9:.1f} GB per GPU"
            f" for weights but only ~{budget / 1e9:.1f} GB is available"
            f" (free {usable / 1e9:.1f} GB x {utilization:.0%})"
            " — increase nproc_per_node or pick a smaller model",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_gpu_memory: OK — weights ~{per_gpu_needed / 1e9:.1f} GB/GPU,"
        f" budget ~{budget / 1e9:.1f} GB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
