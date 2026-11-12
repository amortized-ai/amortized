"""Estimate a training job's VRAM with training-hub's memory estimator.

Runs inside the training image (which ships training_hub) so the estimate comes from
training-hub itself. Takes the request as a JSON argument, prints one JSON object as the
last line of stdout:

    {"training_hub_version": "...", "num_params": N,
     "single_gpu": [low, expected, high], "per_gpu": {"1": expected, "2": ...}}

All values are bytes. The flags passed to estimate() match how amortized runs each method:
sft/osft use mini_trainer FSDP (count_fsdp_gather), osft runs without Liger kernels, and
multi-GPU lora/qlora use model splitting (device_map="balanced"), for which training-hub
returns the busiest GPU's memory.

Usage: python3 estimate_vram.py '<request json>'
"""

from __future__ import annotations

import json
import sys
import warnings
from importlib.metadata import version

from training_hub.profiling import memory_estimator as me

warnings.filterwarnings("ignore")

# training-hub caches each model's sizes in a CSV. Its default location may not be
# writable in the job container, so keep the cache in /tmp.
me.MODEL_STORAGE_PATH = "/tmp/training_hub_model_storage.csv"


def main(request: str) -> None:
    req = json.loads(request)
    method = req["method"]
    kwargs = {
        "training_method": method,
        "model_path": req["model"],
        "batch_size": req["batch_size"],
        "max_seq_len": req["max_seq_len"],
        "verbose": 0,
    }
    if method in ("sft", "osft"):
        kwargs["max_tokens_per_gpu"] = req["max_tokens_per_gpu"]
        kwargs["count_fsdp_gather"] = True
    if method == "osft":
        kwargs["use_liger"] = False
        kwargs["unfreeze_rank_ratio"] = req["unfreeze_rank_ratio"]
    if method in ("lora", "qlora"):
        kwargs["lora_r"] = req["lora_r"]
        kwargs["enable_model_splitting"] = True

    single_gpu = me.estimate(num_gpus=1, **kwargs)
    # The estimator builds the model on the meta device from config.json (no weight
    # download) and counts the lm_head of untied models. This reuses the cached sizes.
    num_params = me.BasicEstimator(
        num_gpus=1,
        model_path=req["model"],
        batch_size=req["batch_size"],
        max_seq_len=req["max_seq_len"],
        verbose=0,
    ).num_params
    per_gpu = {
        str(n): single_gpu[1] if n == 1 else me.estimate(num_gpus=n, **kwargs)[1]
        for n in req["gpu_counts"]
    }
    print(
        json.dumps(
            {
                "training_hub_version": version("training-hub"),
                "num_params": int(num_params),
                "single_gpu": [int(x) for x in single_gpu],
                "per_gpu": {k: int(v) for k, v in per_gpu.items()},
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
