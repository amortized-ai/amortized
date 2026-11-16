"""Estimate a training job's VRAM with training-hub's memory estimator.

Runs inside the training image (which ships training_hub) so the estimate comes from
training-hub itself. Takes a JSON list of requests (one job for a whole comparison card),
prints one JSON object as the last line of stdout:

    {"training_hub_version": "...", "results": [{"num_params": N,
     "single_gpu": [low, expected, high], "per_gpu": {"1": expected, "2": ...}}
     | {"error": "..."}, ...]}

All values are bytes. If the installed training-hub is too old, prints {"error": "..."}.

The flags passed to estimate() match how amortized runs each method:
sft/osft use mini_trainer FSDP (count_fsdp_gather), osft runs without Liger kernels, and
multi-GPU lora/qlora use model splitting (device_map="balanced"), for which training-hub
returns the busiest GPU's memory, unless the request sets enable_model_splitting false.

Usage: python3 estimate_vram.py '<json list of requests>'
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings
from importlib.metadata import version

from training_hub.profiling import memory_estimator as me

warnings.filterwarnings("ignore")

# training-hub caches each model's sizes in a CSV. Its default location may not be
# writable in the job container, so keep the cache in /tmp.
me.MODEL_STORAGE_PATH = "/tmp/training_hub_model_storage.csv"


_REQUIRED_ARGS = {"count_fsdp_gather", "enable_model_splitting"}


def estimate_one(req: dict) -> dict:
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
    # A lora job that turns splitting off runs on one GPU however many it gets.
    split = req.get("enable_model_splitting", True)
    if method in ("lora", "qlora"):
        kwargs["lora_r"] = req["lora_r"]
        kwargs["enable_model_splitting"] = split

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
        str(n): single_gpu[1]
        if n == 1 or (method in ("lora", "qlora") and not split)
        else me.estimate(num_gpus=n, **kwargs)[1]
        for n in req["gpu_counts"]
    }
    return {
        "num_params": int(num_params),
        "single_gpu": [int(x) for x in single_gpu],
        "per_gpu": {k: int(v) for k, v in per_gpu.items()},
    }


def main(requests: str) -> None:
    hub_version = version("training-hub")
    missing = _REQUIRED_ARGS - inspect.signature(me.estimate).parameters.keys()
    if missing:
        # Older training images would otherwise fail with a TypeError from estimate().
        print(
            json.dumps(
                {
                    "error": f"training-hub {hub_version} in the training image is too "
                    f"old for VRAM estimates (estimate() lacks {', '.join(sorted(missing))})"
                    "; rebuild the training image on a newer training-hub"
                }
            )
        )
        sys.exit(2)
    results = []
    for req in json.loads(requests):
        try:
            results.append(estimate_one(req))
        except Exception as exc:  # e.g. a gated model must not fail the whole card
            results.append({"error": f"{type(exc).__name__}: {exc}"[:500]})
    print(json.dumps({"training_hub_version": hub_version, "results": results}))


if __name__ == "__main__":
    main(sys.argv[1])
