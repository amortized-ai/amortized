"""Estimate a training job's VRAM with training-hub's memory estimator.

Runs inside the training image (which ships training_hub) so the estimate comes from
training-hub itself. Takes the request as a JSON argument, prints one JSON object as the
last line of stdout:

    {"training_hub_version": "...", "num_params": N,
     "single_gpu": [low, expected, high], "per_gpu": {"1": expected, "2": ...}}

All values are bytes. The flags passed to estimate() match how amortized runs each method:
sft/osft use mini_trainer FSDP (count_fsdp_gather), osft runs without Liger kernels, and
multi-GPU lora/qlora use model splitting (device_map="balanced").

Usage: python3 estimate_vram.py '<request json>'
"""

from __future__ import annotations

import json
import sys
import warnings
from importlib.metadata import version

import torch
from training_hub.profiling import memory_estimator as me
from transformers import AutoConfig, AutoModel

warnings.filterwarnings("ignore")

_num_params: list[int] = []


def _load_model_from_config(self, warn: str, trust_remote_code: bool = False) -> None:
    """training-hub's _load_model downloads the full weights for any model missing from
    its CSV cache. Build the same AutoModel on the meta device from config.json instead
    (a few KB), which gives identical parameter shapes and counts."""
    config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        self.model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
    self.num_params = self.model.num_parameters(only_trainable=False)
    self.num_trainable_params = self.model.num_parameters(only_trainable=True)
    text_config = config.get_text_config()
    self.num_layers = text_config.num_hidden_layers
    self.hidden_size = text_config.hidden_size
    _num_params.append(self.num_params)


me.BasicEstimator._load_model = _load_model_from_config
# The CSV cache lives in site-packages, which may not be writable in the job container.
me.BasicEstimator._update_model_storage = lambda self: None


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
    per_gpu = {
        str(n): single_gpu[1] if n == 1 else me.estimate(num_gpus=n, **kwargs)[1]
        for n in req["gpu_counts"]
    }
    print(
        json.dumps(
            {
                "training_hub_version": version("training-hub"),
                "num_params": _num_params[0],
                "single_gpu": [int(x) for x in single_gpu],
                "per_gpu": {k: int(v) for k, v in per_gpu.items()},
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
