"""Set ``language_model_only: true`` in a GRPO thub config when the model is multimodal.

Multimodal checkpoints used as text models (e.g. Qwen3.5, whose HF config wraps a text
backbone in a vision-language config) need vLLM's ``language_model_only`` for GRPO rollouts.
OSFT doesn't need this flag because mini_trainer detects these models itself
(``vlm_utils.is_vlm_with_causal_lm`` / ``is_vlm_for_direct_loading``); this applies the same
detection to GRPO, where training-hub only forwards an explicit ``language_model_only``.

The job config is mounted read-only, so the result is written to a copy that thub reads. An
explicit ``language_model_only`` (true or false) in the config is kept as-is.

Usage: python3 grpo_language_model_only.py <config_in.yaml> <config_out.yaml>
"""

from __future__ import annotations

import sys
from typing import Any


def is_multimodal(config: Any) -> bool:
    from transformers.models.auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    )

    text_config = getattr(config, "text_config", None)
    text_is_causal_lm = text_config is not None and text_config.__class__ in (
        MODEL_FOR_CAUSAL_LM_MAPPING
    )
    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        # Dual-registered VLMs (e.g. Gemma 3) resolve to a ForConditionalGeneration class.
        resolved = MODEL_FOR_CAUSAL_LM_MAPPING[config.__class__]
        return "ForConditionalGeneration" in resolved.__name__ and text_is_causal_lm
    return text_is_causal_lm or config.__class__ in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING


def patch(config_in: str, config_out: str) -> bool | None:
    import yaml

    with open(config_in) as f:
        thub_config = yaml.safe_load(f) or {}
    if "language_model_only" not in thub_config:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            thub_config["model_path"],
            trust_remote_code=bool(thub_config.get("trust_remote_code")),
            revision=thub_config.get("revision"),
        )
        if is_multimodal(model_config):
            thub_config["language_model_only"] = True
    with open(config_out, "w") as f:
        yaml.safe_dump(thub_config, f, sort_keys=False)
    value = thub_config.get("language_model_only")
    return None if value is None else bool(value)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    result = patch(sys.argv[1], sys.argv[2])
    print(f"grpo_language_model_only: language_model_only={result}")
