"""Resolve a model's architecture for VRAM estimation.

Lookup order for ``resolve_arch``:
  1. ``KNOWN_ARCHS`` — the verified training models, so estimation works offline.
  2. HuggingFace — ``config.json`` plus the exact parameter count from the hub's
     safetensors metadata (no weights are downloaded).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from typing import Any

import httpx

logger = logging.getLogger("amortized.core.model_arch")


@dataclass(frozen=True)
class ModelArch:
    # The fields training-hub's estimator reads (its model_storage.csv columns) ...
    label: str
    num_params: int
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    weight_size_total: int | None = None  # LoRA target (in + out) sum; None = derive
    osft_params: int | None = None  # bytes for the SVD-decomposed model; None = derive
    tie_word_embeddings: bool = False  # lm_head shares the embedding weights
    # ... plus the config shape used to derive the two above when they are not known.
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_experts: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    source: str = "catalog"


def _label(num_params: int) -> str:
    return f"{num_params / 1e9:.1f}B"


def _first(src: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = src.get(key)
        if value is not None:
            return value
    return None


def arch_from_hf_config(
    config: dict[str, Any], num_params: int | None, *, label: str = "", source: str = "huggingface"
) -> ModelArch:
    """Normalize a HuggingFace ``config.json`` (dense, MoE, or multimodal) into a ModelArch."""
    # Multimodal models (Qwen3.5, Gemma 4, ...) nest the language model under text_config.
    src = config.get("text_config") or config.get("llm_config") or config
    layers = _first(src, "num_hidden_layers", "num_layers", "n_layer")
    hidden = _first(src, "hidden_size", "d_model", "n_embd")
    vocab = _first(src, "vocab_size") or config.get("vocab_size")
    if not (layers and hidden and vocab):
        raise ValueError("config.json is missing num_hidden_layers, hidden_size or vocab_size")

    heads = _first(src, "num_attention_heads", "n_head") or 1
    kv_heads = _first(src, "num_key_value_heads") or heads
    head_dim = _first(src, "head_dim") or hidden // heads
    intermediate = _first(src, "intermediate_size", "ffn_hidden_size") or hidden * 4
    experts = _first(src, "num_experts", "num_local_experts", "n_routed_experts") or 0
    moe_inter = _first(src, "moe_intermediate_size") or (intermediate if experts else 0)
    shared = _first(src, "shared_expert_intermediate_size") or (
        (_first(src, "n_shared_experts") or 0) * moe_inter
    )

    arch = ModelArch(
        label=label,
        num_params=0,
        num_hidden_layers=int(layers),
        hidden_size=int(hidden),
        intermediate_size=int(intermediate),
        vocab_size=int(vocab),
        num_attention_heads=int(heads),
        num_kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        num_experts=int(experts),
        moe_intermediate_size=int(moe_inter),
        shared_expert_intermediate_size=int(shared),
        source=source,
    )
    # training-hub counts AutoModel's parameters, which exclude the lm_head, so drop it from
    # the hub's safetensors total when it is stored. Some tied text models (Qwen3-0.6B) still
    # store it, so for those compare the total with the config shape; multimodal totals also
    # include a vision tower, which defeats that comparison, so trust tie_word_embeddings.
    tied = bool(_first(src, "tie_word_embeddings") or config.get("tie_word_embeddings"))
    approx = approx_num_params(arch)
    params = approx
    if num_params:
        head = arch.vocab_size * arch.hidden_size
        if src is not config:
            has_head = not tied
        else:
            has_head = abs(num_params - head - approx) < abs(num_params - approx)
        params = num_params - head if has_head else num_params
    return replace(
        arch,
        num_params=int(params),
        label=label or _label(int(params)),
        tie_word_embeddings=tied,
    )


def approx_num_params(arch: ModelArch) -> int:
    """AutoModel parameter count (embeddings + decoder layers, no lm_head) from the config
    shape, for when the hub has no safetensors metadata."""
    h, hd = arch.hidden_size, arch.head_dim
    attn = 2 * h * arch.num_attention_heads * hd + 2 * h * arch.num_kv_heads * hd
    if arch.num_experts:
        mlp = arch.num_experts * 3 * h * arch.moe_intermediate_size
        mlp += 3 * h * arch.shared_expert_intermediate_size
    else:
        mlp = 3 * h * arch.intermediate_size
    return int(arch.vocab_size * h + arch.num_hidden_layers * (attn + mlp))


def _known(
    num_params: int,
    num_hidden_layers: int,
    hidden_size: int,
    vocab_size: int,
    weight_size_total: int,
    osft_params: int,
    tie_word_embeddings: bool,
) -> ModelArch:
    return ModelArch(
        label=_label(num_params),
        num_params=num_params,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        weight_size_total=weight_size_total,
        osft_params=osft_params,
        tie_word_embeddings=tie_word_embeddings,
        source="catalog",
    )


# training-hub's own model statistics (the columns of its model_storage.csv: num_params,
# num_layers, hidden_size, vocab_size, weight_size_total, osft_params) for every model in
# agents/training/skills/knowledge-ingestion/osft/supported_models.json, computed by running
# training-hub's memory_estimator.py on each model. Keep the two files in sync.
KNOWN_ARCHS: dict[str, ModelArch] = {
    "qwen/qwen3-0.6b": _known(596_049_920, 28, 1024, 151936, 630_784, 3_207_086_080, True),
    "qwen/qwen3-1.7b": _known(1_720_574_976, 28, 2048, 151936, 1_089_536, 9_467_367_424, True),
    "qwen/qwen3-4b": _known(4_022_468_096, 36, 2560, 151936, 2_064_384, 21_112_592_384, True),
    "qwen/qwen3-8b": _known(7_568_405_504, 36, 4096, 151936, 2_727_936, 42_658_451_456, False),
    "qwen/qwen3-14b": _known(13_990_394_880, 40, 5120, 151936, 4_014_080, 77_273_067_520, False),
    "qwen/qwen3-32b": _known(31_984_210_944, 64, 5120, 151936, 8_388_608, 162_035_224_576, False),
    "qwen/qwen3.5-0.8b": _known(852_985_920, 24, 1024, 248320, 399_360, 3_777_216_768, True),
    "qwen/qwen3.5-2b": _known(2_213_241_664, 24, 2048, 248320, 681_984, 10_275_548_416, True),
    "qwen/qwen3.5-4b": _known(4_539_265_536, 32, 2560, 248320, 1_327_104, 21_161_396_224, True),
    "qwen/qwen3.5-9b": _known(8_392_695_024, 32, 4096, 248320, 1_818_624, 41_155_982_272, False),
    "qwen/qwen3.8-27b": _known(26_085_330_160, 64, 5120, 248320, 4_980_736, 127_968_359_360, False),
    "google/gemma-4-26b-a4b-it": _known(
        25_805_933_872, 30, 2816, 262144, 1_852_368, 108_598_566_200, True
    ),
    "google/gemma-4-31b-it": _known(
        31_273_086_512, 60, 5376, 262144, 8_342_608, 167_664_998_320, True
    ),
    "deepseek-ai/deepseek-r1-distill-qwen-1.5b": _known(
        1_543_714_304, 28, 1536, 151936, 1_154_048, 7_511_660_544, False
    ),
    "deepseek-ai/deepseek-r1-distill-qwen-7b": _known(
        7_070_619_136, 28, 3584, 152064, 2_523_136, 35_536_549_888, False
    ),
    "nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16": _known(
        31_225_615_808, 52, 2688, 131072, 0, 124_902_475_008, False
    ),
}

_HF_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def fetch_hf_arch(model_id: str) -> ModelArch | None:
    """Fetch config.json + parameter count from the hub. None if unreachable or unparseable."""
    if not _HF_ID_RE.match(model_id):
        return None
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    arch: ModelArch | None = None
    try:
        # config.json redirects to a cache URL; httpx drops Authorization on cross-origin hops.
        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True, headers=_hf_headers()
        ) as client:
            resp = await client.get(f"{endpoint}/{model_id}/resolve/main/config.json")
            resp.raise_for_status()
            config = resp.json()
            if not isinstance(config, dict):
                raise ValueError("config.json is not a JSON object")
            num_params: int | None = None
            try:
                meta = await client.get(
                    f"{endpoint}/api/models/{model_id}", params={"expand[]": "safetensors"}
                )
                meta.raise_for_status()
                info = meta.json()
                safetensors = info.get("safetensors") if isinstance(info, dict) else None
                if isinstance(safetensors, dict):
                    num_params = safetensors.get("total")
            except (httpx.HTTPError, ValueError):
                logger.info("no safetensors metadata for %s; approximating params", model_id)
        arch = arch_from_hf_config(config, num_params)
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.warning("could not fetch HuggingFace config for %s: %s", model_id, exc)
    return arch


async def resolve_arch(model_name_or_path: str) -> ModelArch:
    """Architecture for a HuggingFace model id. Raises ValueError if it cannot be resolved."""
    model_id = model_name_or_path.strip()
    known = KNOWN_ARCHS.get(model_id.lower())
    if known:
        return known
    fetched = await fetch_hf_arch(model_id)
    if fetched:
        return fetched
    raise ValueError(
        f"Could not read the config of model {model_id!r} from HuggingFace. Check the model "
        "id (org/name); for a gated model, set HF_TOKEN on the server."
    )
