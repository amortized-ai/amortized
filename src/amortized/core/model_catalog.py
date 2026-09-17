"""Direct-provider model catalog — stopgap for the absent MLflow AI Gateway.

This MLflow distribution has no AI Gateway, so instead of routing through it we use
providers directly via dropped-in API keys. Data-designer ships a builtin provider
catalog whose ``api_key`` is an env-var *name* resolved at runtime; we surface the
providers whose key is set on the server. The same catalog drives two things, so
they always agree:

- ``list_models`` (what Morty can pick), via :func:`enabled_models`.
- the ``model_providers.yaml`` written into each SDG job pod (so data-designer can
  actually reach the provider), via :func:`enabled_provider_defs`.

The job images' builtin default provider is ``gateway`` (pointing at a bundled MLflow
gateway that does not exist here), which is why the provider file must be supplied.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import amortized.config as _config_mod

logger = logging.getLogger("amortized.core.model_catalog")

_PROVIDER_FIELDS = ("name", "endpoint", "provider_type", "api_key")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _supported_models_path() -> Path:
    """Locate supported_models.json — settings override first (needed when the
    package is installed into site-packages, e.g. the Docker image), repo
    layout otherwise. Mirrors the recipes_dir override pattern."""
    base = _config_mod.settings.supported_models_dir
    if base is None:
        base = _REPO_ROOT
    return base / "agents" / "training" / "skills" / "supported_models.json"


_supported_training_models: list[dict[str, Any]] | None = None


def _key_available(api_key: str | None) -> bool:
    """A provider is usable if its api_key is a literal value, or an env-var name
    (UPPER_SNAKE) set to a non-empty value on the server."""
    if not api_key:
        return False
    if api_key.isupper() and "_" in api_key:
        return bool(os.environ.get(api_key))
    return True


def enabled_provider_defs() -> list[dict[str, str]]:
    """Builtin data-designer providers whose API key is configured on the server.

    Returns provider dicts (name/endpoint/provider_type/api_key) ready to serialize
    into a data-designer ``model_providers.yaml``. ``api_key`` stays the env-var name
    so the job pod resolves the forwarded key at runtime.
    """
    try:
        from data_designer.config.utils.constants import PREDEFINED_PROVIDERS
    except Exception:
        logger.warning("data-designer provider catalog unavailable", exc_info=True)
        return []

    defs: list[dict[str, str]] = []
    for provider in PREDEFINED_PROVIDERS:
        if _key_available(provider.get("api_key")):
            defs.append({k: provider[k] for k in _PROVIDER_FIELDS if k in provider})
    return defs


def enabled_models() -> list[tuple[str, str]]:
    """``(provider, model_id)`` pairs for enabled providers, excluding embeddings.

    SDG teacher models are chat models, so embedding aliases are dropped. Deduped on
    ``(provider, model_id)`` since a provider maps several aliases to one model.
    """
    try:
        from data_designer.config.utils.constants import PREDEFINED_PROVIDERS_MODEL_MAP
    except Exception:
        logger.warning("data-designer model catalog unavailable", exc_info=True)
        return []

    enabled = {d["name"] for d in enabled_provider_defs()}
    models: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for provider, alias_map in PREDEFINED_PROVIDERS_MODEL_MAP.items():
        if provider not in enabled:
            continue
        for alias, model_settings in alias_map.items():
            if alias == "embedding":
                continue
            model_id = model_settings["model"]
            if (provider, model_id) in seen:
                continue
            seen.add((provider, model_id))
            models.append((provider, model_id))
    return models


def training_model_cpu_compatibility(model_name_or_path: str) -> str | None:
    """CPU-compatibility verdict for a training model from the supported-models catalog.

    Returns ``"ok"``, ``"warn"`` (very slow — expect hours), ``"reject"`` (too
    large for CPU), or ``None`` for models not in the catalog. Single source of
    truth for the per-model CPU flags — validation and agent guidance both
    derive from it.
    """
    entry = _find_training_model(model_name_or_path)
    return entry.get("cpu_compatible") if entry else None


# CPU RAM (GiB) per training-model size — single source for the builder's
# memory requests (0.8B→8, 2B→16, 4B→32 for the warned 4B ceiling).
_CPU_MEMORY_GB_BY_SIZE: dict[str, int] = {"0.8B": 8, "2B": 16, "4B": 32}
_CPU_MEMORY_GB_DEFAULT = 8


def training_model_cpu_memory_gb(model_name_or_path: str) -> int:
    """CPU RAM request (GiB) for a training model, from the supported-models catalog.

    Unknown models get the conservative default; validation already warns that
    their CPU compatibility is unknown.
    """
    entry = _find_training_model(model_name_or_path)
    if entry is None:
        return _CPU_MEMORY_GB_DEFAULT
    return _CPU_MEMORY_GB_BY_SIZE.get(entry.get("size", ""), _CPU_MEMORY_GB_DEFAULT)


def _find_training_model(model_name_or_path: str) -> dict[str, Any] | None:
    """Catalog entry for a training model, or ``None`` if not in the catalog."""
    global _supported_training_models
    if _supported_training_models is None:
        try:
            _supported_training_models = json.loads(_supported_models_path().read_text())
        except (OSError, ValueError):
            logger.warning("supported training models catalog unavailable", exc_info=True)
            _supported_training_models = []
    for entry in _supported_training_models:
        size = entry.get("size")
        if model_name_or_path in (entry.get("id"), size) or (
            size and model_name_or_path.endswith(f"-{size}")
        ):
            return entry
    return None
