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

import logging
import os

logger = logging.getLogger("amortized.core.model_catalog")

_PROVIDER_FIELDS = ("name", "endpoint", "provider_type", "api_key")


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


def resolve_provider(name: str) -> dict[str, str] | None:
    """Return the enabled provider def whose ``name`` matches (case-insensitive),
    else None. Lets an endpoint reference a provider by name (``base_url: "openai"``)
    and get its real endpoint URL + api-key env-var name."""
    lowered = (name or "").strip().lower()
    if not lowered:
        return None
    for pdef in enabled_provider_defs():
        if pdef.get("name", "").lower() == lowered:
            return pdef
    return None


def inject_enabled_provider_keys(env: dict[str, str]) -> None:
    """Copy every enabled provider's API key from the server env into a job's ``env``
    dict (delivered to the pod as a per-job Secret), so the job can reach any
    configured provider. Shared by SDG and eval so both pick up keys the same way."""
    for pdef in enabled_provider_defs():
        key_name = pdef.get("api_key", "")
        if key_name.isupper() and "_" in key_name and key_name in os.environ:
            env[key_name] = os.environ[key_name]


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
