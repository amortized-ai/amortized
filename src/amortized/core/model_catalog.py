"""Direct-provider model catalog — stopgap for the absent MLflow AI Gateway.

This MLflow distribution has no AI Gateway, so instead of routing through it we use
providers directly via dropped-in API keys. Data-designer ships a builtin provider
catalog whose ``api_key`` is an env-var *name* resolved at runtime; we surface the
providers whose key is set on the server, plus a native Anthropic provider, BYOK
MaaS/EnMaaS endpoints and the user's custom endpoints, which the builtin catalog lacks.
That provider set drives two things:

- ``list_models`` — the SDG-teacher + eval-judge choices — via :func:`available_models`
  (pulled live from each provider's ``/v1/models``), falling back to the static
  :func:`enabled_models`. (Morty's chat picker is separate: it reads OpenCode's
  ``/provider``; see :mod:`amortized.api.agent`.)
- the ``model_providers.yaml`` written into each SDG job pod (so data-designer can
  actually reach the provider), via :func:`enabled_provider_defs`.

The job images' builtin default provider is ``gateway`` (pointing at a bundled MLflow
gateway that does not exist here), which is why the provider file must be supplied.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("amortized.core.model_catalog")

_PROVIDER_FIELDS = ("name", "endpoint", "provider_type", "api_key")

# OpenAI's usable teacher/judge families. A plain startswith list (so gpt-5, gpt-5.6-sol,
# gpt-6-astra, … all match); extend here for gpt-7 etc.
_OPENAI_CHAT_PREFIXES = ("gpt-5", "gpt-6")
# Substrings that mark a NON-chat model (embeddings, speech, image, moderation, rerank).
# Used to keep only chat models across every provider — /v1/models has no capability flag.
_NON_CHAT_MARKERS = (
    "embed",
    "whisper",
    "tts",
    "dall-e",
    "dalle",
    "moderation",
    "rerank",
    "stable-diffusion",
    "sdxl",
)
_MODELS_TTL_SECONDS = 300
# Anthropic authenticates with x-api-key + this version header (not OpenAI's Bearer).
_ANTHROPIC_VERSION = "2023-06-01"
# provider name -> (fetched_at_monotonic, raw model ids). Per-provider so one slow/broken
# provider never blocks or drops the others.
_models_cache: dict[str, tuple[float, list[str]]] = {}


def _key_available(api_key: str | None) -> bool:
    """A provider is usable if its api_key is a literal value, or an env-var name
    (UPPER_SNAKE) set to a non-empty value on the server."""
    if not api_key:
        return False
    if api_key.isupper() and "_" in api_key:
        return bool(os.environ.get(api_key))
    return True


def _maas_provider_def() -> dict[str, str] | None:
    """A BYOK Red Hat MaaS endpoint as an OpenAI-compatible provider.

    Unlike the data-designer builtins (hardcoded endpoints), MaaS is user-supplied, so
    its base URL comes from ``MAAS_BASE_URL`` and its key from ``MAAS_API_KEY`` — both
    injected per-user from the user's BYOK secret. Surfaced only when both are set.
    """
    base_url = os.environ.get("MAAS_BASE_URL", "").strip().rstrip("/")
    if not base_url or not os.environ.get("MAAS_API_KEY"):
        return None
    # The server sends the bearer key to this URL, so require HTTPS: an http:// endpoint would
    # transmit the credential in cleartext and allow server-side requests to internal hosts.
    if not base_url.lower().startswith("https://"):
        logger.warning("ignoring non-HTTPS MAAS_BASE_URL (credentials must not cross plaintext)")
        return None
    return {
        "name": "maas",
        "endpoint": base_url,
        "provider_type": "openai",
        "api_key": "MAAS_API_KEY",
    }


def _anthropic_provider_def() -> dict[str, str] | None:
    """A native Anthropic provider. Data Designer ships the anthropic *adapter* but no
    predefined anthropic provider, so we register one when ``ANTHROPIC_API_KEY`` is set —
    unlocking Claude as an SDG teacher + eval judge (not just Morty chat). ``provider_type``
    'anthropic' routes DD's AnthropicClient and the eval runner to the Messages API."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return {
        "name": "anthropic",
        "endpoint": "https://api.anthropic.com/v1",
        "provider_type": "anthropic",
        "api_key": "ANTHROPIC_API_KEY",
    }


def _enmaas_provider_def() -> dict[str, str] | None:
    """Red Hat EnMaaS (OpenAI dialect) as a source distinct from ``maas``/``openai``, so SDG +
    eval bill it separately. Its gpt + hosted models come from the Bearer ``/v1/models``; Claude
    is a separate anthropic-dialect def (:func:`_enmaas_anthropic_provider_def`). Same per-user
    ENMAAS_BASE_URL + ENMAAS_API_KEY as Morty's enmaas provider; surfaced only when both are set."""
    base_url = os.environ.get("ENMAAS_BASE_URL", "").strip().rstrip("/")
    if not base_url or not os.environ.get("ENMAAS_API_KEY"):
        return None
    if not base_url.lower().startswith("https://"):
        logger.warning("ignoring non-HTTPS ENMAAS_BASE_URL (credentials must not cross plaintext)")
        return None
    return {
        "name": "enmaas",
        "endpoint": base_url,
        "provider_type": "openai",
        "api_key": "ENMAAS_API_KEY",
    }


def _enmaas_anthropic_provider_def() -> dict[str, str] | None:
    """EnMaaS Claude via its anthropic dialect (x-api-key), a distinct source for billing. Same
    key + base URL as :func:`_enmaas_provider_def`; ``provider_type`` 'anthropic' routes DD's
    AnthropicClient + the eval runner to the Messages API."""
    base_url = os.environ.get("ENMAAS_BASE_URL", "").strip().rstrip("/")
    if not base_url or not os.environ.get("ENMAAS_API_KEY"):
        return None
    if not base_url.lower().startswith("https://"):
        return None  # _enmaas_provider_def already logged the warning
    return {
        "name": "enmaas-anthropic",
        "endpoint": base_url,
        "provider_type": "anthropic",
        "api_key": "ENMAAS_API_KEY",
    }


# Names a custom endpoint cannot take: the providers above and data-designer's builtins.
_RESERVED_PROVIDER_NAMES = frozenset(
    {
        "nvidia",
        "openai",
        "openrouter",
        "anthropic",
        "maas",
        "enmaas",
        "enmaas-anthropic",
        "enmaas-openai",
        "vertex",
        "gateway",
    }
)
_CUSTOM_ENDPOINT_BASE_URL = re.compile(r"^CUSTOM_ENDPOINT_([A-Z0-9_]+)_BASE_URL$")


def custom_endpoint_url_allowed(url: str) -> bool:
    """HTTPS, or plain HTTP to an in-cluster service (``*.svc``), with no credentials,
    query or fragment in the URL. The API key is sent to this URL."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host or parts.username or parts.password or parts.query or parts.fragment:
        return False
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and host.endswith((".svc", ".svc.cluster.local"))


def _custom_endpoint_defs() -> list[dict[str, str]]:
    """OpenAI-compatible endpoints added under Settings > Custom endpoints (or set by an
    operator). Each ``CUSTOM_ENDPOINT_<NAME>_BASE_URL`` + ``CUSTOM_ENDPOINT_<NAME>_API_KEY``
    pair becomes a provider named ``<name>`` (lowercase, ``_`` -> ``-``)."""
    defs: list[dict[str, str]] = []
    for env_name in sorted(os.environ):
        match = _CUSTOM_ENDPOINT_BASE_URL.match(env_name)
        if not match:
            continue
        slug = match.group(1)
        name = slug.lower().replace("_", "-")
        key_env = f"CUSTOM_ENDPOINT_{slug}_API_KEY"
        base_url = os.environ[env_name].strip().rstrip("/")
        if not base_url or not os.environ.get(key_env):
            continue
        if name in _RESERVED_PROVIDER_NAMES:
            logger.warning("ignoring custom endpoint %r: a built-in provider has that name", name)
            continue
        if not custom_endpoint_url_allowed(base_url):
            logger.warning(
                "ignoring custom endpoint %r: the base URL must be https (http only for"
                " in-cluster *.svc hosts) with no credentials or query",
                name,
            )
            continue
        defs.append(
            {"name": name, "endpoint": base_url, "provider_type": "openai", "api_key": key_env}
        )
    return defs


def enabled_provider_defs() -> list[dict[str, str]]:
    """Providers whose API key is configured on the server: the builtin data-designer
    catalog plus a native Anthropic provider, BYOK MaaS/EnMaaS endpoints and custom
    endpoints (when configured).

    Returns provider dicts (name/endpoint/provider_type/api_key) ready to serialize
    into a data-designer ``model_providers.yaml``. ``api_key`` stays the env-var name
    so the job pod resolves the forwarded key at runtime.
    """
    defs: list[dict[str, str]] = []
    try:
        from data_designer.config.utils.constants import PREDEFINED_PROVIDERS

        for provider in PREDEFINED_PROVIDERS:
            if _key_available(provider.get("api_key")):
                defs.append({k: provider[k] for k in _PROVIDER_FIELDS if k in provider})
    except Exception:
        logger.warning("data-designer provider catalog unavailable", exc_info=True)

    # Inject providers DD's builtin catalog lacks (it ships only openai-type providers): native
    # Anthropic, a BYOK MaaS endpoint, the EnMaaS gateway (its openai + anthropic dialects as
    # two sources) and custom endpoints, each when its key is set. Skip any name a DD builtin
    # already supplied.
    names = {d.get("name") for d in defs}
    for extra in (
        _anthropic_provider_def(),
        _maas_provider_def(),
        _enmaas_provider_def(),
        _enmaas_anthropic_provider_def(),
        *_custom_endpoint_defs(),
    ):
        if extra and extra["name"] not in names:
            defs.append(extra)
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


def _resolve_key(api_key: str) -> str:
    """Resolve a provider's api_key field to an actual key: an env-var name
    (UPPER_SNAKE) is read from the environment; anything else is a literal."""
    if not api_key:
        return ""
    if api_key.isupper() and "_" in api_key:
        return os.environ.get(api_key, "")
    return api_key


def _keep_model(provider: str, model_id: str) -> bool:
    """Whether a provider's model id belongs in the picker: chat models only, and for
    the ``openai`` provider restricted to the gpt-5/gpt-6 families."""
    if not model_id:
        return False
    if any(marker in model_id.lower() for marker in _NON_CHAT_MARKERS):
        return False
    if provider == "openai":
        return any(model_id.startswith(p) for p in _OPENAI_CHAT_PREFIXES)
    # EnMaaS's anthropic dialect also lists the hosted rits/* models (they serve both dialects);
    # keep those only under the openai-dialect `enmaas` source, not duplicated under Claude.
    if provider == "enmaas-anthropic":
        return not model_id.startswith("rits/")
    return True


async def _fetch_provider_models(pdef: dict[str, str]) -> tuple[list[str], bool]:
    """Fetch raw model ids from one provider's OpenAI-compatible ``/models`` endpoint.

    Returns ``(ids, ok)``; ``ok`` is False ONLY when the live fetch actually errored, so callers
    can tell a legitimately-empty result from an unreachable provider. Cached per provider (TTL);
    a failed fetch returns the last good value (or empty) so one provider never breaks the list."""
    name = pdef.get("name", "")
    now = time.monotonic()
    cached = _models_cache.get(name)
    if cached and (now - cached[0]) < _MODELS_TTL_SECONDS:
        return cached[1], True

    endpoint = pdef.get("endpoint", "").rstrip("/")
    key = _resolve_key(pdef.get("api_key", ""))
    if not endpoint or not key:
        return (cached[1] if cached else []), True

    url = f"{endpoint}/models"
    # Anthropic authenticates with x-api-key + a version header (not OpenAI's Bearer); its
    # /v1/models returns the same {"data": [{"id": ...}]} shape, so only the headers differ.
    if pdef.get("provider_type") == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION}
    else:
        headers = {"Authorization": f"Bearer {key}"}
    try:
        # follow_redirects stays off (httpx's default, made explicit): never replay the bearer
        # key to a redirect target the provider URL points at.
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        ids = [str(m.get("id", "")) for m in data if m.get("id")]
        _models_cache[name] = (now, ids)
        return ids, True
    except Exception:
        logger.warning("failed to fetch models for provider %r from %s", name, url, exc_info=True)
        return (cached[1] if cached else []), False


async def accessible_model_ids() -> dict[str, set[str]]:
    """``{provider_name: {raw model ids the configured key can access}}`` — pulled from each enabled
    provider's OWN ``/v1/models`` (not models.dev), so the set reflects real entitlement.

    Used to access-filter the Morty picker: intersect OpenCode's resolvable catalog with this so a
    model the user's key can't actually use never shows. Best-effort + cached per provider (see
    :func:`_fetch_provider_models`); a provider that can't be reached yields an empty set, and
    callers fall back to the unfiltered list rather than hiding everything.
    """
    defs = enabled_provider_defs()
    if not defs:
        return {}
    fetched = await asyncio.gather(*[_fetch_provider_models(d) for d in defs])
    out: dict[str, set[str]] = {}
    for pdef, (ids, _ok) in zip(defs, fetched, strict=True):
        name = pdef.get("name", "")
        if name:
            out[name] = set(ids)
            # Morty exposes the enmaas openai dialect under two opencode ids (enmaas-openai
            # for gpt via Responses, enmaas for rits via chat); mirror the access set so the
            # picker access-filters the gpt (enmaas-openai) models too.
            if name == "enmaas":
                out["enmaas-openai"] = set(ids)
    return out


async def available_models() -> tuple[list[tuple[str, str]], bool]:
    """``((provider, model_id) chat models, live_ok)`` pulled live from each key-enabled provider's
    ``/v1/models`` — the dynamic replacement for the static data-designer catalog.

    ``live_ok`` is False when any provider's live fetch errored, so callers can distinguish a
    legitimately-empty result (e.g. a key with no gpt-5/gpt-6 access) from an all-unreachable one
    and only fall back to the static catalog in the latter case.

    Filters: the ``openai`` provider to the gpt-5/gpt-6 families; every provider to chat
    models. Best-effort and deduped: a provider that fails to respond is skipped."""
    defs = enabled_provider_defs()
    if not defs:
        return [], True
    fetched = await asyncio.gather(*[_fetch_provider_models(d) for d in defs])
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    live_ok = True
    for pdef, (ids, ok) in zip(defs, fetched, strict=True):
        if not ok:
            live_ok = False
        name = pdef.get("name", "")
        for model_id in ids:
            if not _keep_model(name, model_id):
                continue
            pair = (name, model_id)
            if pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
    return out, live_ok
