"""The contract between job builders and the model-liveness check.

A builder declares the LLM endpoints its job will call as ``ModelEndpoint``s; the checker
probes each one and reports a ``CheckResult``. Credentials are resolved from the same
environment the job container receives, so the check authenticates exactly as the workload
will. Keys never leave this layer except in the outbound check request: results, errors, and
log lines carry only redacted URLs and a non-reversible credential id.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from amortized.core.model_catalog import resolve_provider
from amortized.model_check.types import (
    FAILED,
    PASSED,
    UNVERIFIED,
    CheckResult,
    ModelEndpoint,
    is_http_url,
    is_valid_http_url,
    redact,
    redact_url,
)

# The shared types live in the self-contained checker package; re-exported for callers.
__all__ = [
    "FAILED",
    "PASSED",
    "UNVERIFIED",
    "CheckResult",
    "ConfigError",
    "ModelEndpoint",
    "dedup",
    "endpoint_with_credential",
    "is_http_url",
    "is_valid_http_url",
    "provider_endpoint",
    "redact",
    "redact_url",
]


@dataclass(frozen=True)
class ConfigError:
    """A model endpoint that can't be checked because the config itself is wrong."""

    label: str
    reason: str
    message: str


def endpoint_with_credential(
    label: str,
    base_url: str,
    model: str,
    provider_type: str,
    api_key_env: str,
    env: dict[str, str],
) -> ModelEndpoint:
    """A ModelEndpoint whose key is read from the job's effective env under ``api_key_env``.

    No key there means the workload calls the endpoint unauthenticated, so the check does too."""
    api_key = env.get(api_key_env, "") if api_key_env else ""
    credential_id = (
        f"{api_key_env}:{hashlib.sha256(api_key.encode()).hexdigest()[:12]}" if api_key else ""
    )
    return ModelEndpoint(
        label=label,
        base_url=base_url.rstrip("/"),
        model=model,
        provider_type=provider_type or "openai",
        credential_id=credential_id,
        api_key=api_key,
    )


def provider_endpoint(
    label: str, provider_name: str, model: str, env: dict[str, str]
) -> ModelEndpoint | ConfigError:
    """Resolve a model served by a named provider (e.g. an SDG teacher's ``provider``)."""
    provider = resolve_provider(provider_name)
    if provider is None:
        return ConfigError(
            label,
            "provider_not_configured",
            f"{label}: provider {provider_name!r} is not configured on the server"
            " (unknown, disabled, or missing its API key)",
        )
    key_ref = provider.get("api_key", "")
    if key_ref.isupper() and "_" in key_ref:
        endpoint = endpoint_with_credential(
            label,
            provider.get("endpoint", ""),
            model,
            provider.get("provider_type", "openai"),
            key_ref,
            env,
        )
    else:  # a literal key in the provider definition
        endpoint = endpoint_with_credential(
            label,
            provider.get("endpoint", ""),
            model,
            provider.get("provider_type", "openai"),
            "PROVIDER_KEY",
            {"PROVIDER_KEY": key_ref},
        )
    if not endpoint.api_key:
        return ConfigError(
            label,
            "provider_not_configured",
            f"{label}: provider {provider_name!r} has no API key available to the job",
        )
    return endpoint


def dedup(endpoints: list[ModelEndpoint]) -> list[ModelEndpoint]:
    """Drop repeat checks of the same model with the same credential (order preserved)."""
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[ModelEndpoint] = []
    for endpoint in endpoints:
        key = endpoint.dedup_key()
        if key not in seen:
            seen.add(key)
            unique.append(endpoint)
    return unique
