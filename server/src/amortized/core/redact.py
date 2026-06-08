"""Redact sensitive fields from job configs before returning in API responses."""

from __future__ import annotations

from typing import Any

_REDACT_FIELDS = frozenset({"api_key", "api_secret", "token", "password", "secret"})

_REDACTED = "***redacted***"


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in config.items():
        if k in _REDACT_FIELDS and v:
            result[k] = _REDACTED
        elif isinstance(v, dict):
            result[k] = redact_config(v)
        elif isinstance(v, list):
            result[k] = [redact_config(item) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result
