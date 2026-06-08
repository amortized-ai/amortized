"""Redact sensitive fields from job configs before returning in API responses."""

from __future__ import annotations

from typing import Any

_REDACT_FIELDS = frozenset({"api_key", "api_secret", "token", "password", "secret"})

_REDACTED = "***redacted***"


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: _REDACTED if k in _REDACT_FIELDS and v else v for k, v in config.items()}
