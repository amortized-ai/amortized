"""Redact sensitive fields from job configs and credential values from text."""

from __future__ import annotations

import re
from typing import Any

_REDACT_FIELDS = frozenset({"api_key", "api_secret", "token", "password", "secret"})

_REDACTED = "***redacted***"

_CREDENTIAL_VALUE_PATTERN = re.compile(
    r"((?:[A-Z_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL))\s*[=:]\s*)(\S+)",
    re.IGNORECASE,
)


def redact_text(text: str) -> str:
    """Redact credential values from log/error text."""
    return _CREDENTIAL_VALUE_PATTERN.sub(r"\1***redacted***", text)


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
