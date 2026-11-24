"""Types shared by the model-liveness checker and its callers.

Stdlib only: this module ships in the standalone model-check image, so it must not import
anything else from ``amortized``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PASSED = "passed"
FAILED = "failed"
UNVERIFIED = "unverified"

_URL_RE = re.compile(r"https?://[^\s'\"<>]+")


@dataclass(frozen=True)
class ModelEndpoint:
    label: str
    base_url: str
    model: str
    provider_type: str = "openai"
    credential_id: str = ""
    api_key: str = field(default="", repr=False)

    def dedup_key(self) -> tuple[str, str, str, str]:
        return (self.provider_type, self.base_url, self.model, self.credential_id)

    def to_request(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "base_url": self.base_url,
            "model": self.model,
            "provider_type": self.provider_type,
            "api_key": self.api_key,
        }


@dataclass(frozen=True)
class CheckResult:
    label: str
    model: str
    base_url: str
    outcome: str  # PASSED | FAILED | UNVERIFIED
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "model": self.model,
            "base_url": self.base_url,
            "outcome": self.outcome,
            "reason": self.reason,
            "detail": self.detail,
        }


def is_http_url(value: str) -> bool:
    """Whether ``value`` names an http(s) URL (it may still be malformed — see
    ``is_valid_http_url``)."""
    return value.strip().lower().startswith(("http://", "https://"))


def is_valid_http_url(value: str) -> bool:
    """An http(s) URL with a host and, if given, a numeric port in range."""
    if not is_http_url(value):
        return False
    try:
        parts = urlsplit(value.strip())
        _ = parts.port  # raises ValueError for a non-numeric or out-of-range port
    except ValueError:
        return False
    return bool(parts.hostname)


def redact_url(url: str) -> str:
    """Drop userinfo, query, and fragment.
    Never raises: anything that doesn't parse cleanly becomes ``<invalid url>``."""
    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError for a malformed port
    except ValueError:
        return "<invalid url>"
    if not parts.scheme or not parts.hostname:
        return "<invalid url>"
    netloc = parts.hostname + (f":{port}" if port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def redact(text: str, secrets: list[str] | tuple[str, ...] = ()) -> str:
    """Redact every URL in ``text`` and every occurrence of a known secret."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return _URL_RE.sub(lambda m: redact_url(m.group(0)), text)
