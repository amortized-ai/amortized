"""Secret-masking + size-bounding for monitor capture.

The monitor logs full tool inputs (and, for job-created records, job configs) so
offline checks can read any field without a logger change. Those payloads can
carry credentials and large blobs, so everything is run through ``redact`` first:

- values under secret-*looking* keys are masked (``api_key``, ``authorization``,
  ``*_token`` … — but NOT ``max_tokens`` / ``tokenizer`` / ``max_new_tokens``,
  which are exactly the fields the monitor analyses), and
- every string value is also scrubbed for secret-*shaped* content (URLs with
  embedded credentials, bearer tokens, JWTs, ``sk-…``/``AKIA…`` keys, ``KEY=…``
  assignments) so a secret that rides in a free-text field or a non-secret key
  is masked too, then truncated and depth/breadth bounded.

``redact`` never raises — a redaction failure must never break a turn or a job
submission.
"""

from __future__ import annotations

import re
from typing import Any

_REDACTED = "[REDACTED]"
_MAX_STR = 16_000  # per-string cap (chars)
_MAX_DEPTH = 8  # nesting bound
_MAX_ITEMS = 200  # list/dict breadth bound

# Keys whose VALUE is masked. Deliberately precise so monitor-relevant fields
# (max_tokens, max_new_tokens, max_tokens_per_gpu, tokenizer, num_tokens) are NOT
# caught — only auth-token-shaped names are. False negatives are covered by the
# value scrub below.
_SECRET_EXACT = {
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "auth",
    "bearer",
    "credential",
    "credentials",
}
_SECRET_SUFFIX = ("_token", "_secret", "_password", "_passwd", "_credential", "_credentials")
_SECRET_SUBSTR = (
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "access-key",
    "secret_key",
    "secret-key",
    "private_key",
    "private-key",
    "client_secret",
    "aws_secret",
    "password",
    "passwd",
)


def _is_secret_key(key: str) -> bool:
    k = key.lower()
    if k in _SECRET_EXACT:
        return True
    if any(k.endswith(s) for s in _SECRET_SUFFIX):
        return True
    return any(sub in k for sub in _SECRET_SUBSTR)


# Value-shaped secret patterns, applied to every string leaf.
_SCRUBBERS: tuple[tuple[re.Pattern[str], str], ...] = (
    # URL userinfo credentials: scheme://user:pass@host -> scheme://[REDACTED]@host
    (
        re.compile(r"([a-z][a-z0-9+.\-]*://)[^\s:/@]+:[^\s@/]+@", re.IGNORECASE),
        r"\1" + _REDACTED + "@",
    ),
    # KEY=secret / TOKEN=... assignments (keep the name, mask the value)
    (
        re.compile(
            r"\b([A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Za-z0-9_]*)\s*=\s*\S+",
            re.IGNORECASE,
        ),
        r"\1=" + _REDACTED,
    ),
    # Bearer <token>
    (re.compile(r"\b([Bb]earer)\s+[A-Za-z0-9._\-]{8,}"), r"\1 " + _REDACTED),
    # JWTs
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}"), _REDACTED),
    # Provider key prefixes: sk-…, ghp_…, xoxb-…
    (re.compile(r"\b(?:sk|pk|rk|ghp|gho|ghs|ghu|xox[baprs])[-_][A-Za-z0-9]{10,}\b"), _REDACTED),
    # AWS access key ids
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{12,}\b"), _REDACTED),
)


def _scrub_text(s: str) -> str:
    for pattern, repl in _SCRUBBERS:
        s = pattern.sub(repl, s)
    return s


def _truncate(s: str) -> str:
    if len(s) <= _MAX_STR:
        return s
    return s[:_MAX_STR] + f"…[+{len(s) - _MAX_STR} chars]"


def redact(obj: Any, *, _depth: int = 0) -> Any:
    """Return a JSON-safe copy of ``obj`` with secret-keyed values masked,
    secret-shaped string content scrubbed, strings truncated, and depth/breadth
    bounded. Never raises."""
    try:
        if _depth >= _MAX_DEPTH:
            return "[…]"
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for i, (k, v) in enumerate(obj.items()):
                if i >= _MAX_ITEMS:
                    out["…"] = f"[+{len(obj) - _MAX_ITEMS} keys]"
                    break
                key = str(k)
                if _is_secret_key(key):
                    out[key] = _REDACTED
                else:
                    out[key] = redact(v, _depth=_depth + 1)
            return out
        if isinstance(obj, (list, tuple)):
            items = list(obj)
            capped = [redact(v, _depth=_depth + 1) for v in items[:_MAX_ITEMS]]
            if len(items) > _MAX_ITEMS:
                capped.append(f"[+{len(items) - _MAX_ITEMS} items]")
            return capped
        if isinstance(obj, bool) or obj is None or isinstance(obj, (int, float)):
            return obj
        if isinstance(obj, str):
            return _truncate(_scrub_text(obj))
        # Fallback for anything non-JSON-native (datetime, Path, pydantic, ...).
        return _truncate(_scrub_text(str(obj)))
    except Exception:
        return _REDACTED
