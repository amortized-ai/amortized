"""Secret-masking + size-bounding for monitor capture.

The monitor logs full tool inputs (and, for job-created records, job configs) so
offline checks can read any field without a logger change. Those payloads can
carry credentials and large blobs, so everything is run through ``redact`` first:
values under secret-looking keys are masked, long strings are truncated, and the
structure is depth/breadth bounded. ``redact`` never raises — a redaction failure
must never break a turn or a job submission.
"""

from __future__ import annotations

import re
from typing import Any

_REDACTED = "[REDACTED]"
_MAX_STR = 16_000  # per-string cap (chars)
_MAX_DEPTH = 8  # nesting bound
_MAX_ITEMS = 200  # list/dict breadth bound

# Key names whose VALUE is masked. Conservative on purpose: a false positive only
# hides a field from a check, a false negative leaks a credential into the log.
# Bare "auth" is excluded so it doesn't swallow "author"/"authority".
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|apikey|secret|password|passwd|authorization|access[_-]?key"
    r"|token|credential|bearer|private[_-]?key)",
    re.IGNORECASE,
)


def _truncate(s: str) -> str:
    if len(s) <= _MAX_STR:
        return s
    return s[:_MAX_STR] + f"…[+{len(s) - _MAX_STR} chars]"


def redact(obj: Any, *, _depth: int = 0) -> Any:
    """Return a JSON-safe copy of ``obj`` with secret-keyed values masked, strings
    truncated, and depth/breadth bounded. Never raises."""
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
                if _SECRET_KEY_RE.search(key):
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
        if isinstance(obj, str):
            return _truncate(obj)
        if isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        # Fallback for anything non-JSON-native (datetime, Path, pydantic, ...).
        return _truncate(str(obj))
    except Exception:
        return _REDACTED
