"""Value-only secret scrubbing for the monitor.

The monitor captures an explicit ALLOWLIST of known-safe fields — it never stores
raw tool inputs or job configs, so there is no secret-key denylist to maintain and
no bulk payload to mask. The only residual risk is a secret a user *pastes into a
free-text field* (a present_options question, an option title) or one that an
upstream tool *echoes in its output*. ``scrub_text`` masks those by value shape
(URLs with embedded credentials, bearer tokens, JWTs, ``sk-…``/``AKIA…`` keys,
``KEY=…`` assignments) and bounds length. It never raises.
"""

from __future__ import annotations

import re

_REDACTED = "[REDACTED]"
_MAX_STR = 16_000  # per-string cap (chars)

# Value-shaped secret patterns (no key names — purely by content shape).
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


def scrub_text(s: str) -> str:
    """Mask secret-shaped substrings in a free-text/output string and cap its length.
    Never raises; returns the input unchanged if anything goes wrong."""
    try:
        for pattern, repl in _SCRUBBERS:
            s = pattern.sub(repl, s)
        if len(s) > _MAX_STR:
            s = s[:_MAX_STR] + f"…[+{len(s) - _MAX_STR} chars]"
        return s
    except Exception:
        return _REDACTED
