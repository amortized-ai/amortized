"""Append-only per-session JSONL logging for monitor (vibe-testing) metrics.

Every conversation writes to ``<monitor_log_dir>/<session_id>.jsonl``. Two record
kinds share the file:

- ``turn``      — one per completed proxy turn (tokens, tool calls, timing),
  written by the agent proxy on turn completion.
- ``completion``— written when the user clicks "mark run complete" in Studio.

All functions are best-effort: they swallow every error so that logging can
never break, delay, or alter an agent turn. The processing script in
``monitor/scripts/`` reads these files offline.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from amortized.config import settings

logger = logging.getLogger(__name__)


def _log_path(session_id: str) -> Path:
    # session_id is a server-generated uuid, but guard against path traversal anyway.
    safe = session_id.replace("/", "_").replace("..", "_")
    directory = Path(settings.resolved_monitor_log_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe}.jsonl"


def append_record(session_id: str, record: dict[str, Any]) -> None:
    """Append one JSON line to the session's log. Never raises."""
    try:
        record.setdefault("ts", datetime.now(UTC).isoformat())
        path = _log_path(session_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.warning("monitor: failed to append record for session=%s", session_id, exc_info=True)
