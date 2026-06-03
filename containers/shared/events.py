"""Event emission utilities for container runners.

Events are emitted via two channels:
1. stdout JSON lines — always works, even if control plane is unreachable
2. HTTP POST to AMORTIZED_EVENTS_URL — real-time delivery when reachable
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

import urllib.request


def emit_stdout(event: dict[str, Any]) -> None:
    line = json.dumps(event, default=str)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def emit_http(event: dict[str, Any], url: str) -> None:
    try:
        data = json.dumps(event, default=str).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


class HeartbeatThread:
    """Background thread that emits heartbeat events every interval_seconds."""

    def __init__(
        self,
        job_id: str,
        events_url: str | None,
        interval_seconds: int = 60,
    ) -> None:
        self.job_id = job_id
        self.events_url = events_url
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            event = {
                "type": "heartbeat",
                "timestamp": time.time(),
                "job_id": self.job_id,
            }
            emit_stdout(event)
            if self.events_url:
                emit_http(event, self.events_url)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
