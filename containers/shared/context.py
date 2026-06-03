"""RunContext — the primary interface for container runners.

Container runners bootstrap from the environment:
- /amortized/config.json contains the job config + artifact references
- AMORTIZED_JOB_ID, AMORTIZED_EVENTS_URL, AMORTIZED_WORK_DIR are set by the control plane
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.artifacts import save_to_local
from shared.events import HeartbeatThread, emit_http, emit_stdout


@dataclass
class RunContext:
    job_id: str
    work_dir: Path
    config: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
    _events_url: str | None = field(default=None, repr=False)
    _cancel_file: Path | None = field(default=None, repr=False)
    _heartbeat: HeartbeatThread | None = field(default=None, repr=False)

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "job_id": self.job_id,
            "data": data or {},
        }
        emit_stdout(event)
        if self._events_url:
            emit_http(event, self._events_url)

    def save_artifact(self, name: str, path: Path) -> None:
        storage_dir = self.work_dir / "artifacts"
        storage_dir.mkdir(parents=True, exist_ok=True)
        dest = save_to_local(name, path, storage_dir)
        self.emit("artifact", {"name": name, "path": str(dest)})

    def is_cancelled(self) -> bool:
        if self._cancel_file is None:
            return False
        return self._cancel_file.exists()

    def start_heartbeat(self) -> None:
        self._heartbeat = HeartbeatThread(self.job_id, self._events_url)
        self._heartbeat.start()

    def stop_heartbeat(self) -> None:
        if self._heartbeat:
            self._heartbeat.stop()

    @classmethod
    def from_environment(cls) -> RunContext:
        job_id = os.environ["AMORTIZED_JOB_ID"]
        work_dir = Path(os.environ.get("AMORTIZED_WORK_DIR", "/amortized/work"))
        events_url = os.environ.get("AMORTIZED_EVENTS_URL")
        config_path = Path(os.environ.get("AMORTIZED_CONFIG_PATH", "/amortized/config.json"))

        with open(config_path) as f:
            raw = json.load(f)

        config = raw.get("config", raw)
        artifacts = raw.get("artifacts", {})
        cancel_file = work_dir / ".cancel"

        return cls(
            job_id=job_id,
            work_dir=work_dir,
            config=config,
            artifacts=artifacts,
            _events_url=events_url,
            _cancel_file=cancel_file,
        )
