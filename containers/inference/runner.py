"""Inference container runner — vLLM batch inference (stub for v1)."""

from __future__ import annotations

import sys

sys.path.insert(0, "/app")

from shared.context import RunContext


def main() -> None:
    ctx = RunContext.from_environment()
    ctx.start_heartbeat()

    try:
        ctx.emit("progress", {
            "message": "vLLM batch inference not yet implemented",
            "phase": "not_implemented",
        })
        ctx.emit("state_change", {"state": "succeeded"})

    finally:
        ctx.stop_heartbeat()


if __name__ == "__main__":
    main()
