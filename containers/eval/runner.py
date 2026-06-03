"""Eval container runner — LLM-as-judge evaluation (stub for v1)."""

from __future__ import annotations

import sys

sys.path.insert(0, "/app")

from shared.context import RunContext


def main() -> None:
    ctx = RunContext.from_environment()
    ctx.start_heartbeat()

    try:
        ctx.emit("progress", {
            "message": "LLM-as-judge evaluation not yet implemented",
            "phase": "not_implemented",
        })
        ctx.emit("state_change", {"state": "succeeded"})

    finally:
        ctx.stop_heartbeat()


if __name__ == "__main__":
    main()
