"""Tests for the online capture path: the monitor_enabled gate, full-input
capture with redaction, and the never-break-a-turn guarantee."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from amortized.api import agent, monitor_log


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


def _assistant_msg(mid: str, tool: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "info": {
            "id": mid,
            "role": "assistant",
            "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": 0},
            "cost": 0.001,
            "modelID": "claude-opus-4-8",
            "providerID": "vertex",
            "time": {"created": 1},
        },
        "parts": [
            {"type": "text", "text": "working"},
            {"type": "tool", "tool": f"mcp_amortized__{tool}", "input": tool_input},
        ],
    }


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture append_record calls (bypassing its own gate) so we assert on the
    records _record_turn_metrics produces."""
    recs: list[dict[str, Any]] = []
    monkeypatch.setattr(monitor_log, "append_record", lambda sid, rec: recs.append(rec))
    return recs


def _state() -> agent.SessionState:
    return agent.SessionState(orchestrator_id="orch-1")


def test_gate_off_skips_fetch_and_write(
    monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(agent.settings, "monitor_enabled", False)
    called = {"fetch": False}

    async def _fetch(_sid: str) -> list[dict[str, Any]]:
        called["fetch"] = True
        return []

    monkeypatch.setattr(agent, "_fetch_all_messages", _fetch)
    _run(agent._record_turn_metrics(_state(), agent.TurnState(), "s1", "t1"))

    assert called["fetch"] is False  # fan-out skipped entirely
    assert captured == []


def test_gate_on_captures_redacted_full_input(
    monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]
) -> None:
    monkeypatch.setattr(agent.settings, "monitor_enabled", True)
    msg = _assistant_msg(
        "m1",
        "validate_eval_job",
        {"parent_job_id": "train-1", "judge": {"model": "gpt-oss", "api_key": "sk-secret"}},
    )

    async def _fetch(sid: str) -> list[dict[str, Any]]:
        return [msg] if sid == "orch-1" else []

    monkeypatch.setattr(agent, "_fetch_all_messages", _fetch)
    _run(agent._record_turn_metrics(_state(), agent.TurnState(), "s1", "t1"))

    assert len(captured) == 1
    call = captured[0]["tool_calls"][0]
    # promoted scalars preserved for existing matchers
    assert call["parent_job_id"] == "train-1"
    assert call["judge"] == "gpt-oss"
    # full input captured, with the secret masked
    assert call["input"]["parent_job_id"] == "train-1"
    assert call["input"]["judge"]["model"] == "gpt-oss"
    assert call["input"]["judge"]["api_key"] == "[REDACTED]"


def test_capture_never_raises_on_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent.settings, "monitor_enabled", True)

    async def _boom(_sid: str) -> list[dict[str, Any]]:
        raise RuntimeError("opencode down")

    monkeypatch.setattr(agent, "_fetch_all_messages", _boom)
    # must swallow the error (logging must never break a turn)
    _run(agent._record_turn_metrics(_state(), agent.TurnState(), "s1", "t1"))


def test_append_record_respects_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(monitor_log.settings, "monitor_log_dir", tmp_path)
    monkeypatch.setattr(monitor_log.settings, "monitor_enabled", False)
    monitor_log.append_record("sess", {"kind": "turn"})
    assert list(tmp_path.glob("*.jsonl")) == []

    monkeypatch.setattr(monitor_log.settings, "monitor_enabled", True)
    monitor_log.append_record("sess", {"kind": "turn"})
    assert (tmp_path / "sess.jsonl").exists()
