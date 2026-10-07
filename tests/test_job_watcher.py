"""Tests for push-based job-completion continuation (the backend watcher).

The worker fires _notify_job_terminal when a job reaches a terminal state; it hands
off to agent.notify_job_complete, which drives the next-step turn in the job's
originating conversation. A single-fire claim dedups the watcher against the frontend
monitor card so the continuation runs exactly once. These are fast unit tests — the DB
pool and Repository are faked, no server needed.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from amortized.api import agent


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------- #
# agent.notify_job_complete
# --------------------------------------------------------------------------- #


class TestNotifyJobComplete:
    def _patch_spawn(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any, bool]]:
        spawned: list[tuple[str, Any, bool]] = []

        def _spawn(
            state: Any, session_id: str, user_text: str, body: Any, watcher: bool = False
        ) -> str:
            spawned.append((session_id, body, watcher))
            return "turn-1"

        monkeypatch.setattr(agent, "_spawn_turn", _spawn)
        return spawned

    def test_session_absent_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(agent, "_sessions", {})
        spawned = self._patch_spawn(monkeypatch)
        claimed: list[str] = []
        monkeypatch.setattr(
            agent, "_claim_job_continuation", lambda jid: claimed.append(jid) or True
        )
        ok = _run(agent.notify_job_complete("conv-gone", "job1", "training", "succeeded"))
        assert ok is False
        assert spawned == []
        # Must NOT claim when the session is gone, so the frontend can still fire later.
        assert claimed == []

    def test_claim_lost_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            agent, "_sessions", {"conv1": agent.SessionState(orchestrator_id="orch")}
        )
        spawned = self._patch_spawn(monkeypatch)

        async def _claim(_jid: str) -> bool:
            return False

        monkeypatch.setattr(agent, "_claim_job_continuation", _claim)
        ok = _run(agent.notify_job_complete("conv1", "job1", "training", "succeeded"))
        assert ok is False
        assert spawned == []

    def test_claim_won_drives_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        state.last_model = agent.MessageModel(providerID="p", modelID="m")
        monkeypatch.setattr(agent, "_sessions", {"conv1": state})
        spawned = self._patch_spawn(monkeypatch)

        async def _claim(_jid: str) -> bool:
            return True

        monkeypatch.setattr(agent, "_claim_job_continuation", _claim)
        ok = _run(agent.notify_job_complete("conv1", "job-abc", "eval", "failed"))
        assert ok is True
        assert len(spawned) == 1
        session_id, body, watcher = spawned[0]
        assert session_id == "conv1"
        assert body.event == "job_complete"
        assert body.outcome == "failed"
        # Continuation runs on the conversation's last model.
        assert body.model is state.last_model
        assert "job-abc" in body.parts[0].text
        # Watcher turns are tagged so their result is delivered via /pending.
        assert watcher is True


# --------------------------------------------------------------------------- #
# send_message job_complete dedup (frontend path loses the race -> no duplicate)
# --------------------------------------------------------------------------- #


class TestSendMessageDedup:
    def test_frontend_duplicate_returns_empty_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        monkeypatch.setattr(agent, "_sessions", {"conv1": state})

        async def _resolve(_text: str) -> str:
            return "job-xyz"

        async def _claim_lost(_jid: str) -> bool:
            return False

        spawned: list[Any] = []
        monkeypatch.setattr(agent, "_completed_job_id", _resolve)
        monkeypatch.setattr(agent, "_claim_job_continuation", _claim_lost)
        monkeypatch.setattr(agent, "_spawn_turn", lambda *a, **k: spawned.append(a) or "t")

        body = agent.MessageRequest(
            parts=[agent.MessagePart(type="text", text="Job job-xyz (eval) finished")],
            event="job_complete",
            outcome="succeeded",
        )
        out = _run(agent.send_message("conv1", body))
        # No new turn spawned. The watcher drives the continuation and delivers its result
        # via /pending; the frontend receives an empty sentinel (NOT a stale snapshot) so
        # it suppresses its placeholder.
        assert spawned == []
        assert out["info"]["id"] == "job-complete-duplicate"
        assert out["parts"] == []

    def test_frontend_wins_spawns_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        monkeypatch.setattr(agent, "_sessions", {"conv1": state})

        async def _resolve(_text: str) -> str:
            return "job-xyz"

        async def _claim_won(_jid: str) -> bool:
            return True

        spawned: list[Any] = []
        monkeypatch.setattr(agent, "_completed_job_id", _resolve)
        monkeypatch.setattr(agent, "_claim_job_continuation", _claim_won)
        monkeypatch.setattr(
            agent, "_spawn_turn", lambda *a, **k: spawned.append(a) or "turn-9"
        )

        body = agent.MessageRequest(
            parts=[agent.MessagePart(type="text", text="Job job-xyz (eval) finished")],
            event="job_complete",
            outcome="succeeded",
        )
        out = _run(agent.send_message("conv1", body))
        assert out == {"turn_id": "turn-9", "status": "processing"}
        assert len(spawned) == 1


# --------------------------------------------------------------------------- #
# worker._notify_job_terminal (job -> conversation hand-off)
# --------------------------------------------------------------------------- #


def _patch_pool(monkeypatch: pytest.MonkeyPatch, job: dict[str, Any] | None) -> None:
    from amortized import worker as worker_mod
    from amortized.db import connection as conn_mod

    class FakeRepo:
        def __init__(self, _conn: Any) -> None:
            pass

        async def get_job(self, _job_id: str) -> dict[str, Any] | None:
            return job

    @asynccontextmanager
    async def _acquire() -> Any:
        yield object()

    class FakePool:
        def acquire(self) -> Any:
            return _acquire()

    monkeypatch.setattr(worker_mod, "Repository", FakeRepo)
    monkeypatch.setattr(conn_mod, "get_pool", lambda: FakePool())


class TestNotifyJobTerminal:
    def test_hands_off_with_conversation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from amortized import worker as worker_mod

        _patch_pool(
            monkeypatch,
            {"id": "job1", "type": "training", "conversation_id": "conv1"},
        )
        calls: list[tuple[str, str, str, str]] = []

        async def _notify(conv: str, jid: str, jtype: str, status: str) -> bool:
            calls.append((conv, jid, jtype, status))
            return True

        monkeypatch.setattr(agent, "notify_job_complete", _notify)
        _run(worker_mod._notify_job_terminal("job1", "succeeded"))
        assert calls == [("conv1", "job1", "training", "succeeded")]

    def test_no_conversation_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from amortized import worker as worker_mod

        _patch_pool(monkeypatch, {"id": "job1", "type": "sdg", "conversation_id": ""})
        calls: list[Any] = []
        monkeypatch.setattr(
            agent, "notify_job_complete", lambda *a: calls.append(a)
        )
        _run(worker_mod._notify_job_terminal("job1", "succeeded"))
        assert calls == []

    def test_missing_job_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from amortized import worker as worker_mod

        _patch_pool(monkeypatch, None)
        calls: list[Any] = []
        monkeypatch.setattr(agent, "notify_job_complete", lambda *a: calls.append(a))
        _run(worker_mod._notify_job_terminal("missing", "failed"))
        assert calls == []


# --------------------------------------------------------------------------- #
# /pending delivery of watcher-turn results
# --------------------------------------------------------------------------- #


class TestPendingDelivery:
    def test_get_pending_drains_watcher_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        state.pending_results = [{"info": {"id": "w1"}, "parts": []}]
        monkeypatch.setattr(agent, "_sessions", {"conv1": state})

        async def _proxy_get(_t: str, _p: str, _e: Any, _s: str) -> dict[str, Any]:
            return {"messages": [{"info": {"id": "o1"}}]}

        monkeypatch.setattr(agent, "_proxy_get", _proxy_get)
        out = _run(agent.get_pending("conv1"))
        # opencode-pending messages first, then the drained watcher results.
        assert [m["info"]["id"] for m in out["messages"]] == ["o1", "w1"]
        # Drained exactly once: a second poll no longer carries the watcher result.
        assert state.pending_results == []
        out2 = _run(agent.get_pending("conv1"))
        assert [m["info"]["id"] for m in out2["messages"]] == ["o1"]

    def test_get_pending_with_empty_opencode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        state.pending_results = [{"info": {"id": "w1"}, "parts": []}]
        monkeypatch.setattr(agent, "_sessions", {"conv1": state})

        async def _proxy_get(_t: str, _p: str, _e: Any, _s: str) -> dict[str, Any]:
            return {"messages": []}

        monkeypatch.setattr(agent, "_proxy_get", _proxy_get)
        out = _run(agent.get_pending("conv1"))
        assert [m["info"]["id"] for m in out["messages"]] == ["w1"]


class TestRunTurnEnqueue:
    def _patch_internals(
        self, monkeypatch: pytest.MonkeyPatch, result: Any, tool_parts: Any = None
    ) -> None:
        async def _handle_orch(_state: Any, _sid: str, _text: str, _body: Any) -> Any:
            return result

        async def _passthrough(_state: Any, res: Any, _body: Any, _cache: Any) -> Any:
            return res

        async def _metrics(*_a: Any, **_k: Any) -> None:
            return None

        async def _fetch(_sid: str) -> list[dict[str, Any]]:
            return tool_parts or []

        monkeypatch.setattr(agent, "_handle_orchestrator_message", _handle_orch)
        monkeypatch.setattr(agent, "_apply_provenance_gate", _passthrough)
        monkeypatch.setattr(agent, "_apply_job_claim_gate", _passthrough)
        monkeypatch.setattr(agent, "_ensure_phase_signal", lambda _state, res: res)
        monkeypatch.setattr(agent, "_enforce_single_interaction", lambda res, _state: res)
        monkeypatch.setattr(agent, "_record_turn_metrics", _metrics)
        monkeypatch.setattr(agent, "_fetch_all_assistant_parts", _fetch)

    def test_watcher_turn_enqueues_delivery_with_tool_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        # The delegation-path result carries only step markers; the confirm card's tool
        # part lives in the GET-messages fetch and must be folded into the delivery.
        result = {"info": {"id": "r1"}, "parts": [{"type": "step-start"}]}
        tool_parts = [{"type": "tool", "tool": "validate_sdg_job", "output": "{}"}]
        self._patch_internals(monkeypatch, result, tool_parts)
        state.turns["t1"] = agent.TurnState(watcher=True)
        body = agent.MessageRequest(parts=[agent.MessagePart(type="text", text="go")])
        _run(agent._run_turn(state, "conv1", "t1", "go", body))
        assert len(state.pending_results) == 1
        delivered = state.pending_results[0]
        assert delivered["info"]["id"] == "r1"
        # The frontend only harvests tools from an assistant-role message.
        assert delivered["info"]["role"] == "assistant"
        # Both the gated result parts and the fetched tool-call details are present.
        assert {"type": "step-start"} in delivered["parts"]
        assert tool_parts[0] in delivered["parts"]

    def test_user_turn_not_enqueued(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        result = {"info": {"id": "r1"}, "parts": []}
        self._patch_internals(monkeypatch, result)
        state.turns["t1"] = agent.TurnState(watcher=False)
        body = agent.MessageRequest(parts=[agent.MessagePart(type="text", text="go")])
        _run(agent._run_turn(state, "conv1", "t1", "go", body))
        assert state.pending_results == []
