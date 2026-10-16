"""The proxy injects the active phase's instruction block into each turn message.

Subagents get the block for the phase they last signalled (entry phase on a fresh
delegation); the orchestrator — which never signals — gets the block for the
control-flow path it is on (intake/delegate on a fresh turn, resume after a
completion, fulfill_help on a hand-back). The clean user_text still flows downstream
as the subagent's [USER MESSAGE], so orchestrator-only guidance never leaks.
"""

from typing import Any

import pytest

from amortized.api import agent


def _tool_part(tool: str, input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool", "tool": f"mcp_amortized__{tool}", "input": input}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


class _Router:
    def __init__(self) -> None:
        self._n = 0
        self.last_parts: dict[str, list[dict[str, Any]]] = {}
        self.sent: list[tuple[str, str, str | None]] = []
        self.queued: list[str] = []
        self.scripts: dict[str, list[list[dict[str, Any]]]] = {}

    def add(self, script: list[list[dict[str, Any]]] | None = None) -> str:
        self._n += 1
        sid = f"s{self._n}"
        self.scripts[sid] = [list(p) for p in (script or [])]
        return sid

    def queue(self, script: list[list[dict[str, Any]]]) -> str:
        sid = self.add(script)
        self.queued.append(sid)
        return sid

    async def send(
        self, session_id: str, text: str, agent: str | None = None, model: Any = None
    ) -> dict[str, Any]:
        self.sent.append((session_id, text, agent))
        script = self.scripts.get(session_id) or []
        parts = script.pop(0) if script else []
        self.last_parts[session_id] = parts
        return {"info": {"id": session_id}, "parts": parts}

    async def fetch(self, session_id: str) -> list[dict[str, Any]]:
        return self.last_parts.get(session_id) or []

    def text_to(self, session_id: str) -> str:
        return next(t for sid, t, _ in self.sent if sid == session_id)


@pytest.fixture()
def router(monkeypatch: pytest.MonkeyPatch) -> _Router:
    r = _Router()
    monkeypatch.setattr(agent, "_proxy_send_message", r.send)
    monkeypatch.setattr(agent, "_fetch_all_assistant_parts", r.fetch)

    async def _create() -> str:
        return r.queued.pop(0) if r.queued else r.add()

    monkeypatch.setattr(agent, "_proxy_create_session", _create)
    return r


def _run(coro: Any) -> Any:
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


def _body(text: str = "hi", event: str | None = None) -> agent.MessageRequest:
    return agent.MessageRequest(parts=[agent.MessagePart(type="text", text=text)], event=event)


class TestSubagentInjection:
    def test_fresh_eval_delegation_injects_entry_phase(self, router: _Router) -> None:
        state = agent.SessionState(orchestrator_id=router.add())
        state.last_signal_step = "step-3-metrics"  # stale from a prior subagent
        new_sid = router.queue([[_text_part("Which model?")]])

        _run(
            agent._maybe_delegate(
                state, "s", ("eval", "train job tj-1 done", False), "evaluate it", {}, _body()
            )
        )

        sent = router.text_to(new_sid)
        # entry phase block (step-0), not the stale step-3, leads the handoff
        assert "step-0-what-to-eval" in sent
        assert "Confirm WHAT to evaluate" in sent
        assert "[USER MESSAGE]\nevaluate it" in sent
        # a fresh subagent's cursor is reset to its entry phase
        assert state.last_signal_step == ""

    def test_resume_injects_last_signalled_phase(self, router: _Router) -> None:
        stashed = router.add([[_text_part("resumed")]])
        state = agent.SessionState(orchestrator_id=router.add())
        state.completed_subagents["eval"] = stashed
        state.last_signal_step = "step-1_5-how-much"

        _run(
            agent._maybe_delegate(
                state, "s", ("eval", "set ready", True), "continue", {}, _body()
            )
        )

        sent = router.text_to(stashed)
        assert "step-1_5-how-much" in sent
        assert "[RESUMED]" in sent


class TestOrchestratorInjection:
    def test_fresh_turn_injects_intake_and_delegate(self, router: _Router) -> None:
        orch = router.add([[_text_part("What do you want to build?")]])
        state = agent.SessionState(orchestrator_id=orch)

        _run(agent._handle_orchestrator_message(state, "s", "help me", _body()))

        sent = router.text_to(orch)
        assert "Phase 1 — Identify Intent" in sent
        assert "Phase 2 — Delegate" in sent
        assert "help me" in sent

    def test_job_complete_turn_injects_monitor(self, router: _Router) -> None:
        orch = router.add([[_text_part("Here are next steps")]])
        state = agent.SessionState(orchestrator_id=orch)

        _run(
            agent._handle_orchestrator_message(
                state, "s", "Job x finished", _body(event="job_complete")
            )
        )

        sent = router.text_to(orch)
        assert "Phase 4 — Monitor" in sent
        assert "Phase 1 — Identify Intent" not in sent

    def test_clean_user_text_flows_downstream(self, router: _Router) -> None:
        # The orchestrator delegates to sdg; the sdg handoff must carry the clean
        # user_text as [USER MESSAGE], not the orchestrator's intake/delegate blocks.
        orch = router.add([[_tool_part("delegate_to_subagent", {"target": "sdg", "context": "c"})]])
        state = agent.SessionState(orchestrator_id=orch)
        sdg = router.queue([[_text_part("What task?")]])

        _run(agent._handle_orchestrator_message(state, "s", "make data", _body()))

        sdg_text = router.text_to(sdg)
        assert "make data" in sdg_text
        assert "Phase 1 — Identify Intent" not in sdg_text  # orchestrator-only guidance
        # sdg gets its OWN entry phase block instead
        assert "phase-1-route" in sdg_text
