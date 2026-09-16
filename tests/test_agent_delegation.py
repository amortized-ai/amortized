"""Tests for the agent proxy subagent delegation routing.

Covers subagent → subagent delegation (e.g. eval → sdg) and the
stack-based return of control when the delegated subagent completes.
"""

from typing import Any

import pytest

from amortized.api import agent


def _tool_part(tool: str, input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool", "tool": f"mcp_amortized__{tool}", "input": input}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


class _Router:
    """Stands in for OpenCode: scripted responses per session."""

    def __init__(self) -> None:
        self._next_id = 0
        # session_id -> list of parts per turn (consumed in order)
        self.script: dict[str, list[list[dict[str, Any]]]] = {}
        # session_id -> parts of the most recent response
        self.last_parts: dict[str, list[dict[str, Any]]] = {}
        self.sent: list[tuple[str, str, str | None]] = []
        # sessions handed out by _proxy_create_session, in order
        self._new_session_queue: list[str] = []

    def add_session(self, script: list[list[dict[str, Any]]]) -> str:
        self._next_id += 1
        sid = f"sess-{self._next_id}"
        self.script[sid] = [list(parts) for parts in script]
        return sid

    def queue_session(self, script: list[list[dict[str, Any]]]) -> str:
        """Pre-script a session that _proxy_create_session will hand out."""
        sid = self.add_session(script)
        self._new_session_queue.append(sid)
        return sid

    async def send(
        self,
        session_id: str,
        text: str,
        agent: str | None = None,
        model: Any = None,
    ) -> dict[str, Any]:
        self.sent.append((session_id, text, agent))
        script = self.script.get(session_id) or []
        parts = script.pop(0) if script else []
        self.last_parts[session_id] = parts
        return {"info": {"id": session_id}, "parts": parts}

    async def fetch_parts(self, session_id: str) -> list[dict[str, Any]]:
        return self.last_parts.get(session_id) or []


@pytest.fixture()
def router(monkeypatch: pytest.MonkeyPatch) -> _Router:
    r = _Router()
    monkeypatch.setattr(agent, "_proxy_send_message", r.send)
    monkeypatch.setattr(agent, "_fetch_all_assistant_parts", r.fetch_parts)
    async def _create() -> str:
        if r._new_session_queue:
            return r._new_session_queue.pop(0)
        return r.add_session([])

    monkeypatch.setattr(agent, "_proxy_create_session", _create)
    return r


def _make_state(
    router: _Router, orchestrator_script: list[list[dict[str, Any]]] | None = None
) -> agent.SessionState:
    orch_id = router.add_session(orchestrator_script or [])
    return agent.SessionState(orchestrator_id=orch_id)


def _run(coro: Any) -> Any:
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


def _body(text: str = "hi") -> agent.MessageRequest:
    return agent.MessageRequest(parts=[agent.MessagePart(type="text", text=text)])


class TestSubagentDelegation:
    def test_subagent_delegation_switches_sessions(self, router: _Router) -> None:
        # Active eval subagent delegates to sdg
        sdg_id = router.queue_session([[_text_part("What kind of dataset?")]])
        eval_id = router.add_session(
            [[_tool_part("delegate_to_subagent", {"target": "sdg", "context": "c"})]]
        )
        state = _make_state(router)
        state.subagent_id = eval_id
        state.subagent_target = "eval"

        result = _run(
            agent._handle_subagent_message(state, "s1", "build the dataset", _body())
        )

        # The new sdg session became active; eval was stashed on the stack
        assert state.subagent_target == "sdg"
        assert state.subagent_id == sdg_id
        assert len(state.subagent_stack) == 1
        stashed_target, stashed_id = state.subagent_stack[0]
        assert stashed_target == "eval"
        assert stashed_id == eval_id  # the original eval session
        # Response contains the user-visible parts from the sdg agent
        texts = [p.get("text") for p in result["parts"] if p.get("type") == "text"]
        assert "What kind of dataset?" in texts

    def test_subagent_completion_returns_to_parent_not_orchestrator(self, router: _Router) -> None:
        # sdg (child) completes; eval (parent, on the stack) resumes
        eval_id = router.add_session([[_text_part("Dataset ready — continuing eval setup")]])
        sdg_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "SDG job 123 done"})]]
        )
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"
        state.subagent_stack = [("eval", eval_id)]

        result = _run(
            agent._handle_subagent_message(state, "s1", "confirm", _body())
        )

        # Control returned to the eval agent
        assert state.subagent_id == eval_id
        assert state.subagent_target == "eval"
        assert state.subagent_stack == []
        # The orchestrator was never contacted
        assert all(s != state.orchestrator_id for s, _, _ in router.sent)
        # Parent received the completion summary
        parent_msg = next(m for m in router.sent if m[0] == eval_id)
        assert "[SUBAGENT COMPLETED]" in parent_msg[1]
        assert "SDG job 123 done" in parent_msg[1]
        # User-visible parts include both the sdg tail and the eval response
        texts = [p.get("text") for p in result["parts"] if p.get("type") == "text"]
        assert "Dataset ready — continuing eval setup" in texts

    def test_subagent_completion_without_stack_resumes_orchestrator(self, router: _Router) -> None:
        state = _make_state(router, orchestrator_script=[[ _text_part("What next?") ]])
        sdg_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "done"})]]
        )
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"

        _run(agent._handle_subagent_message(state, "s1", "confirm", _body()))

        assert state.subagent_id is None  # orchestrator takes over
        orch_msg = next(m for m in router.sent if m[0] == state.orchestrator_id)
        assert orch_msg[2] == "morty"

    def test_parent_may_delegate_again_after_child_completes(self, router: _Router) -> None:
        # eval resumes after sdg completes and immediately delegates to training
        training_id = router.queue_session([[_text_part("Which model?")]])
        eval_id = router.add_session(
            [[_tool_part("delegate_to_subagent", {"target": "training", "context": "c2"})]]
        )
        sdg_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "sdg done"})]]
        )
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"
        state.subagent_stack = [("eval", eval_id)]

        _run(agent._handle_subagent_message(state, "s1", "confirm", _body()))

        assert state.subagent_target == "training"
        assert state.subagent_id == training_id
        assert state.subagent_stack == [("eval", eval_id)]

    def test_internal_tools_stripped_from_response(self, router: _Router) -> None:
        _ = router.queue_session([[_text_part("What kind of dataset?")]])
        state = _make_state(router)
        state.subagent_id = router.add_session(
            [[_tool_part("delegate_to_subagent", {"target": "sdg", "context": "c"})]]
        )
        state.subagent_target = "eval"

        result = _run(
            agent._handle_subagent_message(state, "s1", "build it", _body())
        )
        tools = [
            p for p in result["parts"] if p.get("type") == "tool"
        ]
        assert tools == []
