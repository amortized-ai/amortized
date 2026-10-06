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


def _body(text: str = "hi", event: str | None = None) -> agent.MessageRequest:
    return agent.MessageRequest(
        parts=[agent.MessagePart(type="text", text=text)], event=event
    )


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


class TestJobCompleteHandback:
    """When an active subagent's delegated job finishes, the server steers it to
    hand control back instead of letting it loop on post-job inspection calls."""

    def _active_sdg(self, router: _Router) -> agent.SessionState:
        # Subagent responds with plain text (no completion signal) so the handler
        # returns normally and we can inspect what text was forwarded to it.
        sdg_id = router.add_session([[_text_part("ok")]])
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"
        return state

    def test_success_event_rewrites_to_handback(self, router: _Router) -> None:
        state = self._active_sdg(router)
        text = (
            "Job d6be24ce (sdg) finished with status: succeeded."
            " Use present_options to suggest next steps to the user."
        )
        _run(
            agent._handle_subagent_message(
                state, "s1", text, _body(event="job_complete")
            )
        )
        forwarded = router.sent[0][1]
        assert forwarded == agent._SUBAGENT_JOB_DONE_PROMPT
        assert "signal_subagent_completion" in forwarded

    def test_failed_job_not_rewritten(self, router: _Router) -> None:
        state = self._active_sdg(router)
        text = (
            "Job d6be24ce (sdg) finished with status: failed."
            " Use present_options to suggest next steps to the user."
        )
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        assert router.sent[0][1] == text  # untouched — subagent handles the failure

    def test_normal_message_not_rewritten(self, router: _Router) -> None:
        state = self._active_sdg(router)
        _run(agent._handle_subagent_message(state, "s1", "confirm", _body()))
        assert router.sent[0][1] == "confirm"


class TestEnforceSingleInteraction:
    """A turn either ASKS one question or CONFIRMS one job, never both. Whichever
    interaction leads the turn wins; the other kind (and repeats) is stripped so
    the UI never shows a question batched with a confirm card."""

    def test_ask_turn_keeps_only_first_question_but_preserves_passive_parts(self) -> None:
        # Extra questions and a premature confirm card are stripped, but passive
        # content (text, show_* cards) is kept so the user still sees context.
        result = {
            "parts": [
                _text_part("here are your options"),
                _tool_part("present_options", {"question": "which teacher?"}),
                _tool_part("present_options", {"question": "how many records?"}),
                _tool_part("show_prompt", {}),
                _tool_part("present_options", {"question": "prompt ok?"}),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
            ]
        }
        out = agent._enforce_single_interaction(result)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools.count("present_options") == 1  # exactly one question survives
        assert "validate_sdg_job" not in tools  # premature confirm card dropped
        assert "show_prompt" in tools  # passive display card preserved
        assert any(p.get("type") == "text" for p in out["parts"])  # text preserved

    def test_ask_turn_preserves_text_written_after_the_question(self) -> None:
        # Regression: a driver wrote the proposed metric-set table as text AFTER
        # the approval question; it must not be dropped, or the user approves
        # option buttons with no metrics shown.
        result = {
            "parts": [
                _tool_part("present_options", {"question": "Approve this metric set?"}),
                _text_part("| Criterion | What it checks |\n|---|---|\n| accuracy | ... |"),
                _tool_part("present_options", {"question": "Approve this metric set?"}),
            ]
        }
        out = agent._enforce_single_interaction(result)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools == ["present_options"]  # the duplicate question is stripped
        texts = [p for p in out["parts"] if p.get("type") == "text"]
        assert texts and "Criterion" in texts[0]["text"]  # the metric table survives

    def test_confirm_turn_drops_options_stacked_under_the_card(self) -> None:
        # validate_* first, then a redundant "confirm?" present_options -> the
        # confirmation card wins; the stacked question is dropped, passive cards stay.
        result = {
            "parts": [
                _tool_part("validate_training_job", {"algorithm": "sft"}),
                _tool_part("show_vram_estimate", {}),
                _tool_part("signal_phase", {"phase": "training", "step": "confirm"}),
                _tool_part("present_options", {"question": "Submit this job?"}),
            ]
        }
        out = agent._enforce_single_interaction(result)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert "present_options" not in tools  # no options under the confirm card
        assert tools == ["validate_training_job", "show_vram_estimate", "signal_phase"]

    def test_confirm_turn_drops_repeat_confirm_card(self) -> None:
        result = {
            "parts": [
                _text_part("confirm below"),
                _tool_part("validate_training_job", {"algorithm": "osft"}),
                _tool_part("present_options", {"question": "ok?"}),
                _tool_part("validate_training_job", {"algorithm": "osft"}),
            ]
        }
        out = agent._enforce_single_interaction(result)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools == ["validate_training_job"]

    def test_show_prompt_gate_drops_confirm_card_stacked_under_the_prompt(self) -> None:
        # The exact observed batch: show the assessor prompt, then stack the SDG
        # confirmation (repeated) under it in the same turn. The prompt review wins;
        # every confirm card is dropped so the prompt stands alone for review.
        result = {
            "parts": [
                _tool_part("signal_phase", {"phase": "sdg", "step": "confirm"}),
                _tool_part("show_prompt", {"title": "Assessor system prompt"}),
                _tool_part("get_model_pricing", {}),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
            ]
        }
        out = agent._enforce_single_interaction(result)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert "validate_sdg_job" not in tools  # job confirm deferred to next turn
        assert tools == ["signal_phase", "show_prompt", "get_model_pricing"]

    def test_show_prompt_with_review_question_keeps_the_question(self) -> None:
        # The CORRECT flow: show the prompt, then ask approve/edit. This is an ask
        # turn — the present_options leads and the prompt rides along as passive
        # context. The gate must NOT fire here.
        result = {
            "parts": [
                _tool_part("show_prompt", {"title": "Assessor system prompt"}),
                _tool_part("present_options", {"question": "Approve this prompt?"}),
            ]
        }
        out = agent._enforce_single_interaction(result)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools == ["show_prompt", "present_options"]

    def test_noop_without_any_interaction(self) -> None:
        result = {"parts": [_text_part("working"), _tool_part("list_models", {})]}
        out = agent._enforce_single_interaction(result)
        assert out["parts"] == result["parts"]
        assert out is result  # unchanged object when nothing to cut

    def test_noop_when_present_options_is_last_and_alone(self) -> None:
        result = {
            "parts": [
                _text_part("pick one"),
                _tool_part("present_options", {"question": "which?"}),
            ]
        }
        out = agent._enforce_single_interaction(result)
        assert len(out["parts"]) == 2


def _signal_phase_results(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in parts if agent._tool_name(p) == "signal_phase"]


class TestPhaseBackfill:
    """When an active subagent's turn carries no signal_phase, the server
    backfills one from subagent_target so the UI progress bar is never stale."""

    def test_backfills_phase_when_subagent_forgets(self, router: _Router) -> None:
        sdg_id = router.add_session([[_text_part("working on it")]])
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"

        result = _run(agent._handle_subagent_message(state, "s1", "go", _body()))

        signals = _signal_phase_results(result["parts"])
        assert len(signals) == 1
        assert agent._get_tool_input(signals[0]).get("phase") == "sdg"

    def test_does_not_duplicate_when_subagent_signals(self, router: _Router) -> None:
        sdg_id = router.add_session(
            [[_tool_part("signal_phase", {"phase": "sdg", "step": "gather_requirements"})]]
        )
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"

        result = _run(agent._handle_subagent_message(state, "s1", "go", _body()))

        assert len(_signal_phase_results(result["parts"])) == 1
        assert state.last_signal_step == "gather_requirements"  # remembered for next turn

    def test_backfill_carries_last_signalled_step(self, router: _Router) -> None:
        sdg_id = router.add_session([[_text_part("still working")]])
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"
        state.last_signal_step = "confirm"

        result = _run(agent._handle_subagent_message(state, "s1", "go", _body()))

        signals = _signal_phase_results(result["parts"])
        assert agent._get_tool_input(signals[0]).get("step") == "confirm"
