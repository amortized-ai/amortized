"""The interaction gate strips a stacked/duplicate card from the turn result, but the UI
re-derives this turn's cards from its own GET of the opencode history — so a stripped card
would re-render unless the raw history also hides it. The gate records each stripped card's
callID on the session and get_session_messages filters those callIDs out.

This covers the NON-committing strips (stacked present_options, a repeat confirm, a redundant
review) that the _run_turn re-run rail (committing-card holds) does not.
"""

from typing import Any

import pytest

from amortized.api import agent, agent_confirm


def _tool_part(tool: str, input: dict[str, Any], call_id: str | None = None) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "tool", "tool": f"mcp_amortized__{tool}", "input": input}
    if call_id is not None:
        part["callID"] = call_id
    return part


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _fresh_state() -> agent.SessionState:
    return agent.SessionState(orchestrator_id="orch")


def _training_ready() -> agent.SessionState:
    # Training prerequisites met, so the confirm card leads (isolates the single-interaction
    # strip, not the precondition hold / re-run rail).
    state = _fresh_state()
    state.shown_vram = True
    state.confirmed_data = True
    state.offered_choice.add("training")
    return state


def _tool_names(parts: list[dict[str, Any]]) -> list[str]:
    return [agent._tool_name(p) for p in parts if p.get("type") == "tool"]


def _run(coro: Any) -> Any:
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


class TestSuppressionRecording:
    def test_stacked_options_under_confirm_card_callid_recorded(self) -> None:
        result = {
            "parts": [
                _tool_part("validate_training_job", {"algorithm": "sft"}, call_id="card"),
                _tool_part("present_options", {"question": "Submit this job?"}, call_id="opts"),
            ]
        }
        state = _training_ready()
        out = agent._enforce_single_interaction(result, state)
        assert "present_options" not in _tool_names(out["parts"])
        assert state.suppressed_call_ids == {"opts"}

    def test_repeat_confirm_card_callid_recorded(self) -> None:
        result = {
            "parts": [
                _tool_part("validate_training_job", {"algorithm": "osft"}, call_id="card1"),
                _tool_part("present_options", {"question": "ok?"}, call_id="opts"),
                _tool_part("validate_training_job", {"algorithm": "osft"}, call_id="card2"),
            ]
        }
        state = _training_ready()
        agent._enforce_single_interaction(result, state)
        # The LAST valid confirm card is kept (matching the client's #459 last-valid-wins);
        # the stacked options and the earlier duplicate are stripped.
        assert state.suppressed_call_ids == {"opts", "card1"}

    def test_prompt_gate_held_confirm_card_callid_recorded(self) -> None:
        # show_prompt (never reviewed) with a confirm card stacked under it: the review wins,
        # the confirm card is held (soft rail) and must not reach the client.
        result = {
            "parts": [
                _tool_part("show_prompt", {"prompt": "You are an assessor."}, call_id="prompt"),
                _tool_part("validate_sdg_job", {"mode": "create"}, call_id="card"),
            ]
        }
        state = _fresh_state()
        out = agent._enforce_single_interaction(result, state)
        assert _tool_names(out["parts"]) == ["show_prompt"]  # prompt stands alone
        assert state.suppressed_call_ids == {"card"}
        assert state.pending_interaction_steer is not None  # soft steer parked

    def test_kept_card_not_recorded(self) -> None:
        result = {"parts": [_tool_part("present_options", {"question": "which?"}, call_id="keep")]}
        state = _fresh_state()
        agent._enforce_single_interaction(result, state)
        assert state.suppressed_call_ids == set()

    def test_accumulates_across_turns(self) -> None:
        state = _training_ready()
        agent._enforce_single_interaction(
            {
                "parts": [
                    _tool_part("validate_training_job", {}, call_id="card1"),
                    _tool_part("present_options", {"question": "?"}, call_id="opts1"),
                ]
            },
            state,
        )
        agent._enforce_single_interaction(
            {
                "parts": [
                    _tool_part("present_options", {"question": "a?"}, call_id="opts2"),
                    _tool_part("present_options", {"question": "b?"}, call_id="opts3"),
                ]
            },
            state,
        )
        # Turn 1 strips opts1; turn 2 keeps the leading question (opts2), strips opts3.
        assert state.suppressed_call_ids == {"opts1", "opts3"}


class TestSdgTeacherGate:
    """The SDG confirm card requires the no-default teacher model to have been offered via
    present_options first, without double-gating preview→create."""

    def test_sdg_confirm_held_without_teacher_choice(self) -> None:
        result = {"parts": [_tool_part("validate_sdg_job", {"mode": "preview"}, call_id="card")]}
        state = _fresh_state()
        out = agent._enforce_single_interaction(result, state)
        # Card held: not in the client-facing result, callID recorded, model re-steered.
        assert "validate_sdg_job" not in _tool_names(out["parts"])
        assert state.suppressed_call_ids == {"card"}
        assert state.pending_rerun_steer == agent_confirm._SDG_TEACHER_NUDGE
        assert state.deferred_card_job_type == "sdg"  # flagged for re-emit once teacher chosen

    def test_sdg_confirm_renders_after_teacher_choice(self) -> None:
        result = {"parts": [_tool_part("validate_sdg_job", {"mode": "preview"}, call_id="card")]}
        state = _fresh_state()
        state.offered_choice.add("sdg")
        out = agent._enforce_single_interaction(result, state)
        assert _tool_names(out["parts"]) == ["validate_sdg_job"]
        assert state.suppressed_call_ids == set()
        assert state.pending_rerun_steer is None

    def test_sdg_preview_create_not_double_gated(self) -> None:
        # preview and create are two validate_sdg_job calls for the same job. reset=None keeps
        # the teacher signal set, so the create card isn't re-held after the preview renders.
        state = _fresh_state()
        state.offered_choice.add("sdg")
        out_prev = agent._enforce_single_interaction(
            {"parts": [_tool_part("validate_sdg_job", {"mode": "preview"}, call_id="prev")]}, state
        )
        assert _tool_names(out_prev["parts"]) == ["validate_sdg_job"]
        assert "sdg" in state.offered_choice  # not cleared on render (reset=None)
        out_create = agent._enforce_single_interaction(
            {"parts": [_tool_part("validate_sdg_job", {"mode": "create"}, call_id="create")]}, state
        )
        assert _tool_names(out_create["parts"]) == ["validate_sdg_job"]
        assert state.pending_rerun_steer is None

    def test_first_unmet_precondition_teacher_gate(self) -> None:
        card = _tool_part("validate_sdg_job", {"mode": "preview"})
        state = _fresh_state()
        gate = agent_confirm._first_unmet_precondition(state, card)
        assert gate is not None and gate.name == "teacher"
        state.offered_choice.add("sdg")
        assert agent_confirm._first_unmet_precondition(state, card) is None


class TestDeferredCardJobType:
    """_enforce_single_interaction flags the job type of a committing card it DEFERRED so
    _run_turn can park a re-emit steer once the prerequisite is met."""

    def test_held_eval_card_sets_deferred_job_type(self) -> None:
        result = {"parts": [_tool_part("validate_eval_job", {}, call_id="card")]}
        state = _fresh_state()
        agent._enforce_single_interaction(result, state)
        assert state.deferred_card_job_type == "eval"
        assert state.pending_rerun_steer == agent_confirm._EVAL_CHOICE_NUDGE

    def test_stripped_card_behind_question_sets_deferred_job_type(self) -> None:
        # Question leads, card stripped as non-leading (ask-leads path): same deferral.
        result = {
            "parts": [
                _tool_part("present_options", {"question": "judge?"}, call_id="opts"),
                _tool_part("validate_eval_job", {}, call_id="card"),
            ]
        }
        state = _fresh_state()
        agent._enforce_single_interaction(result, state)
        assert state.deferred_card_job_type == "eval"

    def test_rendered_card_clears_deferred_job_type(self) -> None:
        state = _training_ready()
        state.deferred_card_job_type = "stale"  # a prior turn's value must not linger
        result = {"parts": [_tool_part("validate_training_job", {}, call_id="card")]}
        out = agent._enforce_single_interaction(result, state)
        assert _tool_names(out["parts"]) == ["validate_training_job"]  # rendered
        assert state.deferred_card_job_type is None

    def test_no_card_leaves_deferred_job_type_untouched(self) -> None:
        # The gate must not touch the field when no committing card is in the turn (so the
        # rerun's question-only _enforce call preserves the value set by the first call).
        state = _fresh_state()
        state.deferred_card_job_type = "eval"
        agent._enforce_single_interaction(
            {"parts": [_tool_part("present_options", {"question": "?"}, call_id="opts")]}, state
        )
        assert state.deferred_card_job_type == "eval"


class TestGetSessionMessagesFiltersSuppressed:
    def test_suppressed_callids_filtered_from_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _fresh_state()
        state.suppressed_call_ids = {"opts"}
        agent._sessions["conv-x"] = state
        messages = [
            {
                "info": {"role": "assistant"},
                "parts": [
                    _text_part("hello"),
                    _tool_part("validate_training_job", {}, call_id="card"),
                    _tool_part("present_options", {"question": "Submit?"}, call_id="opts"),
                ],
            }
        ]

        async def _proxy_get(_t: str, _p: str, _d: Any, _s: str) -> list[dict[str, Any]]:
            return messages

        monkeypatch.setattr(agent, "_proxy_get", _proxy_get)
        try:
            out = _run(agent.get_session_messages("conv-x"))
        finally:
            del agent._sessions["conv-x"]
        assert _tool_names(out[0]["parts"]) == ["validate_training_job"]
        assert any(p.get("type") == "text" for p in out[0]["parts"])  # text untouched

    def test_passthrough_when_nothing_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = _fresh_state()
        agent._sessions["conv-y"] = state
        messages = [
            {
                "info": {"role": "assistant"},
                "parts": [_tool_part("present_options", {"question": "?"}, call_id="opts")],
            }
        ]

        async def _proxy_get(_t: str, _p: str, _d: Any, _s: str) -> list[dict[str, Any]]:
            return messages

        monkeypatch.setattr(agent, "_proxy_get", _proxy_get)
        try:
            out = _run(agent.get_session_messages("conv-y"))
        finally:
            del agent._sessions["conv-y"]
        assert _tool_names(out[0]["parts"]) == ["present_options"]
