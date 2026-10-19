"""The interaction gate strips a stacked/duplicate card from the turn result, but the UI
re-derives this turn's cards from its own GET of the opencode history — so a stripped card
would re-render unless the raw history also hides it. The gate records each stripped card's
callID on the session and get_session_messages filters those callIDs out.

This covers the NON-committing strips (stacked present_options, a repeat confirm, a redundant
review) that the _run_turn re-run rail (committing-card holds) does not.
"""

from typing import Any

import pytest

from amortized.api import agent


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


def _msg(role: str, parts: list[dict[str, Any]], mid: str) -> dict[str, Any]:
    return {"info": {"role": role, "id": mid}, "parts": parts}


class TestInjectedSteerHiddenFromHistory:
    """A correction gate re-prompts the model by POSTing a user-role steer mid-turn. The client
    scopes 'this turn' to everything after the last user message, so that injected steer would
    shift the boundary past a confirm card the model already emitted — the card then renders as
    stale history, not an actionable card. The proxy records those steer ids and hides them."""

    def _history(self) -> list[dict[str, Any]]:
        # Real user turn → model emits the confirm card → gate injects a steer → model corrects.
        return [
            _msg("user", [_text_part("set up training")], "u-real"),
            _msg("assistant", [_tool_part("validate_training_job", {}, call_id="card")], "a-card"),
            _msg("user", [_text_part("[JOB-CLAIM CHECK ...]")], "u-steer"),
            _msg("assistant", [_text_part("Correction: not created; card above.")], "a-fix"),
        ]

    def test_steer_message_dropped_so_card_stays_in_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _fresh_state()
        state.suppressed_user_msg_ids = {"u-steer"}
        agent._sessions["conv-z"] = state
        history = self._history()

        async def _proxy_get(_t: str, _p: str, _d: Any, _s: str) -> list[dict[str, Any]]:
            return history

        monkeypatch.setattr(agent, "_proxy_get", _proxy_get)
        try:
            out = _run(agent.get_session_messages("conv-z"))
        finally:
            del agent._sessions["conv-z"]
        ids = [m["info"]["id"] for m in out]
        assert "u-steer" not in ids  # the injected steer is hidden
        # The last remaining user message is the REAL turn, so the card is in the current turn.
        last_user = [m for m in out if m["info"]["role"] == "user"][-1]
        assert last_user["info"]["id"] == "u-real"
        card_msg = next(m for m in out if m["info"]["id"] == "a-card")
        assert _tool_names(card_msg["parts"]) == ["validate_training_job"]

    def test_passthrough_when_no_steers_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _fresh_state()  # suppressed_user_msg_ids empty
        agent._sessions["conv-z2"] = state
        history = self._history()

        async def _proxy_get(_t: str, _p: str, _d: Any, _s: str) -> list[dict[str, Any]]:
            return history

        monkeypatch.setattr(agent, "_proxy_get", _proxy_get)
        try:
            out = _run(agent.get_session_messages("conv-z2"))
        finally:
            del agent._sessions["conv-z2"]
        assert len(out) == 4  # nothing hidden

    def test_refetch_after_steer_records_injected_user_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The just-POSTed steer is the LAST user message; _refetch_after_steer captures its id
        # and returns the corrected assistant parts (from _fetch_all_assistant_parts, unchanged).
        state = _fresh_state()
        full = self._history()
        corrected = [_text_part("Correction: not created; card above.")]

        async def _fetch_all_messages(_s: str, cache: Any = None) -> list[dict[str, Any]]:
            return full

        async def _fetch_parts(_s: str) -> list[dict[str, Any]]:
            return corrected

        monkeypatch.setattr(agent, "_fetch_all_messages", _fetch_all_messages)
        monkeypatch.setattr(agent, "_fetch_all_assistant_parts", _fetch_parts)
        parts = _run(agent._refetch_after_steer(state, "sess"))
        assert state.suppressed_user_msg_ids == {"u-steer"}
        assert parts == corrected

    def test_refetch_after_steer_no_user_msg_records_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A transient empty fetch (no user message visible) must not record a bogus id.
        state = _fresh_state()

        async def _fetch_all_messages(_s: str, cache: Any = None) -> list[dict[str, Any]]:
            return []

        async def _fetch_parts(_s: str) -> list[dict[str, Any]]:
            return []

        monkeypatch.setattr(agent, "_fetch_all_messages", _fetch_all_messages)
        monkeypatch.setattr(agent, "_fetch_all_assistant_parts", _fetch_parts)
        _run(agent._refetch_after_steer(state, "sess"))
        assert state.suppressed_user_msg_ids == set()
