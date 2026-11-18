"""The subagent stall nudge — the live consumer of the pure adjudicator.

When a subagent turn stalls on an owed tool edge — prose instead of the required transition — the
proxy re-prompts ONCE on the system channel naming the owed tool, so a weak driver (GLM-flash)
self-corrects by CALLING it. It consumes the pure `adjudicate` library (api/phases.py) and runs
before completion/help detection so a nudged request_orchestrator_help flows through the normal
hand-off path.
"""

import asyncio
from typing import Any

import pytest

from amortized.api import agent


def _tool_part(tool: str, inp: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool", "tool": f"mcp_amortized__{tool}", "input": inp}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _signal(step: str) -> dict[str, Any]:
    return _tool_part("signal_phase", {"phase": "eval", "step": step})


class _Router:
    def __init__(self) -> None:
        self._n = 0
        self.last_parts: dict[str, list[dict[str, Any]]] = {}
        self.sent: list[tuple[str, str, str | None, str | None]] = []
        self.scripts: dict[str, list[list[dict[str, Any]]]] = {}

    def add(self, script: list[list[dict[str, Any]]] | None = None) -> str:
        self._n += 1
        sid = f"s{self._n}"
        self.scripts[sid] = [list(p) for p in (script or [])]
        return sid

    async def send(
        self,
        session_id: str,
        text: str,
        agent: str | None = None,
        model: Any = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        self.sent.append((session_id, text, agent, system))
        script = self.scripts.get(session_id) or []
        parts = script.pop(0) if script else []
        self.last_parts[session_id] = parts
        return {"info": {"id": session_id}, "parts": parts}

    async def fetch(self, session_id: str) -> list[dict[str, Any]]:
        return self.last_parts.get(session_id) or []

    def system_to(self, session_id: str) -> str:
        return next((s or "" for sid, _t, _a, s in self.sent if sid == session_id), "")


@pytest.fixture()
def router(monkeypatch: pytest.MonkeyPatch) -> _Router:
    r = _Router()
    monkeypatch.setattr(agent, "_proxy_send_message", r.send)
    monkeypatch.setattr(agent, "_fetch_all_assistant_parts", r.fetch)
    return r


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


def _body(event: str | None = None) -> agent.MessageRequest:
    return agent.MessageRequest(parts=[agent.MessagePart(type="text", text="hi")], event=event)


def _state(router: _Router, sub: str, step: str) -> agent.SessionState:
    state = agent.SessionState(orchestrator_id=router.add())
    state.subagent_id = sub
    state.subagent_target = "eval"
    state.last_signal_step = step
    return state


_CORRECTED = [[_signal("held-out-request-help"), _tool_part("request_orchestrator_help", {})]]


def test_stall_nudges_and_returns_the_corrected_turn(router: _Router) -> None:
    sub = router.add(_CORRECTED)  # the re-invoke yields the corrected hand-off
    state = _state(router, sub, "held-out-request-help")
    stalled = [
        _text_part("I'll hand this off to get a held-out set."),
        _signal("held-out-request-help"),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), stalled))

    system = router.system_to(sub)
    assert "request_orchestrator_help" in system
    assert "do not describe" in system.lower()
    # the corrected (re-fetched) parts are returned, now carrying the real hand-off call
    assert any(agent._tool_name(p) == "request_orchestrator_help" for p in out)


def test_owed_tool_present_sends_no_nudge(router: _Router) -> None:
    sub = router.add()
    state = _state(router, sub, "held-out-request-help")
    ok_parts = [_signal("held-out-request-help"), _tool_part("request_orchestrator_help", {})]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), ok_parts))

    assert out is ok_parts  # unchanged
    assert not any(sid == sub for sid, *_ in router.sent)  # nothing re-sent


def test_not_solo_handoff_is_honored(router: _Router) -> None:
    # A bundled hand-off — the real request_orchestrator_help call PLUS narrating prose (what
    # GLM-flash emits) — is NOT re-driven. Even though the adjudicator rates it `not_solo`, the
    # empty-user re-drive would ERASE the turn and discard the real call, which
    # _handle_subagent_message is about to route. The nudge consumer exempts any turn already
    # carrying a routable transition (completion / hand-back), so "preserved" == "routed". The
    # bundled prose is cleaned off in the hand-off path, not here.
    sub = router.add(_CORRECTED)
    state = _state(router, sub, "held-out-request-help")
    not_solo = [
        _text_part("Handing this off now."),
        _signal("held-out-request-help"),
        _tool_part("request_orchestrator_help", {}),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), not_solo))

    assert out is not_solo  # returned unchanged — the hand-off is preserved
    assert not any(sid == sub for sid, *_ in router.sent)  # nothing re-driven


def test_lenient_phase_is_not_nudged(router: _Router) -> None:
    sub = router.add()
    state = _state(router, sub, "held-out-ask-count")  # lenient: a free-text question
    prose = [
        _text_part("How many eval records would you like? (default 100)"),
        _signal("held-out-ask-count"),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), prose))

    assert out is prose
    assert not any(sid == sub for sid, *_ in router.sent)


def test_skipped_signal_is_not_judged_off_the_stale_key(router: _Router) -> None:
    # A NON-CARD work turn that skips signal_phase, so last_signal_step ("validate") is a PRIOR
    # turn's step. Adjudicating off that stale key would wrongly owe `validate_eval_job` and
    # re-drive the turn to re-validate; it is left exactly as-is instead.
    sub = router.add([[_signal("validate"), _tool_part("validate_eval_job", {})]])
    state = _state(router, sub, "validate")
    skipped = [
        _text_part("Let me check the memory footprint."),
        _tool_part("show_vram_estimate", {}),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), skipped))

    assert out is skipped  # the turn the user sees is the turn GLM wrote
    assert not any(sid == sub for sid, *_ in router.sent)  # no hidden re-prompt


def test_orchestrator_missing_signal_is_not_guarded(router: _Router) -> None:
    # Only subagents stamp signal_phase; a non-subagent target must fall through to normal
    # adjudication (which is lenient for an unknown key) with no missing-signal nudge.
    sub = router.add()
    state = _state(router, sub, "validate")
    state.subagent_target = "morty"
    parts = [_text_part("What would you like to do next?")]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), parts))

    assert out is parts
    assert not any(sid == sub for sid, *_ in router.sent)


def test_adjudicates_on_this_turns_signal_not_the_lagging_state(router: _Router) -> None:
    # Regression: state.last_signal_step is written by _ensure_phase_signal AFTER this in-turn
    # adjudication, so it lags one turn. Here the prior turn left it on a gather step
    # ("eval-dataset", owes present_options) while THIS turn moves to "validate" and emits the
    # preview card (validate_eval_job). Keying on the stale state field would judge the card a
    # wrong_edge and nudge it away (dropping it); keying on the turn's own signal → ok, no nudge.
    sub = router.add()
    state = _state(router, sub, "eval-dataset")  # stale prior-turn cursor
    turn = [_signal("validate"), _tool_part("validate_eval_job", {})]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), turn))

    assert out is turn  # no nudge: the preview card is preserved, not dropped
    assert not any(sid == sub for sid, *_ in router.sent)


def test_card_stamp_differing_from_signalled_phase_is_not_batching(router: _Router) -> None:
    # Regression: the subagent signals the workflow phase on signal_phase (coarse slug,
    # "gather") but stamps the present_options card with the finer skill-step slug
    # ("training-method"). These are two vocabularies for ONE logical place. Unioning the
    # signal_phase step into present_steps made this read as two steps → false `batching`
    # → the owed method-options card was nudged away and never rendered. Only CARD stamps
    # (present_options) count toward batching.
    sub = router.add()
    state = _state(router, sub, "gather")
    turn = [_signal("gather"), _tool_part("present_options", {"step": "training-method"})]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), turn))

    assert out is turn  # no nudge: the options card is preserved, not dropped
    assert not any(sid == sub for sid, *_ in router.sent)


def test_two_distinct_option_cards_in_one_turn_is_still_batching(router: _Router) -> None:
    # The real batching case still fires: two present_options cards for DIFFERENT steps in
    # one turn (the model racing two asks) → nudge to present one step at a time.
    sub = router.add([[_signal("gather"), _tool_part("present_options", {"step": "a"})]])
    state = _state(router, sub, "gather")
    turn = [
        _tool_part("present_options", {"step": "a"}),
        _tool_part("present_options", {"step": "b"}),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), turn))

    assert any(sid == sub for sid, *_ in router.sent)  # nudged
    assert out is not turn  # re-fetched


def test_card_turn_missing_signal_is_not_nudged_away(router: _Router) -> None:
    # Regression (live): a review revision turn emitted show_prompt + present_options (the owed
    # cards) but skipped signal_phase. The missing-signal re-prompt fired, re-drove the turn, and
    # stranded the rendered prompt card ("where is the prompt card?"). A card-bearing turn did its
    # owed work — the card must survive.
    sub = router.add()
    state = _state(router, sub, "report")  # prior-turn cursor (owes present_options)
    turn = [
        _tool_part("show_prompt", {}),
        _tool_part("present_options", {"step": "rubric-review"}),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), turn))

    assert out is turn  # cards preserved, not re-driven
    assert not any(sid == sub for sid, *_ in router.sent)  # no guard/stall nudge


def test_prose_only_missing_signal_is_not_reprompted(router: _Router) -> None:
    # Regression (live): GLM-flash answered "Software" at `metrics` with find_benchmarks + the
    # benchmark list, skipping signal_phase (same step). A re-prompt for the signal ran a hidden
    # turn whose reply replaced the list, so the user never saw it. A prose answer that skips the
    # signal is shown as written.
    sub = router.add([[_signal("metrics"), _text_part("Which one should I run?")]])
    state = _state(router, sub, "metrics")
    prose = [
        _tool_part("find_benchmarks", {"query": "software"}),
        _text_part("Here are the software benchmarks: HumanEval, MBPP, ..."),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), prose))

    assert out is prose
    assert not any(sid == sub for sid, *_ in router.sent)


def test_off_vocab_signal_prose_only_is_treated_like_a_skip(router: _Router) -> None:
    # #1a: a prose-only turn that signals a slug which is NOT a recognized `## Workflow` section or
    # skill `### Step` (here "gather-requirements" vs a real eval slug) keys to no obligation and
    # names no section — the adjudicator is as blind to it as to a skipped signal — so it gets the
    # same treatment: shown as written, no re-prompt.
    sub = router.add(
        [[_signal("eval-dataset"), _tool_part("present_options", {"step": "eval-dataset"})]]
    )
    state = _state(router, sub, "eval-dataset")
    prose = [
        _text_part("Let me gather the requirements for your eval."),
        _signal("gather-requirements"),  # off-vocab: not an eval section slug
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), prose))

    assert out is prose
    assert not any(sid == sub for sid, *_ in router.sent)


_STEP_MARKERS = [
    {"type": "step-start"},
    {"type": "reasoning", "text": "..."},
    {"type": "step-finish"},
]


def test_empty_reply_is_reprompted(router: _Router) -> None:
    # Live: GLM-flash sometimes ends a turn with reasoning only — no text, no tool call — so the
    # user sees an empty reply. Nothing is shown, so a re-prompt hides nothing.
    sub = router.add([[_signal("validate"), _tool_part("validate_eval_job", {})]])
    state = _state(router, sub, "metrics")

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), list(_STEP_MARKERS)))

    assert "EMPTY RESPONSE" in router.system_to(sub)
    assert any(agent._tool_name(p) == "validate_eval_job" for p in out)  # re-fetched reply


def test_signal_only_reply_is_reprompted_as_empty(router: _Router) -> None:
    # A bare signal_phase renders nothing either — still an empty reply, even at a lenient step.
    sub = router.add([[_text_part("Which area should the benchmark cover?")]])
    state = _state(router, sub, "metrics")

    _run(
        agent._maybe_nudge_subagent_stall(state, "s", _body(), [*_STEP_MARKERS, _signal("metrics")])
    )

    assert "EMPTY RESPONSE" in router.system_to(sub)


def test_empty_reply_on_a_resume_turn_is_reprompted(router: _Router) -> None:
    sub = router.add([[_text_part("Welcome back — here is where we are.")]])
    state = _state(router, sub, "metrics")

    _run(
        agent._maybe_nudge_subagent_stall(
            state, "s", _body(), list(_STEP_MARKERS), guard_missing_signal=False
        )
    )

    assert "EMPTY RESPONSE" in router.system_to(sub)


def test_failed_fetch_is_not_mistaken_for_an_empty_reply(router: _Router) -> None:
    # `_fetch_all_assistant_parts` returns [] when the GET fails; a real reply always carries step
    # markers. Re-prompting on [] would send GLM an extra message after a turn it may have answered.
    sub = router.add()
    state = _state(router, sub, "metrics")

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), []))

    assert out == []
    assert not any(sid == sub for sid, *_ in router.sent)


def test_tool_only_reply_is_not_empty(router: _Router) -> None:
    # A card with no prose is a real reply (the user acts on the card).
    sub = router.add()
    state = _state(router, sub, "eval-dataset")
    turn = [
        *_STEP_MARKERS,
        _signal("eval-dataset"),
        _tool_part("present_options", {"step": "eval-dataset"}),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), turn))

    assert out is turn
    assert not any(sid == sub for sid, *_ in router.sent)


def test_signalled_stall_is_still_nudged_after_an_unsignalled_turn(router: _Router) -> None:
    # Dropping the missing-signal re-prompt leaves the stall re-prompt intact: a turn that DOES
    # signal `validate` but narrates instead of calling validate_eval_job is still re-driven.
    sub = router.add([[_signal("validate"), _tool_part("validate_eval_job", {})]])
    state = _state(router, sub, "metrics")
    stalled = [_signal("validate"), _text_part("I'll validate the config now.")]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), stalled))

    assert "validate_eval_job" in router.system_to(sub)
    assert any(agent._tool_name(p) == "validate_eval_job" for p in out)


def test_resume_turn_without_signal_is_judged_on_the_restored_step(router: _Router) -> None:
    # On a delegate/resume turn (`guard_missing_signal=False`) the step the server just set is the
    # key, so an unsignalled narrated hand-off is still caught (#534 P2c) — only the steady-state
    # skip changed.
    sub = router.add(_CORRECTED)
    state = _state(router, sub, "held-out-request-help")
    stalled = [_text_part("I'll hand this off to get a held-out set.")]

    out = _run(
        agent._maybe_nudge_subagent_stall(state, "s", _body(), stalled, guard_missing_signal=False)
    )

    assert "request_orchestrator_help" in router.system_to(sub)
    assert any(agent._tool_name(p) == "request_orchestrator_help" for p in out)


def test_off_vocab_signal_with_a_card_is_preserved(router: _Router) -> None:
    # #1a is still gated on the card rules: an off-vocab signal that nonetheless rendered a card is
    # NOT re-driven (that would strand the card) — the lenient turn is left exactly as-is.
    sub = router.add()
    state = _state(router, sub, "eval-dataset")
    turn = [
        _signal("gather-requirements"),  # off-vocab
        _tool_part("present_options", {"step": "eval-dataset"}),
    ]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), turn))

    assert out is turn  # card preserved, no guard/stall nudge
    assert not any(sid == sub for sid, *_ in router.sent)


def test_card_on_wrong_edge_is_preserved_not_dropped(router: _Router) -> None:
    # Even when a stale key makes a card-bearing turn look like a wrong_edge, the card is kept —
    # only genuine `batching` is worth dropping a card to re-ask.
    sub = router.add()
    state = _state(router, sub, "validate")  # owes validate_eval_job
    turn = [_signal("validate"), _tool_part("show_prompt", {})]  # a card, not the owed edge

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(), turn))

    assert out is turn  # card preserved despite wrong_edge
    assert not any(sid == sub for sid, *_ in router.sent)


def test_job_complete_event_skips_nudge(router: _Router) -> None:
    sub = router.add()
    state = _state(router, sub, "held-out-request-help")
    stalled = [_text_part("I'll hand this off."), _signal("held-out-request-help")]

    out = _run(agent._maybe_nudge_subagent_stall(state, "s", _body(event="job_complete"), stalled))

    assert out is stalled
    assert not any(sid == sub for sid, *_ in router.sent)
