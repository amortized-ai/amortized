"""Tests for mid-conversation model-switch detection + context retention (5D)."""

from __future__ import annotations

from typing import Any

from monitor.scorer.matchers import score_auto
from monitor.scorer.model import Run
from monitor.scorer.requires import _requires_met


def _turn(model: str, *present_questions: str, role: str = "orchestrator") -> dict[str, Any]:
    calls = [{"tool": "present_options", "role": role, "question": q} for q in present_questions]
    return {"ts": "2026-01-01T00:00:00", "role": role, "model": model, "tool_calls": calls}


def _run(*turns: dict[str, Any]) -> Run:
    # stamp monotonically increasing ts so counted_turns order is stable
    for i, t in enumerate(turns):
        t["ts"] = f"2026-01-01T00:00:0{i}"
    return Run(session_id="t", turns=list(turns))


def test_model_series_and_switch_count() -> None:
    run = _run(_turn("opus"), _turn("opus"), _turn("sonnet"))
    assert run.models == ["opus", "opus", "sonnet"]
    assert run.model_switches == 1
    assert run.model_switch_index == 2


def test_no_switch() -> None:
    run = _run(_turn("opus"), _turn("opus"))
    assert run.model_switches == 0
    assert run.model_switch_index is None
    assert score_auto(run, {"type": "context_after_switch"}) == "n/a"


def test_model_switched_gate() -> None:
    switched = _run(_turn("opus"), _turn("sonnet"))
    not_switched = _run(_turn("opus"), _turn("opus"))
    gate = {"type": "model_switched"}
    assert _requires_met(switched, gate) is True
    assert _requires_met(not_switched, gate) is False


def test_context_after_switch_wrong_on_reask() -> None:
    # a question decided before the switch is re-asked after it -> context lost
    run = _run(
        _turn("opus", "How many examples should we generate?"),
        _turn("sonnet", "How many examples should we generate?"),
    )
    assert score_auto(run, {"type": "context_after_switch"}) == "wrong"


def test_context_after_switch_met_on_new_question() -> None:
    run = _run(
        _turn("opus", "How many examples should we generate?"),
        _turn("sonnet", "Which base model do you want to train?"),
    )
    assert score_auto(run, {"type": "context_after_switch"}) == "met"


def test_context_after_switch_review_when_postswitch_question_uncaptured() -> None:
    # pre-switch prompt captured; post-switch prompt exists but its question wasn't
    # captured (logging gap) -> review, not n/a (the bug: captured_any hid it).
    pre = _turn("opus", "How many examples should we generate?")
    post = _turn("sonnet")  # switch turn...
    post["tool_calls"] = [{"tool": "present_options", "role": "orchestrator"}]  # no question
    run = _run(pre, post)
    assert score_auto(run, {"type": "context_after_switch"}) == "review"


def test_context_after_switch_met_when_nothing_before() -> None:
    run = _run(
        _turn("opus"),
        _turn("sonnet", "Which base model?"),
    )
    assert score_auto(run, {"type": "context_after_switch"}) == "met"
