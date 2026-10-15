"""Tests for stall detection (announce-without-action; needs a nudge)."""

from __future__ import annotations

from typing import Any

from monitor.scorer.matchers import score_auto
from monitor.scorer.model import Run, turn_is_stall


def _turn(
    *, text: str | None = None, tools: list[dict[str, Any]] | None = None, i: int = 0
) -> dict[str, Any]:
    t: dict[str, Any] = {
        "ts": f"2026-01-01T00:00:0{i}",
        "role": "orchestrator",
        "tool_calls": tools or [],
    }
    if text is not None:
        t["texts"] = [{"role": "orchestrator", "text": text}]
    return t


def _run(*turns: dict[str, Any]) -> Run:
    return Run(session_id="t", turns=list(turns))


def test_turn_is_stall() -> None:
    assert turn_is_stall(_turn(text="I'm preparing the workflow now…")) is True
    assert turn_is_stall(_turn(text="Let me set up the SDG config.")) is True
    # has a tool call -> not a stall (it acted)
    assert turn_is_stall(_turn(text="Preparing…", tools=[{"tool": "validate_sdg_job"}])) is False
    # a real question / concluding statement -> not a stall
    assert turn_is_stall(_turn(text="Which base model would you like?")) is False
    # no text captured -> not a stall
    assert turn_is_stall(_turn()) is False


def test_stall_turns_count() -> None:
    run = _run(
        _turn(text="I'm preparing the workflow…", i=0),
        _turn(text="done", tools=[{"tool": "validate_sdg_job"}], i=1),
        _turn(text="generating the dataset now", i=2),
    )
    assert run.stall_turns == 2


def test_no_stall_wrong_when_nudge_recovered() -> None:
    # turn 0 announces + no tool call; turn 1 (after nudge) emits the card
    run = _run(
        _turn(text="I'm preparing the workflow — one moment.", i=0),
        _turn(text="Here's the SDG config.", tools=[{"tool": "validate_sdg_job"}], i=1),
    )
    assert score_auto(run, {"type": "no_stall"}) == "wrong"


def test_no_stall_met_when_clean() -> None:
    run = _run(
        _turn(text="Here's the plan.", tools=[{"tool": "present_options"}], i=0),
        _turn(text="And the card.", tools=[{"tool": "validate_sdg_job"}], i=1),
    )
    assert score_auto(run, {"type": "no_stall"}) == "met"


def test_no_stall_review_when_trailing_unrecovered() -> None:
    # a stall as the last turn with no later progress -> review (likely abandoned)
    run = _run(
        _turn(text="ok", tools=[{"tool": "validate_sdg_job"}], i=0),
        _turn(text="I'll now prepare the next step…", i=1),
    )
    assert score_auto(run, {"type": "no_stall"}) == "review"


def test_no_stall_review_when_no_texts() -> None:
    run = _run(_turn(tools=[{"tool": "validate_sdg_job"}], i=0), _turn(i=1))
    assert score_auto(run, {"type": "no_stall"}) == "review"


def test_no_stall_na_when_no_turns() -> None:
    assert score_auto(Run(session_id="t"), {"type": "no_stall"}) == "n/a"
