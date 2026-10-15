"""Tests for lexical similarity + the duplicate/repeated-option matchers (5A)."""

from __future__ import annotations

from typing import Any

from monitor.scorer.matchers import score_auto
from monitor.scorer.model import Run
from monitor.scorer.similarity import similarity


def _run(*tool_calls: dict[str, Any]) -> Run:
    turn = {"ts": "2026-01-01T00:00:00", "role": "orchestrator", "tool_calls": list(tool_calls)}
    return Run(session_id="t", turns=[turn])


def _po(question: str, options: list[str]) -> dict[str, Any]:
    return {"tool": "present_options", "role": "sdg", "question": question, "options": options}


def test_similarity_bounds() -> None:
    assert similarity("hello world", "hello world") == 1.0
    assert similarity("alpha", "beta") == 0.0
    assert similarity("", "") == 0.0
    s = similarity("use the full dataset", "use the whole dataset")
    assert 0.0 < s < 1.0


# --- duplicate_options -----------------------------------------------------


def test_duplicate_options_na_when_none() -> None:
    assert score_auto(_run(), {"type": "duplicate_options"}) == "n/a"


def test_duplicate_options_met_when_distinct() -> None:
    run = _run(_po("Which base model?", ["Qwen2.5-7B", "Llama-3-8B"]))
    assert score_auto(run, {"type": "duplicate_options"}) == "met"


def test_duplicate_options_wrong_on_near_identical() -> None:
    run = _run(_po("How many?", ["Generate 2000 examples", "Generate 2000 examples now"]))
    assert score_auto(run, {"type": "duplicate_options"}) == "wrong"


def test_duplicate_options_review_when_not_captured() -> None:
    # present_options call with no options field at all -> can't check -> review
    run = _run({"tool": "present_options", "role": "sdg", "question": "pick"})
    assert score_auto(run, {"type": "duplicate_options"}) == "review"


def test_duplicate_options_wrong_on_identical_titles() -> None:
    call = {
        "tool": "present_options",
        "role": "training",
        "input": {
            "options": [
                {"title": "Use the full dataset", "description": "d1"},
                {"title": "Use the full dataset", "description": "d2"},
            ]
        },
    }
    assert score_auto(_run(call), {"type": "duplicate_options"}) == "wrong"


def test_duplicate_options_distinct_titles_shared_description() -> None:
    # boilerplate description must NOT make distinct model cards look duplicate
    call = {
        "tool": "present_options",
        "role": "training",
        "input": {
            "options": [
                {"title": "Qwen2.5-7B", "description": "a strong general-purpose model"},
                {"title": "Llama-3-8B", "description": "a strong general-purpose model"},
            ]
        },
    }
    assert score_auto(_run(call), {"type": "duplicate_options"}) == "met"


def test_duplicate_options_numeric_sizes_are_distinct() -> None:
    # size choices differing only by a number are distinct, not duplicates -> met
    run = _run(_po("How many examples?", ["Generate 1000 examples", "Generate 2000 examples"]))
    assert score_auto(run, {"type": "duplicate_options"}) == "met"


# --- repeated_decision -----------------------------------------------------


def test_repeated_decision_na_with_one_prompt() -> None:
    run = _run(_po("Which base model?", ["Qwen", "Llama"]))
    assert score_auto(run, {"type": "repeated_decision"}) == "n/a"


def test_repeated_decision_wrong_on_reask() -> None:
    run = _run(
        _po("How many examples should we generate?", ["1000", "2000"]),
        _po("How many examples should we generate?", ["1000", "4000"]),
    )
    assert score_auto(run, {"type": "repeated_decision"}) == "wrong"


def test_repeated_decision_met_on_distinct_questions() -> None:
    run = _run(
        _po("Which base model?", ["Qwen", "Llama"]),
        _po("Which urgency levels?", ["three", "four"]),
    )
    assert score_auto(run, {"type": "repeated_decision"}) == "met"


def test_repeated_decision_review_when_questions_missing() -> None:
    run = _run(
        {"tool": "present_options", "role": "sdg"},
        {"tool": "present_options", "role": "sdg"},
    )
    assert score_auto(run, {"type": "repeated_decision"}) == "review"


def test_repeated_decision_scoped_by_stage() -> None:
    # the SAME question in two different delegation segments (training-set size vs
    # held-out eval-set size) is a fresh decision per stage -> not flagged.
    delegate = {"tool": "delegate_to_subagent", "role": "orchestrator", "target": "training"}
    delegate2 = {"tool": "delegate_to_subagent", "role": "orchestrator", "target": "eval"}
    run = _run(
        delegate,
        _po("How many examples should we generate?", ["1000", "2000"]),
        delegate2,
        _po("How many examples should we generate?", ["1000", "2000"]),
    )
    assert score_auto(run, {"type": "repeated_decision"}) == "met"


def test_repeated_decision_flags_reask_within_stage() -> None:
    # same question twice within ONE segment (no delegate between) -> wrong
    run = _run(
        _po("How many examples should we generate?", ["1000", "2000"]),
        _po("How many examples should we generate?", ["1000", "4000"]),
    )
    assert score_auto(run, {"type": "repeated_decision"}) == "wrong"
