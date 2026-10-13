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


def test_duplicate_options_uses_full_input_descriptions() -> None:
    call = {
        "tool": "present_options",
        "role": "training",
        "input": {
            "options": [
                {"title": "A", "description": "train on the full dataset"},
                {"title": "B", "description": "train on the full dataset"},
            ]
        },
    }
    assert score_auto(_run(call), {"type": "duplicate_options"}) == "wrong"


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
