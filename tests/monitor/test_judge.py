"""Tests for the LLM-judge path: real prompt build, JSON parse, verdict
normalisation, malformed output, and judge-failure — none monkeypatch the parser,
only the network call."""

from __future__ import annotations

from typing import Any

from monitor.scorer import judge
from monitor.scorer.model import Run


def test_parse_judge_json_variants() -> None:
    # bare object
    assert judge._parse_judge_json('{"a": {"verdict": "met"}}') == {"a": {"verdict": "met"}}
    # fenced ```json block
    fenced = '```json\n{"a": {"verdict": "wrong"}}\n```'
    assert judge._parse_judge_json(fenced) == {"a": {"verdict": "wrong"}}
    # object embedded in prose
    assert judge._parse_judge_json('here: {"a": 1} done') == {"a": 1}
    # malformed -> {}
    assert judge._parse_judge_json("not json at all") == {}
    assert judge._parse_judge_json('{"a": ') == {}


def test_norm_verdict() -> None:
    assert judge._norm_verdict("met") == "met"
    assert judge._norm_verdict("WRONG") == "wrong"
    assert judge._norm_verdict("n/a") == "n/a"
    assert judge._norm_verdict("na") == "n/a"
    assert judge._norm_verdict("n_a") == "n/a"
    assert judge._norm_verdict("unknown") == "review"
    assert judge._norm_verdict(None) == "review"
    assert judge._norm_verdict("garbage") == "review"


def _checklist() -> dict[str, Any]:
    return {
        "rows": [
            {"id": "grounding_data", "aspect": "E", "expected": "x", "adjudicate": "llm_judge"},
            {"id": "no_fabrication", "aspect": "E", "expected": "y", "adjudicate": "llm_judge"},
            {"id": "auto_row", "aspect": "A", "expected": "z", "adjudicate": "auto"},
        ]
    }


def _run() -> Run:
    return Run(
        session_id="s1",
        turns=[
            {
                "ts": "2026-01-01T00:00:00",
                "role": "orchestrator",
                "texts": [{"role": "orchestrator", "text": "hi"}],
                "tool_calls": [],
            }
        ],
    )


def test_llm_judge_runs_parses_verdicts(monkeypatch: Any) -> None:
    captured = {}

    def _fake(model: str):
        def call(system: str, prompt: str) -> str:
            captured["system"] = system
            captured["prompt"] = prompt
            return (
                '{"grounding_data": {"verdict": "met", "reason": "ids match"}, '
                '"no_fabrication": {"verdict": "wrong", "reason": "claimed success"}}'
            )

        return call

    monkeypatch.setattr(judge, "_make_judge_call", _fake)
    verdicts, reasons = judge.llm_judge_runs([_run()], _checklist(), "fake")
    assert verdicts[("s1", "grounding_data")] == "met"
    assert verdicts[("s1", "no_fabrication")] == "wrong"
    assert reasons[("s1", "no_fabrication")] == "claimed success"
    # only llm_judge rows are in the rubric sent to the judge
    assert "grounding_data" in captured["prompt"]
    assert "auto_row" not in captured["prompt"]


def test_llm_judge_runs_malformed_output_defers_to_review(monkeypatch: Any) -> None:
    monkeypatch.setattr(judge, "_make_judge_call", lambda m: lambda s, p: "sorry, no json")
    verdicts, _ = judge.llm_judge_runs([_run()], _checklist(), "fake")
    assert verdicts[("s1", "grounding_data")] == "review"
    assert verdicts[("s1", "no_fabrication")] == "review"


def test_llm_judge_runs_call_failure_is_caught(monkeypatch: Any) -> None:
    def _boom(model: str):
        def call(system: str, prompt: str) -> str:
            raise RuntimeError("judge down")

        return call

    monkeypatch.setattr(judge, "_make_judge_call", _boom)
    # must not raise; unresolved rows stay review
    verdicts, _ = judge.llm_judge_runs([_run()], _checklist(), "fake")
    assert verdicts[("s1", "grounding_data")] == "review"


def test_llm_judge_runs_tolerates_bare_string_verdict(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        judge,
        "_make_judge_call",
        lambda m: lambda s, p: '{"grounding_data": "met", "no_fabrication": "n/a"}',
    )
    verdicts, _ = judge.llm_judge_runs([_run()], _checklist(), "fake")
    assert verdicts[("s1", "grounding_data")] == "met"
    assert verdicts[("s1", "no_fabrication")] == "n/a"
