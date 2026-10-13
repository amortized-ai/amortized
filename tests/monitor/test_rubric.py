"""Tests for graded skill-adherence rubrics (skill-adherence failure point)."""

from __future__ import annotations

from typing import Any

from monitor.scorer import rubric
from monitor.scorer.model import Run


def _run_with(*tool_calls: dict[str, Any]) -> Run:
    turn = {"ts": "2026-01-01T00:00:00", "role": "sdg", "tool_calls": list(tool_calls)}
    return Run(session_id="r", turns=[turn])


def _loaded_distillation(*extra: dict[str, Any]) -> Run:
    read = {
        "tool": "read",
        "role": "sdg",
        "status": "completed",
        "output": "skills/sdg/task-distillation/guide.md",
    }
    return _run_with(read, *extra)


def test_load_rubrics_discovers_all() -> None:
    skills = {r["skill"] for r in rubric.load_rubrics()}
    assert skills == {
        "sdg/task-distillation",
        "sdg/classification",
        "sdg/knowledge-ingestion",
        "training/osft",
    }
    # every step carries id/weight/detect
    for rb in rubric.load_rubrics():
        for step in rb["steps"]:
            assert step.get("id") and step.get("detect") and "weight" in step


def _distillation_rubric() -> dict[str, Any]:
    return next(r for r in rubric.load_rubrics() if r["skill"] == "sdg/task-distillation")


def test_gate_applies_only_when_skill_loaded() -> None:
    rb = _distillation_rubric()
    assert rubric.applies(_loaded_distillation(), rb) is True
    # a run that loaded a different guide does not apply
    other = _run_with(
        {"tool": "read", "role": "sdg", "output": "skills/sdg/classification/guide.md"}
    )
    assert rubric.applies(other, rb) is False


def test_auto_step_scored_llm_steps_default_review() -> None:
    rb = _distillation_rubric()
    run = _loaded_distillation({"tool": "show_prompt", "role": "sdg", "status": "completed"})
    result = rubric.score_rubric(run, rb)  # no judge_map
    steps = {s["step_id"]: s["status"] for s in result["steps"]}
    assert steps["prompt_shown_before_validate"] == "met"  # auto via prompt_review
    assert steps["asks_examples_first"] == "review"  # llm, not judged
    # only the met auto step counts -> 100%
    assert result["adherence_pct"] == 100.0


def test_weighted_adherence_and_skipped() -> None:
    rb = _distillation_rubric()
    run = _loaded_distillation({"tool": "show_prompt", "role": "sdg", "status": "completed"})
    judge_map = {
        "asks_examples_first": ("met", "asked early"),  # w2
        "confirms_output_format": ("missed", "never shown"),  # w2
        "corpus_inferred_silently": ("n/a", ""),  # excluded
        "teacher_name_verbatim": ("review", ""),  # excluded
    }
    result = rubric.score_rubric(run, rb, judge_map)
    # met: asks_examples_first(2) + prompt_shown_before_validate(2) = 4
    # denom: + confirms_output_format missed(2) = 6  -> 66.7
    assert result["adherence_pct"] == 66.7
    assert result["skipped"] == ["confirms_output_format"]


def test_score_all_only_applicable(monkeypatch: Any) -> None:
    rubrics = rubric.load_rubrics()
    run = _loaded_distillation()
    results = rubric.score_all([run], rubrics)
    assert {r["skill"] for r in results} == {"sdg/task-distillation"}


def test_judge_rubrics_merges_verdicts(monkeypatch: Any) -> None:
    rubrics = [r for r in rubric.load_rubrics() if r["skill"] == "sdg/task-distillation"]
    run = _loaded_distillation()

    def _fake_call(model: str):
        def call(system: str, prompt: str) -> str:
            return (
                '{"asks_examples_first": {"verdict": "met", "reason": "ok"}, '
                '"confirms_output_format": {"verdict": "wrong", "reason": "skipped"}}'
            )

        return call

    monkeypatch.setattr(rubric, "_make_judge_call", _fake_call)
    judge = rubric.judge_rubrics([run], rubrics, "fake-model")
    step_map = judge[(run.session_id, "sdg/task-distillation")]
    assert step_map["asks_examples_first"][0] == "met"
    assert step_map["confirms_output_format"][0] == "wrong"
