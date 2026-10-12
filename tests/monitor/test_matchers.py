"""Unit tests for the deterministic matcher registry (Phase 0 extraction)."""

from __future__ import annotations

from typing import Any

from monitor.scorer.matchers import MATCHERS, score_auto, score_run
from monitor.scorer.model import Run


def _run(*tool_calls: dict[str, Any], completion: dict[str, Any] | None = None) -> Run:
    """A one-turn Run carrying the given tool calls (role defaults to orchestrator)."""
    turn = {"ts": "2026-01-01T00:00:00", "role": "orchestrator", "tool_calls": list(tool_calls)}
    return Run(session_id="t", turns=[turn], completion=completion)


# --- registry wiring -------------------------------------------------------


def test_all_builtin_matchers_registered() -> None:
    expected = {
        "tool_called",
        "chained",
        "validate_ok",
        "tool_before",
        "recovery",
        "completion",
        "scores_present",
        "solo_delegation",
        "no_error_calls",
        "prompt_review",
        "eval_phase",
        "model_choice",
        "judge_valid",
    }
    assert expected <= set(MATCHERS)


def test_unknown_match_type_defers_to_review() -> None:
    assert score_auto(_run(), {"type": "does_not_exist"}) == "review"


# --- tool_called -----------------------------------------------------------


def test_tool_called_met_with_where_and_output_contains() -> None:
    run = _run({"tool": "read", "role": "sdg", "output": "skills/sdg/foo/guide.md"})
    m = {
        "type": "tool_called",
        "tool": "read",
        "where": {"role": "sdg"},
        "output_contains": "skills/sdg/",
    }
    assert score_auto(run, m) == "met"


def test_tool_called_missed_when_absent() -> None:
    assert score_auto(_run(), {"type": "tool_called", "tool": "read"}) == "missed"


def test_tool_called_any_of_tools() -> None:
    run = _run({"tool": "show_vram_estimate", "role": "training"})
    m = {
        "type": "tool_called",
        "tools": ["estimate_training_resources", "show_vram_estimate"],
        "where": {"role": "training"},
    }
    assert score_auto(run, m) == "met"


def test_tool_called_na_if_reused() -> None:
    run = _run({"tool": "get_job", "output": "reused prior teacher"})
    m = {
        "type": "tool_called",
        "tool": "get_model_pricing",
        "na_if_reused": {"tool": "get_job", "output_contains": "reused"},
    }
    assert score_auto(run, m) == "n/a"


# --- chained / validate_ok -------------------------------------------------


def test_chained_met_and_missed() -> None:
    m = {"type": "chained", "tool": "validate_training_job", "any_of": ["parent_job_id"]}
    assert score_auto(_run({"tool": "validate_training_job", "parent_job_id": "x"}), m) == "met"
    assert score_auto(_run({"tool": "validate_training_job"}), m) == "missed"


def test_validate_ok_met_wrong_missed() -> None:
    m = {"type": "validate_ok", "tool": "validate_sdg_job"}
    assert (
        score_auto(_run({"tool": "validate_sdg_job", "status": "completed", "output": "ok"}), m)
        == "met"
    )
    assert (
        score_auto(_run({"tool": "validate_sdg_job", "status": "error", "output": "bad"}), m)
        == "wrong"
    )
    assert score_auto(_run(), m) == "missed"


# --- tool_before -----------------------------------------------------------


def test_tool_before_met_wrong_missed() -> None:
    m = {"type": "tool_before", "before": "validate_training_job", "after": "validate_eval_job"}
    ok = _run({"tool": "validate_training_job"}, {"tool": "validate_eval_job"})
    assert score_auto(ok, m) == "met"
    bad = _run({"tool": "validate_eval_job"})
    assert score_auto(bad, m) == "wrong"
    assert score_auto(_run({"tool": "validate_training_job"}), m) == "missed"


def test_tool_before_where_field_missing_is_review() -> None:
    # before spec filters on mode=preview, but the logged call has no mode -> review.
    m = {
        "type": "tool_before",
        "before": {"tool": "validate_sdg_job", "where": {"mode": "preview"}},
        "after": {"tool": "validate_sdg_job", "where": {"mode": "create"}},
    }
    run = _run({"tool": "validate_sdg_job"}, {"tool": "validate_sdg_job"})
    assert score_auto(run, m) == "review"


# --- recovery / no_error_calls / completion --------------------------------


def test_recovery() -> None:
    m = {"type": "recovery"}
    assert score_auto(_run({"tool": "a", "status": "completed"}), m) == "n/a"
    assert (
        score_auto(_run({"tool": "a", "status": "error"}, {"tool": "b", "status": "completed"}), m)
        == "met"
    )
    assert score_auto(_run({"tool": "a", "status": "error"}), m) == "wrong"


def test_no_error_calls() -> None:
    m = {"type": "no_error_calls"}
    assert score_auto(_run({"tool": "a", "status": "completed"}), m) == "met"
    assert score_auto(_run({"tool": "a", "status": "error"}), m) == "wrong"


def test_completion_outcome() -> None:
    m = {"type": "completion", "outcome": "success"}
    assert score_auto(_run(completion={"outcome": "success"}), m) == "met"
    assert score_auto(_run(completion={"outcome": "gave_up"}), m) == "wrong"
    assert score_auto(_run(), m) == "missed"


# --- scores_present / solo_delegation --------------------------------------


def test_scores_present() -> None:
    m = {"type": "scores_present", "tool": "get_eval_results"}
    assert (
        score_auto(_run({"tool": "get_eval_results", "output": '{"scores":{"acc":0.9}}'}), m)
        == "met"
    )
    assert (
        score_auto(_run({"tool": "get_eval_results", "output": '{"scores":{"acc":null}}'}), m)
        == "wrong"
    )
    assert score_auto(_run(), m) == "missed"


def test_solo_delegation() -> None:
    m = {"type": "solo_delegation"}
    assert score_auto(_run({"tool": "delegate_to_subagent", "solo": True}), m) == "met"
    assert score_auto(_run({"tool": "delegate_to_subagent", "solo": False}), m) == "wrong"
    # solo flag absent -> log predates signal -> review
    assert score_auto(_run({"tool": "delegate_to_subagent"}), m) == "review"
    assert score_auto(_run(), m) == "missed"


# --- judge_valid / eval_phase / model_choice -------------------------------


def test_judge_valid() -> None:
    m = {"type": "judge_valid"}
    catalog = {"tool": "list_models", "output": '{"models":[{"name":"gpt-oss"}]}'}
    assert score_auto(_run(catalog, {"tool": "validate_eval_job", "judge": "gpt-oss"}), m) == "met"
    assert (
        score_auto(_run(catalog, {"tool": "validate_eval_job", "judge": "openai/gpt-oss-120b"}), m)
        == "wrong"
    )
    assert (
        score_auto(_run({"tool": "validate_eval_job", "judge": "gpt-oss"}), m) == "review"
    )  # no catalog
    assert score_auto(_run(), m) == "missed"


def test_eval_phase() -> None:
    m = {"type": "eval_phase"}
    base = {"tool": "validate_eval_job", "role": "eval"}
    assert (
        score_auto(_run(base, {"tool": "signal_phase", "role": "eval", "phase": "eval"}), m)
        == "met"
    )
    assert (
        score_auto(_run(base, {"tool": "signal_phase", "role": "eval", "phase": "training"}), m)
        == "wrong"
    )
    assert score_auto(_run(base), m) == "missed"
    assert score_auto(_run(), m) == "n/a"


def test_model_choice() -> None:
    m = {"type": "model_choice"}
    base = {"tool": "validate_training_job", "role": "training"}
    picks = {
        "tool": "present_options",
        "role": "training",
        "question": "Which base model?",
        "options": ["Qwen2.5"],
    }
    assert score_auto(_run(base, picks), m) == "met"
    assert score_auto(_run(), m) == "n/a"


# --- score_run gating ------------------------------------------------------


def test_score_run_gates_on_requires() -> None:
    checklist = {
        "rows": [
            {
                "id": "gated",
                "aspect": "G",
                "stage": "s",
                "adjudicate": "llm_judge",
                "requires": {
                    "type": "skill_loaded",
                    "role": "sdg",
                    "path_contains": "skills/sdg/x/",
                },
            },
            {
                "id": "auto_row",
                "aspect": "A",
                "stage": "s",
                "adjudicate": "auto",
                "match": {"type": "completion", "outcome": "success"},
            },
        ]
    }
    run = _run(completion={"outcome": "success"})
    scores = score_run(run, checklist)
    assert scores["gated"] == "n/a"  # skill not loaded -> gated out
    assert scores["auto_row"] == "met"
