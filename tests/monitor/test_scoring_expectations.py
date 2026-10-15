"""Hand-authored expected scores for the fixture runs.

Unlike test_scoring_golden (a regenerated change-detector), these are reasoned by
hand from the fixture logs, so they're a real correctness oracle for the newer
matchers (duplicate options, model-switch, job lifecycle, judge_valid).
"""

from __future__ import annotations

from pathlib import Path

from monitor.scorer.matchers import score_run
from monitor.scorer.model import load_checklist, load_runs

_FIXTURES = Path(__file__).parent / "fixtures"


def _scores() -> dict[str, dict[str, str]]:
    runs = load_runs(_FIXTURES / "logs")
    checklist = load_checklist("general")
    return {r.session_id: score_run(r, checklist) for r in runs}


def test_alpha_expected_rows() -> None:
    a = _scores()["sess-alpha"]
    expected = {
        # delegation + ordering + chaining
        "delegate_sdg": "met",
        "order_sdg_before_training": "met",
        "order_training_before_eval": "met",
        "chain_training": "met",
        "chain_eval": "met",
        # eval judge named from the catalog (gpt-oss is in list_models output)
        "eval_judge_valid": "met",
        # one present_options with two distinct options; only one prompt total
        "no_duplicate_options_auto": "met",
        "no_repeated_decisions_auto": "n/a",
        # no model switch in this run
        "context_after_switch_auto": "n/a",
        "context_retained_after_switch": "n/a",
        # jobs created (job_created records) and each polled to terminal AFTER creation
        "sdg_job_created": "met",
        "training_job_created": "met",
        "eval_job_created": "met",
        "training_job_monitored": "met",
        "eval_job_monitored": "met",
        # structural integrity / progress
        "prompt_shown_for_review": "met",
        "eval_phase_signaled": "met",
        # no errors occurred
        "robustness_recovery": "n/a",
        "no_error_tool_calls": "met",
        "e2e_complete": "met",
    }
    for row_id, want in expected.items():
        assert a[row_id] == want, f"{row_id}: got {a[row_id]!r}, want {want!r}"


def test_beta_expected_rows() -> None:
    b = _scores()["sess-beta"]
    expected = {
        "sdg_config_valid": "wrong",  # validate_sdg_job returned an error
        "robustness_recovery": "wrong",  # errored, never recovered
        "no_error_tool_calls": "wrong",
        "e2e_complete": "missed",  # no completion record
        "sdg_job_created": "missed",  # no job_created records
        "training_job_monitored": "n/a",  # nothing created to monitor
        "no_duplicate_options_auto": "n/a",  # no present_options
        "context_after_switch_auto": "n/a",  # single turn, no switch
    }
    for row_id, want in expected.items():
        assert b[row_id] == want, f"{row_id}: got {b[row_id]!r}, want {want!r}"
