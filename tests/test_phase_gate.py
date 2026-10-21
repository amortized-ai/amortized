"""Unit tests for the pure phase-liveness adjudicator in api.phases.

The adjudicator classifies one turn from its tool-call set + phase. It has no live
consumer (the proxy has no gate layer by design); these tests pin the pure logic so an
offline consumer can rely on it.
"""

from __future__ import annotations

import amortized.api.phases as phases
from amortized.api.phases import Owed, adjudicate, owed_for

# --------------------------------------------------------------------------- #
# Lenient phases — bare prose is a legitimate free-text question
# --------------------------------------------------------------------------- #


def test_lenient_phase_bare_prose_is_ok():
    # step-3-metrics asks "How do you want to evaluate?" as free text.
    assert adjudicate("eval", "step-3-metrics", []).kind == "ok"
    assert adjudicate("eval", "step-0-what-to-eval", []).kind == "ok"
    assert adjudicate("eval", "held-out-ask-count", []).kind == "ok"


def test_unknown_slug_is_lenient():
    assert adjudicate("eval", "no-such-step", []).kind == "ok"
    assert owed_for("eval", "no-such-step") is None


# --------------------------------------------------------------------------- #
# Tool-owed phases — bare prose is a stall
# --------------------------------------------------------------------------- #


def test_stall_when_tool_owed_phase_yields_bare_prose():
    # The original bug: eval owes request_orchestrator_help but narrates the handoff.
    v = adjudicate("eval", "held-out-request-help", [])
    assert v.kind == "stall"
    assert v.owed is not None and "request_orchestrator_help" in v.owed.any_of


def test_owed_tool_present_is_ok():
    assert adjudicate("eval", "held-out-request-help", ["request_orchestrator_help"]).kind == "ok"
    assert adjudicate("eval", "step-4-validate", ["validate_eval_job"]).kind == "ok"
    assert adjudicate("sdg", "phase-3-validate", ["validate_sdg_job"]).kind == "ok"
    assert adjudicate("training", "phase-4-complete", ["signal_subagent_completion"]).kind == "ok"


def test_prerequisite_fetches_are_ignored():
    # list_models / get_dataset are not edges; present_options still satisfies.
    v = adjudicate("eval", "step-1-dataset", ["get_dataset", "list_jobs", "present_options"])
    assert v.kind == "ok"
    # … but fetches alone, with no card, still stall.
    assert adjudicate("eval", "step-1-dataset", ["get_dataset", "list_jobs"]).kind == "stall"


# --------------------------------------------------------------------------- #
# Wrong-edge — a card appeared but not the owed one
# --------------------------------------------------------------------------- #


def test_present_options_cannot_satisfy_a_validate_phase():
    # Using present_options as a submit control in a validate phase.
    v = adjudicate("eval", "step-4-validate", ["present_options"])
    assert v.kind == "wrong_edge"


def test_control_transfer_is_a_bail_not_a_stall():
    # Completing instead of validating transfers control → not a stall (premature-ness
    # is a tool-boundary/state concern, not a liveness one).
    assert adjudicate("eval", "step-4-validate", ["signal_subagent_completion"]).kind == "ok"


# --------------------------------------------------------------------------- #
# Edge #4 — awaiting a job is not a stall
# --------------------------------------------------------------------------- #


def test_awaiting_job_is_not_a_stall():
    assert adjudicate("eval", "step-5-report", [], awaiting_job=True).kind == "ok"
    assert adjudicate("eval", "step-5-report", []).kind == "stall"


# --------------------------------------------------------------------------- #
# Solo hand-off rule (subagent control-transfer)
# --------------------------------------------------------------------------- #


def test_handoff_must_be_solo():
    # held-out-request-help owes a SOLO request_orchestrator_help: the phase file
    # forbids narrating the hand-off, so prose or a co-tool alongside it is not_solo.
    ok = adjudicate("eval", "held-out-request-help", ["request_orchestrator_help"])
    assert ok.kind == "ok"
    # the mandatory per-turn signal_phase does NOT break solo
    with_signal = adjudicate(
        "eval", "held-out-request-help", ["request_orchestrator_help", "signal_phase"]
    )
    assert with_signal.kind == "ok"
    with_text = adjudicate(
        "eval", "held-out-request-help", ["request_orchestrator_help"], has_text=True
    )
    assert with_text.kind == "not_solo"
    with_cotool = adjudicate(
        "eval", "held-out-request-help", ["request_orchestrator_help", "get_dataset"]
    )
    assert with_cotool.kind == "not_solo"


def test_handoff_missing_is_a_stall_not_not_solo():
    assert adjudicate("eval", "held-out-request-help", []).kind == "stall"


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def test_batching_multiple_step_slugs_in_one_turn():
    v = adjudicate(
        "eval",
        "step-1-dataset",
        ["present_options"],
        present_steps=["step-1-dataset", "step-1_5-how-much"],
    )
    assert v.kind == "batching"


def test_single_step_slug_is_not_batching():
    v = adjudicate(
        "eval", "step-1-dataset", ["present_options"], present_steps=["step-1-dataset"]
    )
    assert v.kind == "ok"


# --------------------------------------------------------------------------- #
# Skill-step rule (gather)
# --------------------------------------------------------------------------- #


def test_gather_step_owes_present_options():
    # task-distillation step 02 (examples) owes present_options.
    v = adjudicate(
        "sdg",
        "phase-2-gather",
        [],
        skill="sdg-task-distillation",
        active_skill_step="examples",
    )
    assert v.kind == "stall"
    ok = adjudicate(
        "sdg",
        "phase-2-gather",
        ["present_options"],
        skill="sdg-task-distillation",
        active_skill_step="examples",
    )
    assert ok.kind == "ok"


def test_review_step_owes_both_show_prompt_and_present_options():
    owed = phases._skill_owed("sdg-task-distillation", "system-prompt")
    assert owed is not None and owed.all_of == frozenset({"show_prompt", "present_options"})
    # present_options alone is NOT enough on the review step.
    only_po = adjudicate(
        "sdg",
        "phase-2-gather",
        ["present_options"],
        skill="sdg-task-distillation",
        active_skill_step="system-prompt",
    )
    assert only_po.kind in {"wrong_edge", "stall"}
    both = adjudicate(
        "sdg",
        "phase-2-gather",
        ["show_prompt", "present_options"],
        skill="sdg-task-distillation",
        active_skill_step="system-prompt",
    )
    assert both.kind == "ok"


def test_gather_without_active_step_is_lenient():
    # No cursor info → cannot adjudicate the gather turn.
    assert adjudicate("sdg", "phase-2-gather", []).kind == "ok"


# --------------------------------------------------------------------------- #
# The held-out split produced two real, loadable phase slugs
# --------------------------------------------------------------------------- #


def test_held_out_split_phase_files_exist_and_old_slug_gone():
    phases._load.cache_clear()
    assert phases.phase_block("eval", "held-out-ask-count")
    assert phases.phase_block("eval", "held-out-request-help")
    assert phases.phase_block("eval", "held-out-eval-set") is None


def test_held_out_slugs_have_expected_policies():
    assert owed_for("eval", "held-out-ask-count") is None  # lenient
    assert owed_for("eval", "held-out-request-help") == Owed(
        any_of=frozenset({"request_orchestrator_help"}), solo=True
    )


def test_requires_phase_signal():
    assert phases.requires_phase_signal("eval")
    assert not phases.requires_phase_signal("orchestrator")
