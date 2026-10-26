"""Unit tests for the pure phase-liveness adjudicator in api.phases.

The adjudicator classifies one turn from its tool-call set + phase. It has no live consumer (the
proxy has no gate layer by design); these tests pin the pure logic so an offline consumer can rely
on it. Slugs are the resident `<!--signal:slug-->` vocabulary (sdg/training: route/gather/validate/
complete; eval: understand-task/eval-dataset/how-much/model-source/metrics/validate/report/complete
plus the held-out-ask-count / held-out-request-help branch).
"""

from __future__ import annotations

import amortized.api.phases as phases
from amortized.api.phases import Owed, adjudicate, owed_for, signal_is_known

# --------------------------------------------------------------------------- #
# Lenient phases — bare prose is a legitimate free-text question
# --------------------------------------------------------------------------- #


def test_lenient_phase_bare_prose_is_ok() -> None:
    # metrics asks "How do you want to evaluate?" as free text.
    assert adjudicate("eval", "metrics", []).kind == "ok"
    assert adjudicate("eval", "understand-task", []).kind == "ok"
    assert adjudicate("eval", "held-out-ask-count", []).kind == "ok"


def test_unknown_slug_is_lenient() -> None:
    assert adjudicate("eval", "no-such-step", []).kind == "ok"
    assert owed_for("eval", "no-such-step") is None


# --------------------------------------------------------------------------- #
# signal_is_known — the #1a vocabulary predicate (keys the signalled-guard, NOT adjudicate)
# --------------------------------------------------------------------------- #


def test_signal_is_known_accepts_required_and_section_slugs() -> None:
    # _REQUIRED keys and `## Workflow` section slugs are recognized (exact, normalized).
    assert signal_is_known("eval", "validate")  # _REQUIRED
    assert signal_is_known("eval", "held-out-ask-count")  # section-only (lenient but keyed)
    assert signal_is_known("training", "gather")  # the coarse umbrella IS a real section slug
    assert signal_is_known("sdg", "route")
    assert signal_is_known("eval", "Validate")  # normalization applies


def test_signal_is_known_rejects_off_vocab_and_prefix_variants() -> None:
    # The driver inventing a slug (or an off-by-prefix one) is NOT keyed — the guard treats it
    # like a skipped signal even though the pointer would prefix-resolve "gather-requirements".
    assert not signal_is_known("training", "gather-requirements")
    assert not signal_is_known("eval", "no-such-step")
    assert not signal_is_known("eval", "")
    # A real slug for the WRONG role is still unknown for this role.
    assert not signal_is_known("training", "eval-dataset")


# --------------------------------------------------------------------------- #
# Tool-owed phases — bare prose is a stall
# --------------------------------------------------------------------------- #


def test_stall_when_tool_owed_phase_yields_bare_prose() -> None:
    # The original bug: eval owes request_orchestrator_help but narrates the handoff.
    v = adjudicate("eval", "held-out-request-help", [])
    assert v.kind == "stall"
    assert v.owed is not None and "request_orchestrator_help" in v.owed.any_of


def test_owed_tool_present_is_ok() -> None:
    assert adjudicate("eval", "held-out-request-help", ["request_orchestrator_help"]).kind == "ok"
    assert adjudicate("eval", "validate", ["validate_eval_job"]).kind == "ok"
    assert adjudicate("sdg", "validate", ["validate_sdg_job"]).kind == "ok"
    assert adjudicate("training", "complete", ["signal_subagent_completion"]).kind == "ok"


def test_prerequisite_fetches_are_ignored() -> None:
    # list_models / get_dataset are not edges; present_options still satisfies.
    v = adjudicate("eval", "eval-dataset", ["get_dataset", "list_jobs", "present_options"])
    assert v.kind == "ok"
    # … but fetches alone, with no card, still stall.
    assert adjudicate("eval", "eval-dataset", ["get_dataset", "list_jobs"]).kind == "stall"


# --------------------------------------------------------------------------- #
# Wrong-edge — a card appeared but not the owed one
# --------------------------------------------------------------------------- #


def test_present_options_cannot_satisfy_a_validate_phase() -> None:
    # Using present_options as a submit control in a validate phase.
    v = adjudicate("eval", "validate", ["present_options"])
    assert v.kind == "wrong_edge"


def test_control_transfer_is_a_bail_not_a_stall() -> None:
    # Completing instead of validating transfers control → not a stall (premature-ness is a
    # tool-boundary/state concern, not a liveness one).
    assert adjudicate("eval", "validate", ["signal_subagent_completion"]).kind == "ok"


# --------------------------------------------------------------------------- #
# Edge #4 — awaiting a job is not a stall
# --------------------------------------------------------------------------- #


def test_awaiting_job_is_not_a_stall() -> None:
    assert adjudicate("eval", "report", [], awaiting_job=True).kind == "ok"
    assert adjudicate("eval", "report", []).kind == "stall"


# --------------------------------------------------------------------------- #
# Solo hand-off rule (subagent control-transfer)
# --------------------------------------------------------------------------- #


def test_handoff_must_be_solo() -> None:
    # held-out-request-help owes a SOLO request_orchestrator_help: the resident section forbids
    # narrating the hand-off, so prose or a co-tool alongside it is not_solo.
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


def test_handoff_missing_is_a_stall_not_not_solo() -> None:
    assert adjudicate("eval", "held-out-request-help", []).kind == "stall"


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def test_batching_multiple_step_slugs_in_one_turn() -> None:
    v = adjudicate(
        "eval",
        "eval-dataset",
        ["present_options"],
        present_steps=["eval-dataset", "how-much"],
    )
    assert v.kind == "batching"


def test_single_step_slug_is_not_batching() -> None:
    v = adjudicate("eval", "eval-dataset", ["present_options"], present_steps=["eval-dataset"])
    assert v.kind == "ok"


# --------------------------------------------------------------------------- #
# Skill-step rule (gather)
# --------------------------------------------------------------------------- #


def test_skill_step_owes_present_options() -> None:
    # The skill step is now signalled directly (signal_phase.step == the skill slug); its obligation
    # resolves from that one cursor, scoped by the loaded skill. task-distillation "examples" owes
    # present_options.
    v = adjudicate("sdg", "examples", [], skill="sdg-task-distillation")
    assert v.kind == "stall"
    ok = adjudicate("sdg", "examples", ["present_options"], skill="sdg-task-distillation")
    assert ok.kind == "ok"


def test_review_step_owes_both_show_prompt_and_present_options() -> None:
    owed = phases._skill_owed("sdg-task-distillation", "system-prompt")
    assert owed is not None and owed.all_of == frozenset({"show_prompt", "present_options"})
    # present_options alone is NOT enough on the review step.
    only_po = adjudicate(
        "sdg", "system-prompt", ["present_options"], skill="sdg-task-distillation"
    )
    assert only_po.kind in {"wrong_edge", "stall"}
    both = adjudicate(
        "sdg", "system-prompt", ["show_prompt", "present_options"], skill="sdg-task-distillation"
    )
    assert both.kind == "ok"


def test_skill_slug_scoped_by_active_skill() -> None:
    # `teacher-model` recurs across sdg skills; the obligation is resolved WITHIN the loaded skill,
    # so the same signalled slug is adjudicated against whichever skill is active (no global clash).
    for skill in ("sdg-task-distillation", "sdg-knowledge-ingestion", "sdg-classification"):
        assert adjudicate("sdg", "teacher-model", [], skill=skill).kind == "stall"
        assert (
            adjudicate("sdg", "teacher-model", ["present_options"], skill=skill).kind == "ok"
        )


def test_coarse_gather_umbrella_is_lenient() -> None:
    # A non-stepped skill (or no cursor) signals the coarse `gather` umbrella: not a workflow phase
    # and not a skill step → lenient, so bare prose is allowed.
    assert adjudicate("sdg", "gather", []).kind == "ok"
    assert adjudicate("sdg", "gather", [], skill="sdg-embedding-classifier").kind == "ok"


# --------------------------------------------------------------------------- #
# The held-out split produced two real, resident section slugs
# --------------------------------------------------------------------------- #


def test_held_out_split_sections_exist_and_old_reuse_gone() -> None:
    phases._sections.cache_clear()
    slugs = {slug for slug, _title in phases._sections("eval")}
    assert "held-out-ask-count" in slugs
    assert "held-out-request-help" in slugs
    # the old branch heading reused `eval-dataset`; after the split it has its own slugs.
    assert "held-out-eval-set" not in slugs


def test_held_out_slugs_have_expected_policies() -> None:
    assert owed_for("eval", "held-out-ask-count") is None  # lenient
    assert owed_for("eval", "held-out-request-help") == Owed(
        any_of=frozenset({"request_orchestrator_help"}), solo=True
    )


def test_requires_phase_signal() -> None:
    assert phases.requires_phase_signal("eval")
    assert not phases.requires_phase_signal("orchestrator")
