"""Per-step SKILL injection: a stepped skill's full body stays loaded, but each step's
detail is surfaced one at a time as the subagent advances through gathering.

The SKILL.md keeps routing, rules, the Step N titles, and every non-step section; only the
per-step detail lives in `steps/<NN>-<slug>.md` and is injected into the gather turn, the
cursor advancing one step per turn. Non-stepped skills (no `steps/` dir) get nothing injected.
"""

from typing import Any

from amortized.api import agent, phases

PILOT = "sdg-task-distillation"  # a stepped skill (9 steps)

# skills with a clear `### Step N` sequence → per-step injection
STEPPED = {
    "sdg-task-distillation": 9,
    "sdg-knowledge-ingestion": 6,
    "sdg-classification": 5,
}
NON_STEPPED = "sdg-embedding-classifier"  # inline one-line list, no steps/ dir


# ── phases.py: step resolution ──────────────────────────────────────────────────


def test_only_stepped_skills_have_steps() -> None:
    counts = {k: phases.skill_step_count(k) for k in phases._skill_index()}
    for name, expected in STEPPED.items():
        assert counts[name] == expected
    # every other skill is not stepped (no steps/ dir) → no injection
    assert all(v == 0 for k, v in counts.items() if k not in STEPPED)


def test_stepped_skills_resolve_in_order() -> None:
    for name, count in STEPPED.items():
        assert phases.skill_step_block(name, 0).startswith("### Step 1")
        assert phases.skill_step_block(name, count - 1) is not None
        assert phases.skill_step_block(name, count) is None  # past the last step


def test_step_blocks_in_order_then_none() -> None:
    assert phases.skill_step_block(PILOT, 0).startswith("### Step 1")
    assert phases.skill_step_block(PILOT, 8).startswith("### Step 9")
    assert phases.skill_step_block(PILOT, 9) is None  # past the last step
    assert phases.skill_step_block(PILOT, -1) is None
    assert phases.skill_step_block("no-such-skill", 0) is None
    assert phases.skill_step_count(None) == 0


def test_step_slug_resolves_with_normalization() -> None:
    # The driver stamps present_options.step; the resolver maps it to the step index, tolerant
    # of the numeric prefix, case, and separator the driver may or may not include.
    assert phases.skill_step_index_for_slug(PILOT, "examples") == 1
    assert phases.skill_step_index_for_slug(PILOT, "output-format") == 2
    assert phases.skill_step_index_for_slug(PILOT, "03-output-format") == 2
    assert phases.skill_step_index_for_slug(PILOT, "Output Format") == 2
    assert phases.skill_step_index_for_slug(PILOT, "nope") == -1
    assert phases.skill_step_index_for_slug(PILOT, "") == -1
    assert phases.skill_step_index_for_slug(PILOT, None) == -1
    assert phases.skill_step_index_for_slug("no-such-skill", "examples") == -1


def test_step_slug_read_from_in_file_stamp() -> None:
    # The slug is the value the step block ALREADY tells the driver to stamp, not the filename.
    # Classification's steps declare an `sdg-*` vocabulary; it must resolve. Regression: deriving
    # the slug from the filename made these never match, silently dropping the cursor back to blind
    # turn-counting for classification.
    assert phases.skill_step_index_for_slug("sdg-classification", "sdg-domain") == 0
    assert phases.skill_step_index_for_slug("sdg-classification", "sdg-samples") == 3
    assert phases.skill_step_index_for_slug("sdg-classification", "sdg-teacher-model") == 4
    # the filename stem is NOT this skill's vocabulary, so it must not accidentally match
    assert phases.skill_step_index_for_slug("sdg-classification", "domain") == -1


def test_step_is_review_identifies_the_show_prompt_step() -> None:
    # The gate pairs a review card with its approval question by step: the approval question
    # carries the slug of the step whose detail drives show_prompt. Task-distillation's step 8
    # ("system-prompt") is that step; no other step is a review step.
    assert phases.skill_step_is_review(PILOT, "system-prompt") is True
    assert phases.skill_step_is_review(PILOT, "08-system-prompt") is True  # prefix-tolerant
    assert phases.skill_step_is_review(PILOT, "examples") is False  # a non-review step
    assert phases.skill_step_is_review(PILOT, "nope") is False
    assert phases.skill_step_is_review(PILOT, None) is False
    assert phases.skill_step_is_review(None, "system-prompt") is False
    assert phases.skill_step_is_review("no-such-skill", "system-prompt") is False


def test_with_skill_step_block_wraps_and_noops() -> None:
    wrapped = phases.with_skill_step_block(PILOT, 0, "hi")
    assert wrapped.startswith("<system-reminder>")
    assert "### Step 1" in wrapped
    assert wrapped.endswith("hi")
    # out of range / unknown → pass through untouched
    assert phases.with_skill_step_block(PILOT, 99, "hi") == "hi"
    assert phases.with_skill_step_block("no-such-skill", 0, "hi") == "hi"


# ── agent.py: recording the loaded skill ────────────────────────────────────────


def _skill_part(name: str) -> dict[str, Any]:
    return {
        "type": "tool",
        "tool": "skill",
        "state": {"status": "completed", "input": {"name": name}},
    }


def _signal_part(step: str) -> dict[str, Any]:
    return {"type": "tool", "tool": "mcp_amortized__signal_phase", "input": {"step": step}}


def _present_options_part(step: str) -> dict[str, Any]:
    return {"type": "tool", "tool": "mcp_amortized__present_options", "input": {"step": step}}


def test_skill_load_recorded_even_with_signal_in_same_turn() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.subagent_target = "sdg"
    result = {"parts": [_signal_part("phase-2-gather"), _skill_part(PILOT)]}
    agent._ensure_phase_signal(state, result)
    assert state.last_signal_step == "phase-2-gather"
    assert state.active_skill_by_target["sdg"] == PILOT


def test_skill_recording_per_target_no_bleed() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["eval"] = "other"
    state.subagent_target = "sdg"
    agent._ensure_phase_signal(state, {"parts": [_skill_part(PILOT)]})
    assert state.active_skill_by_target["sdg"] == PILOT
    assert state.active_skill_by_target["eval"] == "other"


# ── agent.py: the gather-turn step coverage ──────────────────────────────────────


def test_inject_is_resident_aware_and_does_not_advance() -> None:
    # Step 1 is baked into the served SKILL.md at build time, so injection starts at step 2
    # (the lowest NON-resident uncovered step). Injection is READ-ONLY: coverage is recorded by
    # the response handler, not by the act of injecting, so two injections with no response
    # between them surface the SAME step.
    assert agent._RESIDENT_SKILL_STEPS == 1
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT

    out1 = agent._active_skill_step_block(state, "sdg", "phase-2-gather") or ""
    assert "### Step 2" in out1 and "### Step 1" not in out1
    assert "still to gather" not in out1.lower()  # nothing covered yet → no holes behind
    out2 = agent._active_skill_step_block(state, "sdg", "phase-2-gather") or ""
    assert "### Step 2" in out2  # unchanged: injection never records coverage itself
    assert state.skill_steps_done.get("sdg") is None  # no write from injection


def test_coverage_tracks_reported_step() -> None:
    # Coverage follows the step the driver reports reaching via present_options.step: asking
    # step 2's question (slug "examples", index 1) records it, so the next injected step is 3.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("examples")])
    assert state.skill_steps_done["sdg"] == {1}
    out = agent._active_skill_step_block(state, "sdg", "phase-2-gather") or ""
    assert "### Step 3" in out


def test_jump_ahead_leaves_holes_filled_lowest_first() -> None:
    # A driver that jumps ahead (asks step 2 then step 5) leaves steps 3-4 UNCOVERED. The next
    # injection pulls it back to the lowest hole (step 3), and the turn lists the remaining hole
    # (step 4) as still-to-gather — so the skipped steps are never silently lost.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("examples")])  # index 1
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("variation-dimensions")])  # 4
    assert state.skill_steps_done["sdg"] == {1, 4}
    out = agent._active_skill_step_block(state, "sdg", "phase-2-gather") or ""
    assert "### Step 3" in out  # pulled BACK to the lowest hole, not forward to step 6
    assert "still to gather" in out.lower()
    assert "Task criteria" in out  # step 4 (index 3) flagged as still outstanding


def test_reask_a_covered_step_is_idempotent() -> None:
    # Re-asking a step already covered (a stall, or the interaction rerun re-observing the card)
    # must not change coverage — the set-add is naturally idempotent.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_steps_done["sdg"] = {1}
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("examples")])  # idx 1, covered
    assert state.skill_steps_done["sdg"] == {1}


def test_no_question_no_coverage_change() -> None:
    # A turn with no present_options (narration only, or a non-gather tool) is no evidence of
    # progress — coverage must not change.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_steps_done["sdg"] = {1, 2}
    agent._advance_skill_cursor(state, "sdg", [_signal_part("phase-2-gather")])
    assert state.skill_steps_done["sdg"] == {1, 2}


def test_fallback_for_unrecognized_label_covers_current_step_idempotently() -> None:
    # A driver that ignores the slug vocabulary (free-text label) still makes gathering progress
    # by covering the step currently being injected — but a repeat of the same label (e.g. the
    # interaction rerun re-observing the card) must not double-count.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_steps_done["sdg"] = {1}  # next injectable is index 2
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("my-own-label")])
    assert state.skill_steps_done["sdg"] == {1, 2}  # unrecognized, new → cover the current step
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("my-own-label")])
    assert state.skill_steps_done["sdg"] == {1, 2}  # same label again → held (rerun-safe)


def test_all_covered_injects_nothing() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_steps_done["sdg"] = set(range(9))  # every step covered (count == 9)
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("generate-outputs")])
    assert state.skill_steps_done["sdg"] == set(range(9))  # unchanged
    out = agent._active_skill_step_block(state, "sdg", "phase-2-gather")
    assert out is None  # nothing left to inject


def test_optional_step_never_blocks_or_lists() -> None:
    # Step 9 is marked "(optional)". With every REQUIRED step covered but the optional one not,
    # the optional step is offered (its detail is still useful) but it is NOT listed as an
    # outstanding hole, so it never blocks the move to building.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_steps_done["sdg"] = set(range(8))  # indices 0-7 (all required) covered; 8 optional
    out = agent._active_skill_step_block(state, "sdg", "phase-2-gather") or ""
    assert "### Step 9" in out  # the optional step is still offered
    assert "still to gather" not in out.lower()  # but never flagged as an outstanding hole


def test_no_injection_outside_gather_phase() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    # building/validating: step detail must not bleed in, coverage must not change
    out = agent._active_skill_step_block(state, "sdg", "phase-3-validate")
    assert out is None
    assert "sdg" not in state.skill_steps_done


def test_non_stepped_skill_is_noop() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = NON_STEPPED  # inline list, no steps/ dir
    out = agent._active_skill_step_block(state, "sdg", "phase-2-gather")
    assert out is None
    assert "sdg" not in state.skill_steps_done
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("whatever")])
    assert "sdg" not in state.skill_steps_done  # non-stepped → nothing to track
