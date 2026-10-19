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


# ── agent.py: the gather-turn cursor ────────────────────────────────────────────


def test_inject_is_resident_aware_and_does_not_advance() -> None:
    # Step 1 is baked into the served SKILL.md at build time, so injection starts at step 2
    # (the first NON-resident step). Injection is READ-ONLY now: the cursor is advanced by the
    # response handler, not by the act of injecting, so two injections with no response between
    # them surface the SAME step.
    assert agent._RESIDENT_SKILL_STEPS == 1
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT

    out1 = agent._with_active_skill_step(state, "sdg", "phase-2-gather", "u")
    assert "### Step 2" in out1 and "### Step 1" not in out1
    out2 = agent._with_active_skill_step(state, "sdg", "phase-2-gather", "u")
    assert "### Step 2" in out2  # unchanged: injection never advances the cursor itself
    assert state.skill_step_idx.get("sdg") is None  # no write from injection


def test_cursor_tracks_reported_step() -> None:
    # The cursor follows the step the driver reports reaching via present_options.step: asking
    # step 2's question (slug "examples", index 1) moves the cursor to inject step 3 next.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("examples")])
    assert state.skill_step_idx["sdg"] == 2
    out = agent._with_active_skill_step(state, "sdg", "phase-2-gather", "u")
    assert "### Step 3" in out


def test_cursor_catches_up_on_a_jump() -> None:
    # A driver that jumps ahead (asks step 4's question) must not keep getting stale early steps:
    # the cursor leaps to one past the furthest reached, honouring a prefixed/odd-cased label too.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("04-task-criteria")])
    assert state.skill_step_idx["sdg"] == 4  # index 3 reached → inject step 5 next


def test_cursor_holds_on_stall_or_reask() -> None:
    # Re-asking the same step (or a stall) must NOT skip ahead: the cursor holds so the step is
    # re-injected, never jumped over.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_step_idx["sdg"] = 2
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("examples")])  # index 1 < cur
    assert state.skill_step_idx["sdg"] == 2  # max(2, 2) — held


def test_cursor_no_question_no_advance() -> None:
    # A turn with no present_options (narration only, or a non-gather tool) is no evidence of
    # forward progress — the cursor must not move.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_step_idx["sdg"] = 2
    agent._advance_skill_cursor(state, "sdg", [_signal_part("phase-2-gather")])
    assert state.skill_step_idx["sdg"] == 2


def test_cursor_fallback_for_unrecognized_label_is_idempotent() -> None:
    # A driver that ignores the slug vocabulary (free-text label) still makes gathering progress
    # (one forward step per NEW question) — but a repeat of the same label (e.g. the interaction
    # rerun re-observing the card) must not double-count.
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_step_idx["sdg"] = 2
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("my-own-label")])
    assert state.skill_step_idx["sdg"] == 3  # unrecognized, new → +1
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("my-own-label")])
    assert state.skill_step_idx["sdg"] == 3  # same label again → held (rerun-safe)


def test_cursor_clamps_at_last_step() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_step_idx["sdg"] = 9  # exhausted (count == 9)
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("generate-outputs")])
    assert state.skill_step_idx["sdg"] == 9  # never past the last step
    out = agent._with_active_skill_step(state, "sdg", "phase-2-gather", "u")
    assert out == "u"  # nothing more to inject


def test_no_injection_outside_gather_phase() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    # building/validating: step detail must not bleed in, cursor must not move
    out = agent._with_active_skill_step(state, "sdg", "phase-3-validate", "u")
    assert out == "u"
    assert "sdg" not in state.skill_step_idx


def test_non_stepped_skill_is_noop() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = NON_STEPPED  # inline list, no steps/ dir
    out = agent._with_active_skill_step(state, "sdg", "phase-2-gather", "u")
    assert out == "u"
    assert "sdg" not in state.skill_step_idx
    agent._advance_skill_cursor(state, "sdg", [_present_options_part("whatever")])
    assert "sdg" not in state.skill_step_idx  # non-stepped → nothing to advance
