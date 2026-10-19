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


def test_cursor_injects_then_advances_per_turn() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT

    out1 = agent._with_active_skill_step(state, "sdg", "phase-2-gather", "u")
    assert "### Step 1" in out1 and state.skill_step_idx["sdg"] == 1
    out2 = agent._with_active_skill_step(state, "sdg", "phase-2-gather", "u")
    assert "### Step 2" in out2 and state.skill_step_idx["sdg"] == 2


def test_cursor_stops_at_last_step() -> None:
    state = agent.SessionState(orchestrator_id="o")
    state.active_skill_by_target["sdg"] = PILOT
    state.skill_step_idx["sdg"] = 9  # exhausted
    out = agent._with_active_skill_step(state, "sdg", "phase-2-gather", "u")
    assert out == "u"  # nothing more to inject, cursor does not grow
    assert state.skill_step_idx["sdg"] == 9


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
