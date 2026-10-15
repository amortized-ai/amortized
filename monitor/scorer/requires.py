"""Row-level ``requires`` preconditions, as a registry.

A row whose precondition is unmet scores ``n/a`` and (for ``llm_judge`` rows) is
not sent to the judge. New gate types register with ``@requires_gate("name")`` so
they plug in without editing a central function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from monitor.scorer.model import Run, _calls

RequiresGate = Callable[[Run, dict[str, Any]], bool]
REQUIRES: dict[str, RequiresGate] = {}


def requires_gate(name: str) -> Callable[[RequiresGate], RequiresGate]:
    def deco(fn: RequiresGate) -> RequiresGate:
        REQUIRES[name] = fn
        return fn

    return deco


def skill_role(skill_name: str | None) -> str | None:
    """Subagent role implied by a flattened opencode skill name (sdg-* -> sdg,
    training-* -> training)."""
    if not skill_name:
        return None
    if skill_name.startswith("sdg-"):
        return "sdg"
    if skill_name.startswith("training-"):
        return "training"
    return None


def skill_loaded(
    run: Run,
    *,
    skill: str | None = None,
    role: str | None = None,
    path_contains: str | None = None,
) -> bool:
    """Whether the run loaded a (matching) sub-skill. Native opencode skills load via
    the `skill` tool (`skill({name: "sdg-task-distillation"})`); its name is promoted
    to the tool-call `skill` field. Matches by exact `skill` name and/or `role`
    (derived from the name prefix or the call's role). Falls back to the pre-migration
    `read` of a guide path (`path_contains`) for old logs."""
    for c in run.tool_calls:
        if c.get("tool") != "skill" or c.get("status") == "error":
            continue
        name = c.get("skill") or (c.get("input") or {}).get("name")
        if skill and name != skill:
            continue
        if role and (skill_role(name) or c.get("role")) != role:
            continue
        return True
    # back-compat: pre-migration logs where the guide was loaded with `read`
    if path_contains:
        for c in run.tool_calls:
            if c.get("tool") != "read" or c.get("status") == "error":
                continue
            if role and c.get("role") != role:
                continue
            if path_contains not in (c.get("output") or ""):
                continue
            return True
    return False


@requires_gate("skill_loaded")
def _gate_skill_loaded(run: Run, requires: dict[str, Any]) -> bool:
    """Gate aspect-G / rubric rows on the matching sub-skill having been loaded."""
    return skill_loaded(
        run,
        skill=requires.get("skill"),
        role=requires.get("role"),
        path_contains=requires.get("path_contains"),
    )


@requires_gate("eval_for_trained_model")
def _gate_eval_for_trained_model(run: Run, requires: dict[str, Any]) -> bool:
    """The run validated an eval job whose config carries a `training_job_id` (the
    subject under eval is a tuned model, not a base/gateway model). Gates rows that
    only apply when an eval set is being built FOR a trained model."""
    return any(c.get("training_job_id") for c in _calls(run, "validate_eval_job"))


@requires_gate("model_switched")
def _gate_model_switched(run: Run, requires: dict[str, Any]) -> bool:
    """The Morty orchestrator model changed at least once mid-conversation. Gates
    the context-retention rows so they only apply when a switch actually happened."""
    return run.model_switches >= 1


def _requires_met(run: Run, requires: dict[str, Any] | None) -> bool:
    """Evaluate a row-level `requires` precondition against the run. Unknown types
    don't gate (return True)."""
    if not requires:
        return True
    gate = REQUIRES.get(requires.get("type", ""))
    if gate is None:
        return True  # unknown precondition -> do not gate
    return gate(run, requires)
