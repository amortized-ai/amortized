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


@requires_gate("skill_loaded")
def _gate_skill_loaded(run: Run, requires: dict[str, Any]) -> bool:
    """A `read` whose output echoes the guide path (`path_contains`), optionally
    scoped to a subagent `role` (sub-skills belong to subagents, so role-scoping
    disambiguates which agent loaded which guide). Reuses the same read/
    output_contains detection as the auto `skill_loaded` mechanic rows."""
    role = requires.get("role")
    needle = requires.get("path_contains")
    for c in run.tool_calls:
        if c.get("tool") != "read":
            continue
        if role and c.get("role") != role:
            continue
        if needle and needle not in (c.get("output") or ""):
            continue
        return True
    return False


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
