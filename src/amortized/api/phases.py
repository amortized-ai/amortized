"""Per-phase instruction blocks injected into the turn message at run time.

The problem: a weaker driver (e.g. GLM-flash) reads a long system prompt and then
substitutes prose for a required transition tool — it narrates "I'll have a dataset
generated" instead of actually calling `request_orchestrator_help`, and the pipeline
stalls. Surfacing the current phase's instructions in the *freshest* position — the
turn message itself — instead of buried in a 400-line workflow.md makes the driver
reliably emit the transition tool it otherwise only narrates. An A/B on the
eval→SDG handback (glm-flash, N=6) lifted `request_orchestrator_help` from 1/6 to
6/6 with the block injected; see memory `phase-injection-lifts-glm-handback`.

Design: each agent's workflow.md is thinned to the phase TITLES only (an outline);
the full body of each phase lives in `agents/<agent>/phases/<phase>.md` and is
prepended to the turn message by the proxy (agent.py). Phases are keyed one-per-
section (the workflow's own `Step 0`, `Phase 2`, `held-out-eval-set`, …), so the key
IS the section slug the agent reports via `signal_phase.step` (subagents) or that the
proxy selects by control-flow path (the orchestrator, which never signals).

Opt-in by file presence: an agent with no `phases/` dir, or an unknown key, gets no
injection and its workflow.md still drives it — so agents migrate one at a time.

Content is authored under `agents/` (the single source of truth, baked into the
server image via `COPY agents/ agents/`) so phase text and the thinned workflow.md it
was split from stay side by side and version together.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from amortized.core.recipes import _get_base_dir

# The phase a freshly delegated subagent starts at, before it has signalled anything
# of its own (state.last_signal_step is still ""). The key is the first section slug
# in that agent's workflow. Keep in sync with the agents/<agent>/phases/ filenames and
# the thinned workflow.md outline. The orchestrator is absent: it never signals — the
# proxy picks its phase by control-flow path (see agent.py).
_FIRST_PHASE: dict[str, str] = {
    "sdg": "phase-1-route",
    "training": "phase-1-route",
    "eval": "step-0-what-to-eval",
}


def _phase_path(agent: str, phase: str) -> Path:
    return _get_base_dir() / "agents" / agent / "phases" / f"{phase}.md"


@lru_cache(maxsize=256)
def _load(agent: str, phase: str) -> str | None:
    """Read `agents/<agent>/phases/<phase>.md`, or None if absent/empty.

    Cached: phase files are baked at build time and never change within a process.
    Tests that write phase files on the fly must call `_load.cache_clear()`.
    """
    try:
        text = _phase_path(agent, phase).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def phase_block(agent: str, phase: str) -> str | None:
    """The instruction block for (agent, phase), or None when the agent has no such
    phase file (not migrated yet, or an unknown phase key)."""
    if not agent or not phase:
        return None
    return _load(agent, phase)


def subagent_phase(agent: str, step: str | None) -> str | None:
    """The phase key for a subagent turn, given the last step it signalled.

    Empty/unknown step → the agent's entry phase, so a freshly delegated subagent gets
    its starting block. None when the agent has no entry phase registered (not
    migrated), so callers inject nothing for it."""
    if step:
        return step
    return _FIRST_PHASE.get(agent or "")


def system_reminder(text: str) -> str:
    """Wrap platform-injected steering in a `<system-reminder>` block so the model reads
    it as a system directive, not as words the user typed. Used for every prepended or
    appended turn-message injection (phase blocks, steers, hand-off framing) so the clean
    user text stays the only user-authored part of the turn."""
    return f"<system-reminder>\n{text}\n</system-reminder>"


def with_block(agent: str, phase: str | None, text: str) -> str:
    """Prepend the (agent, phase) block to a turn message, if one exists.

    The block is wrapped in a `<system-reminder>` so it does not read as user text.
    No-op when phase is None or the agent/phase has no block, so callers can wrap every
    handoff unconditionally without first checking migration state."""
    block = phase_block(agent, phase) if phase else None
    if not block:
        return text
    wrapped = system_reminder(block)
    return f"{wrapped}\n\n{text}" if text else wrapped
