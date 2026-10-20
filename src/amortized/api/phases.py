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

import re
from functools import lru_cache
from pathlib import Path

from amortized.core.recipes import _get_base_dir

# The phase a freshly delegated subagent starts at, before it has signalled anything of
# its own (that target's last_signal_step_by_target entry is absent/""). The key is the
# first section slug in that agent's workflow. Keep in sync with the
# agents/<agent>/phases/ filenames and the thinned workflow.md outline. The
# orchestrator is absent: it never signals — the
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


# ── Per-step SKILL slices ──────────────────────────────────────────────────────
# Same idea as workflow phase injection, one level down and finer-grained. When a subagent
# loads a skill, opencode dumps the whole SKILL.md into its context. For a skill with a clear
# numbered-step gathering sequence, that whole body at once is more than a weak driver tracks.
# Fix: the SKILL.md keeps everything — routing, rules, the Step N TITLES (as an outline), and
# every non-step section (column pipeline, model config, checklist, …) — but each step's
# DETAIL body moves to `agents/<agent>/skills/<skill>/steps/<NN>-<slug>.md`, and the proxy
# injects one step's detail at a time as the subagent advances through gathering. The outline
# of titles stays resident so the driver is always oriented; the injected block surfaces the
# current step's detail in the freshest position.
#
# Opt-in by file presence: a skill with no `steps/` dir gets nothing injected and its full
# SKILL.md drives it as before. The `steps/` subdir sits one level below the skill dir, so
# `gen_native_skills` (maxdepth-1 copy) never ships it into opencode's skill tree; it lives
# only in the server image (COPY agents/) for injection here. Skills are keyed by their
# frontmatter `name` (what the `skill` tool call reports), resolved to a dir via the index.


def _frontmatter_name(text: str) -> str | None:
    """The `name:` from a SKILL.md YAML frontmatter block, or None."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("name:"):
            return line[len("name:") :].strip()
    return None


@lru_cache(maxsize=1)
def _skill_index() -> dict[str, tuple[str, str]]:
    """Map each skill's frontmatter `name` → (agent, skill-dir relative to the agent's
    skills/ root). Built once by scanning agents/*/skills/**/SKILL.md; the skill tool call
    reports the name, and this resolves it to where the step slices live."""
    base = _get_base_dir() / "agents"
    index: dict[str, tuple[str, str]] = {}
    for skill_md in base.glob("*/skills/**/SKILL.md"):
        try:
            name = _frontmatter_name(skill_md.read_text(encoding="utf-8"))
        except OSError:
            continue
        if not name:
            continue
        agent = skill_md.relative_to(base).parts[0]
        reldir = skill_md.parent.relative_to(base / agent / "skills").as_posix()
        index[name] = (agent, reldir)
    return index


# A step's canonical slug is the value it already tells the driver to stamp onto
# `present_options.step` — e.g. `Call present_options with step="sdg-domain"` → `sdg-domain`. The
# step block is the SINGLE source of truth, so the value the driver emits and the value the proxy
# resolves can never drift (deriving the slug from the filename instead silently broke skills whose
# in-file stamp used a different vocabulary). A block with no explicit stamp falls back to its
# filename stem minus the ordering prefix (`03-output-format.md` → `output-format`). _norm_step
# normalizes both sides so a driver emitting `Output Format`/`output_format`/`03-output-format`
# still resolves.
_STEP_PREFIX = re.compile(r"^\d+[-_ ]*")
_STEP_STAMP_RE = re.compile(
    r"present_options[^\n]*?\bstep\s*[=:]\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)


def _norm_step(value: str) -> str:
    out = _STEP_PREFIX.sub("", value.strip().lower())
    out = re.sub(r"[\s_]+", "-", out)
    return out.strip("-")


def _step_slug(path: Path, text: str) -> str:
    """The step id the driver is told to stamp on `present_options.step` for this step: the value
    from the block's own present_options instruction, else the filename stem (both normalized)."""
    match = _STEP_STAMP_RE.search(text)
    return _norm_step(match.group(1) if match else path.stem)


@lru_cache(maxsize=256)
def _skill_step_entries(skill_name: str) -> tuple[tuple[str, str], ...]:
    """The ordered `(slug, detail-block)` pairs for a skill, or () when it is unknown or has no
    `steps/` dir (not a stepped skill — no injection). `slug` is the step id the driver stamps on
    `present_options.step` (see _step_slug); the block is the file body. Files are ordered by name,
    so they must carry a zero-padded numeric prefix (01-, 02-, …). Empty files are skipped, so slug
    and block indices stay aligned. Cached like `_load`."""
    entry = _skill_index().get(skill_name)
    if entry is None:
        return ()
    agent, reldir = entry
    steps_dir = _get_base_dir() / "agents" / agent / "skills" / reldir / "steps"
    try:
        files = sorted(steps_dir.glob("*.md"))
    except OSError:
        return ()
    out: list[tuple[str, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            out.append((_step_slug(path, text), text))
    return tuple(out)


def _skill_steps(skill_name: str) -> tuple[str, ...]:
    """The ordered step-detail blocks for a skill (see _skill_step_entries)."""
    return tuple(block for _slug, block in _skill_step_entries(skill_name))


def skill_step_index_for_slug(skill_name: str | None, value: str | None) -> int:
    """Resolve a `present_options.step` value the driver reported to the 0-based index of the
    matching step, or -1 when the skill is not stepped or the value matches no step. Matching
    is on the normalized slug, so prefix/case/separator drift still resolves."""
    if not skill_name or not value:
        return -1
    target = _norm_step(value)
    if not target:
        return -1
    for idx, (slug, _block) in enumerate(_skill_step_entries(skill_name)):
        if slug == target:
            return idx
    return -1


def skill_step_count(skill_name: str | None) -> int:
    """How many step slices a skill has (0 if it is not a stepped skill)."""
    if not skill_name:
        return 0
    return len(_skill_steps(skill_name))


def _step_heading(block: str) -> str:
    """The step block's first heading line, stripped of leading `#`/space (e.g.
    `### Step 3 — Output format` → `Step 3 — Output format`), or "" if none."""
    for line in block.splitlines():
        s = line.strip()
        if s:
            return s.lstrip("#").strip()
    return ""


def skill_step_title(skill_name: str | None, idx: int) -> str | None:
    """The heading text of a skill's `idx`-th step (for a 'still to gather' list), or None
    when the skill is not stepped or `idx` is out of range."""
    block = skill_step_block(skill_name, idx)
    return _step_heading(block) if block is not None else None


def skill_optional_step_indices(skill_name: str | None) -> frozenset[int]:
    """Indices of steps marked OPTIONAL — their heading contains "(optional)". These are
    injectable if the driver reaches them, but never counted as a required gap (so coverage
    tracking neither lists them as outstanding nor pulls the driver back to them)."""
    if not skill_name:
        return frozenset()
    return frozenset(
        idx
        for idx, block in enumerate(_skill_steps(skill_name))
        if "optional" in _step_heading(block).lower()
    )


def skill_step_block(skill_name: str | None, idx: int) -> str | None:
    """The `idx`-th (0-based) step-detail block for a skill, or None when the skill is not
    stepped or `idx` is out of range (past the last step — nothing more to inject)."""
    if not skill_name or idx < 0:
        return None
    steps = _skill_steps(skill_name)
    if idx >= len(steps):
        return None
    return steps[idx]


def with_skill_step_block(skill_name: str | None, idx: int, text: str) -> str:
    """Prepend a skill's `idx`-th step-detail block to a turn message, if one exists.

    Wrapped in a `<system-reminder>` like `with_block`. No-op when the skill is not stepped
    or `idx` is out of range, so callers wrap unconditionally."""
    block = skill_step_block(skill_name, idx)
    if not block:
        return text
    wrapped = system_reminder(block)
    return f"{wrapped}\n\n{text}" if text else wrapped
