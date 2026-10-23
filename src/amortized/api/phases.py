"""A short per-turn POINTER that re-anchors a subagent to its RESIDENT workflow.

The full step-by-step workflow lives resident in each agent's `workflow.md` (= main; no
thinning, no `phases/` slices). A weaker driver (e.g. GLM-5.3-flash) reading a long resident
prompt sometimes substitutes prose for a required transition tool — it narrates "I'll have a
dataset generated" instead of calling `request_orchestrator_help` — or works ahead out of
order. A SHORT pointer on the system channel ("you're in the <agent> workflow; your
instructions are in `## Workflow`; CALL the transition tool, don't narrate it; don't work
ahead") re-anchors the driver each turn without injecting a second copy of the workflow.

Validated at N=6 (memory `pr526-resident-inject-flash`): un-thinned/resident content fixes the
out-of-order formula probe, and resident+pointer holds the eval→SDG handback (6/6, matching the
full-block inject). This is the SLICE-FREE form of that result — the pointer derives the current
section title from the resident `workflow.md` headings themselves (no `phases/` files), and it is
delivered via opencode's message-level `system` field (the PR509 system-channel machinery), never
prepended to the user text.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from amortized.core.recipes import _get_base_dir

# Only leaf workflow subagents get a pointer. The orchestrator drives on its resident
# workflow.md exactly like main (it never signals a phase; its turns are not re-anchored).
_SUBAGENTS = frozenset({"sdg", "training", "eval"})


def _workflow_path(agent: str) -> Path:
    return _get_base_dir() / "agents" / agent / "workflow.md"


def _norm(value: str) -> str:
    """Normalize a signalled step to a comparable slug: drop a leading numeric ordering prefix
    (`03-`) and collapse every run of non-alphanumeric characters (spaces, em dashes,
    underscores, …) to a single hyphen."""
    out = re.sub(r"^\d+[-_.) ]*", "", value.strip().lower())
    return re.sub(r"[^a-z0-9]+", "-", out).strip("-")


# A section's canonical signal slug is authored right in its heading as an HTML-comment marker —
# `### Step 0 — Confirm what to evaluate <!--signal:understand-task-->` — so the slug the agent
# signals (`signal_phase.step`) and the slug the pointer resolves against live in ONE place (the
# resident workflow.md), and a human-readable title can carry a short, stable slug that isn't just
# the slugified title. Headings with no marker fall back to the normalized title.
_SIGNAL_MARKER = re.compile(r"<!--\s*signal:\s*([a-z0-9-]+)\s*-->")


@lru_cache(maxsize=16)
def _sections(agent: str) -> tuple[tuple[str, str], ...]:
    """Ordered `(slug, title)` for each `### ` heading under the resident workflow.md's
    `## Workflow` section. `slug` is the heading's `<!--signal:…-->` marker, else the normalized
    title. `()` when the file is absent or has no Workflow section. Cached: workflow.md is baked at
    build time; tests that write it on the fly call `_sections.cache_clear()`."""
    try:
        text = _workflow_path(agent).read_text(encoding="utf-8")
    except OSError:
        return ()
    out: list[tuple[str, str]] = []
    in_workflow = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            in_workflow = s[3:].strip().lower() == "workflow"
            continue
        if in_workflow and s.startswith("### "):
            raw = s[4:].strip()
            marker = _SIGNAL_MARKER.search(raw)
            title = _SIGNAL_MARKER.sub("", raw).strip()
            out.append((marker.group(1) if marker else _norm(title), title))
    return tuple(out)


def _resolve_title(agent: str, step: str | None) -> str | None:
    """The resident section title to name in the pointer.

    - No step yet (fresh delegation) → the entry (first) section, so a just-delegated subagent
      is pointed at where to start.
    - A signalled step that matches a section heading (exact or prefix, both normalized) → that
      section.
    - A signalled step that matches no heading (e.g. the coarse UI progress vocabulary like
      `gather_requirements`, which is not a section slug) → None, so the pointer stays generic
      rather than naming a WRONG section."""
    sections = _sections(agent)
    if not sections:
        return None
    if not step:
        return sections[0][1]
    target = _norm(step)
    if not target:
        return sections[0][1]
    for slug, title in sections:
        if slug == target or slug.startswith(f"{target}-") or target.startswith(f"{slug}-"):
            return title
    return None


# ── Skill-step resolution (for the in-gather skill clause of the pointer) ─────────────────
# A subagent's gather phase is driven by a resident SKILL.md with a numbered `### Step N — …`
# sequence. Each step pairs its heading with the `present_options(step="…")` stamp the driver is
# told to emit for that question — so the stamp the agent ALREADY sends is enough to name the
# current step, with no new slug markers in the SKILL.md (the user's "skills don't need slugs").
# This resolves that stamp → the step's heading title; the agent.py scan records the active skill
# (from the `skill` tool call) and the latest stamp (from present_options) per turn.

_STEP_STAMP_RE = re.compile(
    r"present_options[^\n]*?\bstep\s*[=:]\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)


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
    """Map each skill's frontmatter `name` → (agent, skill-dir relative to that agent's `skills/`
    root), by scanning `agents/*/skills/**/SKILL.md`. The `skill` tool call reports the name; this
    resolves it to where the SKILL.md lives."""
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


@lru_cache(maxsize=64)
def _skill_steps(skill_name: str) -> tuple[tuple[str, str], ...]:
    """Ordered `(slug, title)` for each `### ` step heading in a skill's resident SKILL.md that
    carries a slug — a heading `<!--signal:…-->` marker (preferred; the same convention workflow
    sections use) or, failing that, a `present_options(step="…")` stamp in its body. Headings with
    neither (reference sections like `### columns`, or a generate-only step) are skipped. `()` for
    an unknown skill. Cached like `_sections`; tests writing SKILL.md on the fly call
    `_skill_steps.cache_clear()`."""
    entry = _skill_index().get(skill_name)
    if entry is None:
        return ()
    agent, reldir = entry
    path = _get_base_dir() / "agents" / agent / "skills" / reldir / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    out: list[tuple[str, str]] = []
    title: str | None = None
    marker: str | None = None
    body: list[str] = []

    def _flush() -> None:
        if title is None:
            return
        if marker:
            out.append((marker, title))
            return
        stamp = _STEP_STAMP_RE.search("\n".join(body))
        if stamp:
            out.append((_norm(stamp.group(1)), title))

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("### "):
            _flush()
            raw = s[4:].strip()
            found = _SIGNAL_MARKER.search(raw)
            title, marker, body = _SIGNAL_MARKER.sub("", raw).strip(), (
                found.group(1) if found else None
            ), []
        else:
            body.append(line)
    _flush()
    return tuple(out)


def skill_step_title(skill_name: str | None, stamp: str | None) -> str | None:
    """The SKILL.md step title for the `present_options.step` stamp the agent last emitted, matched
    on the normalized stamp (exact or prefix). None when the skill or stamp is unknown."""
    if not skill_name or not stamp:
        return None
    target = _norm(stamp)
    if not target:
        return None
    for slug, title in _skill_steps(skill_name):
        if slug == target or slug.startswith(f"{target}-") or target.startswith(f"{slug}-"):
            return title
    return None


# Workflow phases AFTER the skill-driven gathering is done (shared by the skill-loading
# subagents, sdg + training). Once a subagent signals one of these, it is confirming or
# finishing, not gathering — so the pointer must stop steering it back through the SKILL.md
# gathering steps even though the loaded skill is still tracked for the rest of its life.
_POST_GATHER_STEPS = frozenset({"validate", "complete"})


def subagent_pointer(
    agent: str | None,
    step: str | None,
    skill: str | None = None,
    skill_step: str | None = None,
) -> str | None:
    """A SHORT system-channel pointer re-anchoring a subagent to its RESIDENT instructions.

    Names the current `## Workflow` section (from the signalled step) and, when a skill is loaded
    during gathering, the current SKILL.md step (from the `present_options` stamp). None for a
    non-subagent (the orchestrator) or an agent with no workflow.md (no-op, so callers can compose
    it unconditionally). Delivered via opencode's `system` field — a raw, tagged instruction like
    the other system-channel steers (`[SYSTEM EVENT]`, …), NOT wrapped in `<system-reminder>` and
    never prepended to the user text."""
    if not agent or agent not in _SUBAGENTS or not _sections(agent):
        return None
    wf_title = _resolve_title(agent, step)
    wf_at = f" — your current step is **{wf_title}**" if wf_title else ""

    skill_clause = ""
    if skill and (_norm(step) if step else "") not in _POST_GATHER_STEPS:
        step_title = skill_step_title(skill, skill_step)
        on = f", and you are on **{step_title}**" if step_title else ""
        skill_clause = (
            f" You have the **{skill}** skill loaded; its gathering steps are resident in the "
            f"SKILL.md{on}. Work that `### Step` sequence IN ORDER — ask ONE question per turn "
            "with `present_options` and do not skip or batch steps."
        )

    return (
        f"[ACTIVE PHASE] You are in the {agent} workflow{wf_at}. Your full step-by-step "
        "instructions are resident in the `## Workflow` section of your system prompt."
        f"{skill_clause} Find your current step there and follow it now, in order. When a step "
        "calls for a transition tool (e.g. `request_orchestrator_help`, "
        "`signal_subagent_completion`), CALL the tool — do NOT describe the action in prose. Do "
        "not work ahead to a later step."
    )


# ── Skill-step review detection (for the gather-step obligation) ───────────────────────────
# A gather step whose resident SKILL.md body drives a review card (`show_prompt`) owes BOTH the
# review and its approval `present_options` — so the adjudicator can tell a review step apart from
# an ordinary question step. This mirrors the slug resolution in `_skill_steps` (marker, else the
# present_options stamp) but keeps the body so it can test for the review tool.
_REVIEW_TOOL_IN_STEP = "show_prompt"


@lru_cache(maxsize=64)
def _skill_review_slugs(skill_name: str) -> frozenset[str]:
    """Slugs of a skill's `### ` steps whose SKILL.md body contains a `show_prompt` review. `()`
    for an unknown skill. Cached like `_skill_steps`; tests writing SKILL.md on the fly call
    `_skill_review_slugs.cache_clear()`."""
    entry = _skill_index().get(skill_name)
    if entry is None:
        return frozenset()
    agent, reldir = entry
    path = _get_base_dir() / "agents" / agent / "skills" / reldir / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    out: set[str] = set()
    marker: str | None = None
    body: list[str] = []
    seen = False

    def _flush() -> None:
        joined = "\n".join(body)
        if marker:
            slug: str | None = marker
        else:
            stamp = _STEP_STAMP_RE.search(joined)
            slug = _norm(stamp.group(1)) if stamp else None
        if slug and _REVIEW_TOOL_IN_STEP in joined:
            out.add(slug)

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("### "):
            if seen:
                _flush()
            seen = True
            found = _SIGNAL_MARKER.search(s[4:].strip())
            marker = found.group(1) if found else None
            body = []
        elif seen:
            body.append(line)
    if seen:
        _flush()
    return frozenset(out)


def skill_step_is_review(skill_name: str | None, step_value: str | None) -> bool:
    """True when `step_value` (a `present_options.step` the driver stamped) names a skill step
    whose detail drives a review card (`show_prompt`). The approval question for a prompt carries
    that step's own slug, so such a step legitimately emits both the review and the question."""
    if not skill_name or not step_value:
        return False
    target = _norm(step_value)
    return bool(target) and target in _skill_review_slugs(skill_name)


# ── Phase-liveness adjudication (pure library) ────────────────────────────────────────────
#
# A weak driver (GLM-flash) sometimes substitutes PROSE for a required transition — it narrates
# "I'll have the eval set generated" and yields without calling `request_orchestrator_help`, and
# the pipeline stalls (nobody is driving). The resident workflow + per-turn pointer push the driver
# to emit the right tool; this section is the *detector* for when it did not.
#
# It is deliberately a PURE library — a map plus functions over a turn's tool-call set — with NO
# live consumer here. The proxy has no gate/correction layer by design (see
# docs/agent-gates-proposal.md: in-band correction induces servility in frontier models; invariants
# live at the tool boundary). An offline consumer (the monitor) scores each logged turn with
# `adjudicate`; a future live use must stay tier-aware and off the user role.
#
# The model: scan a turn's tool-call multiset and classify it —
#   - stall      : a tool-owed phase yielded on bare prose (no edge at all)
#   - wrong_edge : an interactive card appeared, but not the one the phase owes
#   - not_solo   : a control-transfer hand-off carried text or other tool calls (must be solo)
#   - batching   : >1 distinct step slug signalled in one turn
# Phases absent from `_REQUIRED` are LENIENT (a genuine free-text question is expected). The map is
# SUBAGENT-only (sdg/training/eval); the orchestrator's phase is proxy control-flow, not a signalled
# step, so it has no reliable (role, step) key and is never adjudicated. Keys use the resident
# `<!--signal:slug-->` vocabulary.

# Tools that transfer control to someone else (so the turn is never a stall) …
_CONTROL_TRANSFER: frozenset[str] = frozenset(
    {"delegate_to_subagent", "request_orchestrator_help", "signal_subagent_completion"}
)
# … and tools that render an interactive card the user acts on.
_CARDS: frozenset[str] = frozenset(
    {
        "present_options",
        "show_prompt",
        "validate_sdg_job",
        "validate_training_job",
        "validate_eval_job",
    }
)
# Any of these means control is accounted for (not a bare-prose stall).
_EDGES: frozenset[str] = _CONTROL_TRANSFER | _CARDS


@dataclass(frozen=True)
class Owed:
    """What a tool-owed phase must emit in a turn.

    `any_of` — at least one must appear (the common case: a single required edge).
    `all_of` — every one must appear (today only the task-distillation system-prompt review step,
               which owes BOTH `show_prompt` and the approval `present_options`).
    `solo`   — the turn must be ONLY the required tool, no text or other tool calls (a pure
               control-transfer hand-off — e.g. eval's held-out-request-help, whose resident
               section forbids narrating the hand-off).
    """

    any_of: frozenset[str] = frozenset()
    all_of: frozenset[str] = frozenset()
    solo: bool = False


# Tool-owed phases, keyed (role, signal slug). A (role, slug) NOT here is lenient. `gather` is
# absent on purpose — its obligation is per skill step, resolved by `_skill_owed`.
_REQUIRED: dict[tuple[str, str], Owed] = {
    # sdg
    ("sdg", "route"): Owed(
        any_of=frozenset({"skill", "present_options", "clone_sdg_config_for_eval"})
    ),
    ("sdg", "validate"): Owed(any_of=frozenset({"validate_sdg_job"})),
    ("sdg", "complete"): Owed(any_of=frozenset({"signal_subagent_completion"})),
    # training
    ("training", "route"): Owed(any_of=frozenset({"skill", "present_options"})),
    ("training", "validate"): Owed(any_of=frozenset({"validate_training_job"})),
    ("training", "complete"): Owed(any_of=frozenset({"signal_subagent_completion"})),
    # eval
    ("eval", "eval-dataset"): Owed(any_of=frozenset({"present_options"})),
    ("eval", "validate"): Owed(any_of=frozenset({"validate_eval_job"})),
    ("eval", "report"): Owed(any_of=frozenset({"present_options"})),
    ("eval", "complete"): Owed(any_of=frozenset({"signal_subagent_completion"})),
    ("eval", "held-out-request-help"): Owed(
        any_of=frozenset({"request_orchestrator_help"}), solo=True
    ),
}


def _skill_owed(skill_name: str | None, step_value: str | None) -> Owed | None:
    """The obligation for an active stepped-skill gather step, or None when the slug is not a step
    of the skill. Every gather step owes `present_options`; a review step (its body drives
    `show_prompt`) owes BOTH `show_prompt` and `present_options`."""
    if skill_step_title(skill_name, step_value) is None:
        return None
    if skill_step_is_review(skill_name, step_value):
        return Owed(all_of=frozenset({"show_prompt", "present_options"}))
    return Owed(any_of=frozenset({"present_options"}))


def owed_for(
    role: str,
    phase_step: str,
    *,
    skill: str | None = None,
    active_skill_step: str | None = None,
) -> Owed | None:
    """Resolve what a (role, phase_step) turn owes, or None when the phase is lenient.

    Phase-level entries win; the `gather` phase delegates to the active skill step (caller supplies
    `skill` + `active_skill_step`, which it tracks via the skill-step cursor — a stalling turn
    emits no step slug of its own)."""
    owed = _REQUIRED.get((role, phase_step))
    if owed is not None:
        return owed
    if phase_step == "gather":
        return _skill_owed(skill, active_skill_step)
    return None


@dataclass(frozen=True)
class Verdict:
    """Outcome of adjudicating one turn. `kind` is one of: "ok", "stall", "wrong_edge",
    "not_solo", "batching". `owed` is the obligation that was checked (None for a lenient phase),
    for building a diagnostic message."""

    kind: str
    owed: Owed | None = None
    detail: str = ""


_OK = Verdict("ok")


def requires_phase_signal(role: str) -> bool:
    """Subagents must stamp `signal_phase` every turn (it keys the phase); the orchestrator never
    signals (its phase is control-flow). A missing signal on a subagent turn means the phase is
    unattributable — a separate failure, upstream of `adjudicate`, checkable with this."""
    return role in _SUBAGENTS


def adjudicate(
    role: str,
    phase_step: str,
    tools: Collection[str],
    *,
    skill: str | None = None,
    active_skill_step: str | None = None,
    awaiting_job: bool = False,
    has_text: bool = False,
    present_steps: Sequence[str] = (),
) -> Verdict:
    """Classify one turn from its emitted tool-call set.

    `tools` — tool names the turn emitted (the caller should EXCLUDE errored calls). `awaiting_job`
      — a non-terminal job is in scope being awaited (waiting is not a stall). `has_text` — the turn
      also emitted assistant prose, for the `solo` hand-off rule. `present_steps` — the `step` slugs
      of each card this turn, to catch batching (>1 distinct step signalled in one turn).

    Returns a `Verdict`; a lenient phase (or an unknown slug) is always "ok"."""
    tool_set = set(tools)

    # Batching is orthogonal to the owed-edge check and applies in any phase.
    if len({s for s in present_steps if s}) > 1:
        return Verdict("batching", detail="multiple step slugs signalled in one turn")

    owed = owed_for(role, phase_step, skill=skill, active_skill_step=active_skill_step)
    if owed is None:
        return _OK  # lenient phase: bare prose is a legitimate free-text question

    # Solo hand-off: the turn must be ONLY the required call — no prose, no other tool calls. The
    # mandatory per-turn `signal_phase` is NOT a co-tool (every subagent turn stamps it; requiring
    # its absence would make `solo` unsatisfiable), so it is excluded.
    if owed.solo and (owed.any_of & tool_set):
        extraneous = tool_set - owed.any_of - {"signal_phase"}
        if has_text or extraneous:
            return Verdict("not_solo", owed, "hand-off turn carried text or other tool calls")

    # Obligation satisfied?
    if owed.all_of and owed.all_of <= tool_set:
        return Verdict("ok", owed)
    if owed.any_of & tool_set:
        return Verdict("ok", owed)

    # A control transfer always accounts for control — even when it is not THIS phase's owed edge it
    # is a bail, not a stall. (Whether such a handback is premature is a STATE question for the
    # tool-boundary gate, not here.)
    if _CONTROL_TRANSFER & tool_set:
        return Verdict("ok", owed)

    # A card appeared but not the owed one → wrong edge. No edge at all, and no job awaited → stall.
    if _EDGES & tool_set:
        return Verdict("wrong_edge", owed, "an interactive card appeared but not the owed one")
    if awaiting_job:
        return _OK  # a job is in flight and the agent is waiting on it
    return Verdict("stall", owed, "tool-owed phase yielded with no edge")
