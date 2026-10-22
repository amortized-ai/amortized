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
# sequence. A step's slug comes from a `<!--signal:…-->` heading marker — the preferred convention,
# identical to the `## Workflow` sections — and, for a skill that carries none, falls back to the
# `present_options(step="…")` stamp the driver is already told to emit for that question (so an
# existing stamped skill needs no edit, while a new one should mark its headings like the workflow
# sections do). Either way the agent sends that stamp per turn; this resolves slug → the step's
# heading title, and the agent.py scan records the active skill (from the `skill` tool call) and
# the latest stamp (from present_options) per turn. `tests/test_phase_marker_drift.py` fails loudly
# if an instructed slug no longer resolves.

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
