"""Drift guard for the resident-workflow signal-slug convention (PR #526 follow-up).

The per-turn pointer (`api/phases.py`) can only name a subagent's current step if the slug the
prompt tells it to signal (via `signal_phase`) actually resolves to a `### ` heading — a
`## Workflow` section OR, when a skill is loaded, one of its `### Step`s (skill steps now ride the
same signal cursor). That link is implicit — it lives in places that can silently drift apart:

  1. the slugs named in a workflow.md's `## Progress Signal` (what the agent is TOLD to send),
  2. the `<!--signal:slug-->` markers on `## Workflow` and SKILL.md `### ` headings (what the
     pointer resolves AGAINST), and
  3. the `present_options(step="…")` stamps a SKILL.md instructs (a legacy slug source for a
     heading without a marker).

If a heading loses its marker, or an instructed slug maps to no heading, the pointer degrades to a
generic "find your current step" with no section named — silently, with no error. And because the
pointer resolves a signalled slug against workflow sections FIRST, a skill step slug that shadows a
workflow phase slug would be mis-resolved. GLM-5.3-flash is exactly the driver that needs the named
step, so this guards the convention at build time: every instructed slug resolves, every
`## Workflow`/`### Step` heading carries a marker, and skill and workflow slug spaces stay disjoint.
See memory `phase-injection-implementation` and `pr526-resident-inject-flash`.
"""

from __future__ import annotations

import re

from amortized.api import phases

_SUBAGENTS = ("sdg", "training", "eval")
# Backtick tokens inside `## Progress Signal` that look like a signal slug (lowercase, hyphenated).
# Excludes `signal_phase`, `phase="sdg"`, `<!--signal:…-->`, etc. Each must resolve to a workflow
# section OR a loaded skill's `### Step` — the two things the agent can legitimately signal.
_BACKTICK = re.compile(r"`([^`]+)`")
_SLUGLIKE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# Backticked field/tool names that appear in the instruction prose but are NOT signal slugs.
_NON_SLUG_WORDS = frozenset({"step", "phase"})


def _progress_signal_slugs(agent: str) -> set[str]:
    """Slug-like backtick tokens the workflow.md's `## Progress Signal` section instructs."""
    text = phases._workflow_path(agent).read_text(encoding="utf-8")
    slugs: set[str] = set()
    in_section = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            in_section = s[3:].strip().lower() == "progress signal"
            continue
        if in_section:
            for tok in _BACKTICK.findall(s):
                if _SLUGLIKE.match(tok) and tok not in _NON_SLUG_WORDS:
                    slugs.add(tok)
    return slugs


def _workflow_section_slugs(agent: str) -> set[str]:
    return {slug for slug, _title in phases._sections(agent)}


def _agent_skill_step_slugs(agent: str) -> set[str]:
    """Every `### Step` slug across the skills that belong to this agent."""
    out: set[str] = set()
    for name, (ag, _reldir) in phases._skill_index().items():
        if ag == agent:
            out |= {slug for slug, _title in phases._skill_steps(name)}
    return out


def _workflow_headings(agent: str) -> list[str]:
    """Raw (marker-retaining) `### ` heading text for each heading under `## Workflow`."""
    text = phases._workflow_path(agent).read_text(encoding="utf-8")
    headings: list[str] = []
    in_workflow = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            in_workflow = s[3:].strip().lower() == "workflow"
            continue
        if in_workflow and s.startswith("### "):
            headings.append(s[4:].strip())
    return headings


class TestWorkflowSignalMarkers:
    def test_every_workflow_heading_carries_a_signal_marker(self) -> None:
        # No `### ` under `## Workflow` may rely on the normalized-title fallback: a title tweak
        # would then silently change the slug the pointer resolves against.
        missing: list[str] = []
        for agent in _SUBAGENTS:
            for raw in _workflow_headings(agent):
                if not phases._SIGNAL_MARKER.search(raw):
                    missing.append(f"{agent}: {raw}")
        assert not missing, f"`## Workflow` headings without a <!--signal:--> marker: {missing}"

    def test_every_progress_signal_slug_resolves(self) -> None:
        # Every slug the `## Progress Signal` section names (in backticks) must resolve to something
        # the agent can legitimately signal: a workflow section OR one of its skills' `### Step`s.
        # A dangling instructed slug would silently degrade the pointer to generic.
        for agent in _SUBAGENTS:
            wf = _workflow_section_slugs(agent)
            skill_slugs = _agent_skill_step_slugs(agent)
            instructed = _progress_signal_slugs(agent)
            assert instructed, f"{agent} workflow.md has no `## Progress Signal` slugs"
            dangling = {s for s in instructed if s not in wf and s not in skill_slugs}
            assert not dangling, (
                f"{agent}: Progress-Signal slug(s) resolve to no workflow section or skill step: "
                f"{dangling}"
            )

    def test_every_workflow_section_slug_is_instructed(self) -> None:
        # The reverse for WORKFLOW slugs (skills are example-listed, not exhaustively): every
        # `## Workflow` section slug must be named in `## Progress Signal`, so no phase is a dead,
        # never-signalled step.
        for agent in _SUBAGENTS:
            missing = _workflow_section_slugs(agent) - _progress_signal_slugs(agent)
            assert not missing, (
                f"{agent}: `## Workflow` slug(s) not named in `## Progress Signal`: {missing}"
            )

    def test_skill_slugs_do_not_shadow_workflow_slugs(self) -> None:
        # try-then-fallback resolves a signalled slug against workflow sections FIRST, so a skill
        # step slug equal to a workflow phase slug would be mis-resolved (wrong title + obligation).
        # Keep the two slug spaces disjoint per agent.
        for agent in _SUBAGENTS:
            clash = _workflow_section_slugs(agent) & _agent_skill_step_slugs(agent)
            assert not clash, (
                f"{agent}: skill step slug(s) shadow a `## Workflow` section slug: {clash}"
            )


class TestSkillStepMarkers:
    def test_every_step_heading_resolves(self) -> None:
        # Every `### Step …` heading in a SKILL.md (its gather steps, by convention) must surface in
        # `_skill_steps` — i.e. carry a marker or a `present_options` stamp. Reference headings
        # (`### columns`, `### document_ids`, …) are not "Step" and are correctly skipped.
        for name in sorted(phases._skill_index()):
            agent, reldir = phases._skill_index()[name]
            path = phases._get_base_dir() / "agents" / agent / "skills" / reldir / "SKILL.md"
            titles = {title for _slug, title in phases._skill_steps(name)}
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("### Step "):
                    title = phases._SIGNAL_MARKER.sub("", s[4:]).strip()
                    assert title in titles, (
                        f"{name}: step heading {title!r} resolves to no slug "
                        "(needs a <!--signal:--> marker or a present_options stamp)"
                    )

    def test_every_present_options_stamp_resolves(self) -> None:
        # Every `present_options(step="X")` a SKILL.md instructs must map to a step the pointer can
        # name, so the in-gather skill clause never points at a stamp the agent sends but can't
        # resolve.
        stamp_re = re.compile(r"present_options[^\n]*?\bstep\s*[=:]\s*[\"']([^\"']+)[\"']", re.I)
        for name in sorted(phases._skill_index()):
            agent, reldir = phases._skill_index()[name]
            path = phases._get_base_dir() / "agents" / agent / "skills" / reldir / "SKILL.md"
            for stamp in stamp_re.findall(path.read_text(encoding="utf-8")):
                assert phases.skill_step_title(name, stamp) is not None, (
                    f"{name}: present_options step {stamp!r} resolves to no SKILL.md step"
                )
