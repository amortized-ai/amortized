"""Drift guard for the resident-workflow signal-slug convention (PR #526 follow-up).

The per-turn pointer (`api/phases.py`) can only name a subagent's current step if the slug the
prompt tells it to signal actually resolves to a `### ` heading. That link is implicit — it lives
in three separate places that can silently drift apart:

  1. the `## Progress Signal` bullets in a workflow.md (the slugs the agent is TOLD to send),
  2. the `<!--signal:slug-->` markers on the `## Workflow` `### ` headings (what the pointer
     resolves AGAINST), and
  3. the `present_options(step="…")` stamps a SKILL.md instructs (the skill-clause slugs).

If a heading loses its marker, or a bullet/stamp names a slug no heading provides, the pointer
degrades to a generic "find your current step" with no section named — silently, with no error.
GLM-5.3-flash is exactly the driver that needs the named step, so this guards the convention at
build time: every instructed slug must resolve, and every `## Workflow`/`### Step` heading must
carry a marker. See memory `phase-injection-implementation` and `pr526-resident-inject-flash`.
"""

from __future__ import annotations

import re

from amortized.api import phases

_SUBAGENTS = ("sdg", "training", "eval")
# A `## Progress Signal` bullet: `- \`understand-task\` — Step 0 …`
_BULLET_SLUG = re.compile(r"^-\s+`([a-z0-9-]+)`")


def _progress_signal_slugs(agent: str) -> set[str]:
    """The slugs the workflow.md's `## Progress Signal` section tells the agent to send."""
    text = phases._workflow_path(agent).read_text(encoding="utf-8")
    slugs: set[str] = set()
    in_section = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            in_section = s[3:].strip().lower() == "progress signal"
            continue
        if in_section:
            m = _BULLET_SLUG.match(s)
            if m:
                slugs.add(m.group(1))
    return slugs


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

    def test_progress_signal_bullets_match_workflow_sections(self) -> None:
        # Both directions: every instructed slug resolves to a section (no dangling pointer), and
        # every section slug is listed as an instructed bullet (no dead, never-signalled step).
        for agent in _SUBAGENTS:
            bullets = _progress_signal_slugs(agent)
            sections = {slug for slug, _title in phases._sections(agent)}
            assert bullets, f"{agent} workflow.md has no `## Progress Signal` bullets"
            assert bullets == sections, (
                f"{agent}: Progress-Signal slugs {bullets} != `## Workflow` section slugs "
                f"{sections} (symmetric difference {bullets ^ sections})"
            )

    def test_every_instructed_slug_resolves_to_a_title(self) -> None:
        for agent in _SUBAGENTS:
            for slug in _progress_signal_slugs(agent):
                assert phases._resolve_title(agent, slug) is not None, (
                    f"{agent}: instructed slug {slug!r} resolves to no `## Workflow` section"
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
