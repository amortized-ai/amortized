"""Tests for the slice-free resident-workflow POINTER (api/phases.py).

The pointer re-anchors a leaf subagent to its RESIDENT `## Workflow` section each turn, on the
system channel. It derives section titles from the resident workflow.md headings themselves (no
`phases/` slices), names the entry section for a fresh delegation, names a matched section when the
signalled step maps to a heading, and stays generic (no WRONG section) for the coarse UI progress
vocabulary. See memory `pr526-resident-inject-flash`.
"""

from amortized.api import phases


class TestSections:
    def test_parses_marker_slugs_under_workflow_only(self) -> None:
        # Real resident files: sdg/training are 4-phase; eval is a finer 8-step sequence. The slug
        # is the heading's <!--signal:…--> marker; the title has the marker stripped.
        assert phases._sections("sdg")[:2] == (
            ("route", "Phase 1 — Route to Sub-Skill"),
            ("gather", "Phase 2 — Gather Requirements"),
        )
        # 8 distinct eval slugs (the held-out branch reuses the `eval-dataset` slug).
        assert len({slug for slug, _t in phases._sections("eval")}) == 8
        assert phases._sections("eval")[0] == (
            "understand-task",
            "Step 0 — Confirm WHAT to evaluate",
        )
        # Headings OUTSIDE `## Workflow` (e.g. sdg's `### How to Choose`) are not sections.
        titles = [t for _slug, t in phases._sections("sdg")]
        assert "How to Choose" not in titles

    def test_unknown_agent_has_no_sections(self) -> None:
        assert phases._sections("nope-not-an-agent") == ()


class TestNorm:
    def test_collapses_em_dash_and_ordering_prefix(self) -> None:
        assert phases._norm("Step 4 — Validate and submit") == "step-4-validate-and-submit"
        assert phases._norm("03-output-format") == "output-format"
        assert phases._norm("understand_task") == "understand-task"


class TestResolveTitle:
    def test_fresh_delegation_names_entry_section(self) -> None:
        assert phases._resolve_title("eval", "") == "Step 0 — Confirm WHAT to evaluate"
        assert phases._resolve_title("sdg", None) == "Phase 1 — Route to Sub-Skill"

    def test_marker_slug_match(self) -> None:
        assert phases._resolve_title("eval", "validate") == "Step 4 — Validate and submit"
        assert phases._resolve_title("eval", "understand-task") == (
            "Step 0 — Confirm WHAT to evaluate"
        )
        assert phases._resolve_title("sdg", "gather") == "Phase 2 — Gather Requirements"
        # Underscore/case variants normalize to the same slug.
        assert phases._resolve_title("eval", "eval_dataset") == "Step 1 — Confirm the eval dataset"

    def test_unknown_step_names_no_section(self) -> None:
        # A step that matches no marker (e.g. the old coarse vocab) → None, so the pointer stays
        # generic rather than claiming a wrong section.
        assert phases._resolve_title("eval", "gather_requirements") is None
        assert phases._resolve_title("eval", "estimate_cost") is None


class TestSubagentPointer:
    def test_none_for_orchestrator(self) -> None:
        assert phases.subagent_pointer("orchestrator", "") is None
        assert phases.subagent_pointer(None, "") is None

    def test_none_for_unknown_agent(self) -> None:
        assert phases.subagent_pointer("nope-not-an-agent", "") is None

    def test_fresh_pointer_names_entry_and_carries_imperatives(self) -> None:
        p = phases.subagent_pointer("eval", "")
        assert p is not None
        assert p.startswith("[ACTIVE PHASE]")
        assert "Step 0 — Confirm WHAT to evaluate" in p
        # The handback/completion imperative (CALL, don't narrate) and the ordering guard.
        assert "request_orchestrator_help" in p
        assert "CALL the tool" in p
        assert "Do not work ahead" in p

    def test_generic_pointer_omits_section_but_keeps_imperatives(self) -> None:
        p = phases.subagent_pointer("eval", "gather_requirements")
        assert p is not None
        assert "current step is" not in p  # no (wrong) section named
        assert "request_orchestrator_help" in p
        assert "Do not work ahead" in p

    def test_matched_step_names_its_section(self) -> None:
        p = phases.subagent_pointer("sdg", "gather")
        assert p is not None
        assert "Phase 2 — Gather Requirements" in p

    def test_skill_clause_names_current_skill_step(self) -> None:
        # When a skill is loaded during gathering, the pointer also names the resident SKILL.md step
        # the agent is on (resolved from the present_options stamp it emits — no new slugs needed).
        p = phases.subagent_pointer("sdg", "gather", "sdg-knowledge-ingestion", "samples")
        assert p is not None
        assert "sdg-knowledge-ingestion" in p
        assert "Step 5 — How many samples?" in p
        assert "ONE question per turn" in p

    def test_no_skill_clause_without_active_skill(self) -> None:
        p = phases.subagent_pointer("sdg", "gather")
        assert p is not None
        assert "skill loaded" not in p
