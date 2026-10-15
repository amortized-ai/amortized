"""Tests for the SKILL.md action contract loader + action_emitted matcher."""

from __future__ import annotations

from typing import Any

from monitor.scorer import skills
from monitor.scorer.matchers import score_auto
from monitor.scorer.model import Run


def _run(*tool_calls: dict[str, Any]) -> Run:
    return Run(
        session_id="t",
        turns=[{"ts": "2026-01-01T00:00:00", "role": "sdg", "tool_calls": list(tool_calls)}],
    )


def test_load_skill_contracts_from_skill_md() -> None:
    contracts = skills.load_skill_contracts()
    # the six native skills on main carry the action contract
    assert "sdg-task-distillation" in contracts
    td = contracts["sdg-task-distillation"]
    assert td["action_required"] is True
    assert td["emits"] == ["validate_sdg_job"]
    assert td["stage"] == "sdg"
    osft = contracts["training-knowledge-ingestion-osft"]
    assert osft["emits"] == ["validate_training_job"]


def test_deliverable_tools_union() -> None:
    tools = skills.deliverable_tools()
    assert "validate_sdg_job" in tools
    assert "validate_training_job" in tools


def test_deliverable_tools_fallback(tmp_path: Any) -> None:
    # no agents tree -> fall back to the known validate_* cards
    tools = skills.deliverable_tools(tmp_path / "nope")
    assert tools == {"validate_sdg_job", "validate_training_job", "validate_eval_job"}


def test_action_emitted_matcher() -> None:
    m = {"type": "action_emitted", "emits": ["validate_sdg_job"]}
    assert score_auto(_run({"tool": "validate_sdg_job", "role": "sdg"}), m) == "met"
    assert score_auto(_run({"tool": "present_options", "role": "sdg"}), m) == "missed"
    assert score_auto(_run(), m) == "missed"
