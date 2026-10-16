"""Unit tests for the per-phase instruction-block catalog (api/phases.py).

The catalog reads `agents/<agent>/phases/<phase>.md` and prepends the block to a
turn message so the active step's instructions ride in the freshest position. The
loader is opt-in by file presence (an unmigrated agent gets no injection) and keyed
one-file-per-section (the key is the slug the agent reports via signal_phase.step).
"""

from pathlib import Path

import pytest

from amortized.api import phases
from amortized.config import settings


@pytest.fixture()
def phase_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway base dir whose agents/<x>/phases tree the loader reads from."""
    monkeypatch.setattr(settings, "recipes_dir", tmp_path)
    phases._load.cache_clear()
    yield tmp_path
    phases._load.cache_clear()


def _write(root: Path, agent: str, phase: str, body: str) -> None:
    d = root / "agents" / agent / "phases"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{phase}.md").write_text(body, encoding="utf-8")


class TestPhaseBlock:
    def test_loads_existing_block(self, phase_root: Path) -> None:
        _write(phase_root, "eval", "step-1-dataset", "  Confirm the eval dataset  ")
        # stripped of surrounding whitespace
        assert phases.phase_block("eval", "step-1-dataset") == "Confirm the eval dataset"

    def test_missing_file_returns_none(self, phase_root: Path) -> None:
        assert phases.phase_block("eval", "no-such-phase") is None

    def test_empty_file_returns_none(self, phase_root: Path) -> None:
        _write(phase_root, "sdg", "phase-1-route", "   \n  ")
        assert phases.phase_block("sdg", "phase-1-route") is None

    def test_blank_agent_or_phase_returns_none(self, phase_root: Path) -> None:
        assert phases.phase_block("", "step-1-dataset") is None
        assert phases.phase_block("eval", "") is None


class TestSubagentPhase:
    def test_known_step_passes_through(self) -> None:
        assert phases.subagent_phase("eval", "step-3-metrics") == "step-3-metrics"

    def test_empty_step_resolves_to_entry(self) -> None:
        assert phases.subagent_phase("eval", "") == "step-0-what-to-eval"
        assert phases.subagent_phase("sdg", None) == "phase-1-route"
        assert phases.subagent_phase("training", "") == "phase-1-route"

    def test_unregistered_agent_has_no_entry(self) -> None:
        assert phases.subagent_phase("orchestrator", "") is None


class TestWithBlock:
    def test_prepends_block(self, phase_root: Path) -> None:
        _write(phase_root, "eval", "step-0-what-to-eval", "ENTRY BLOCK")
        out = phases.with_block("eval", "step-0-what-to-eval", "turn text")
        assert out == "ENTRY BLOCK\n\nturn text"

    def test_noop_when_block_absent(self, phase_root: Path) -> None:
        # Unmigrated agent / unknown phase → the turn text is returned unchanged.
        assert phases.with_block("eval", "ghost", "turn text") == "turn text"

    def test_noop_when_phase_is_none(self, phase_root: Path) -> None:
        assert phases.with_block("orchestrator", None, "turn text") == "turn text"

    def test_block_only_when_text_empty(self, phase_root: Path) -> None:
        _write(phase_root, "orchestrator", "resume", "RESUME BLOCK")
        assert phases.with_block("orchestrator", "resume", "") == "RESUME BLOCK"


class TestRealCatalog:
    """The real files shipped under agents/ — every migrated phase must load, so a
    renamed/removed section can't silently drop its injected block."""

    def test_every_shipped_phase_loads(self) -> None:
        phases._load.cache_clear()
        expected = {
            "eval": [
                "step-0-what-to-eval",
                "step-1-dataset",
                "step-1_5-how-much",
                "step-2-model-source",
                "step-3-metrics",
                "step-4-validate",
                "step-5-report",
                "step-6-complete",
                "held-out-eval-set",
            ],
            "sdg": ["phase-1-route", "phase-2-gather", "phase-3-validate", "phase-4-complete"],
            "training": [
                "phase-1-route",
                "phase-2-gather",
                "phase-3-validate",
                "phase-4-complete",
            ],
            "orchestrator": ["intake", "delegate", "fulfill_help", "resume", "monitor"],
        }
        for agent, keys in expected.items():
            for key in keys:
                assert phases.phase_block(agent, key), f"missing/empty: {agent}/{key}"

    def test_fulfill_help_has_only_known_placeholders(self) -> None:
        # fulfill_help.md is .format()'d by the proxy, so it must carry exactly the
        # {target}/{need}/{context} placeholders and no stray braces.
        phases._load.cache_clear()
        block = phases.phase_block("orchestrator", "fulfill_help")
        assert block is not None
        block.format(target="eval", need="a held-out set", context="bg")
