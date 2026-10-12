"""Golden regression test for `score_run` over a fixture log dir.

The expected statuses were generated from the scorer immediately after the
Phase 0 package extraction, which was verified byte-for-byte against the original
`process_monitor_logs.py`. This locks the full per-row scoring so future refactors
can't silently change a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

from monitor.scorer.matchers import score_run
from monitor.scorer.model import load_checklist, load_runs

_FIXTURES = Path(__file__).parent / "fixtures"


def test_score_run_matches_golden() -> None:
    runs = load_runs(_FIXTURES / "logs")
    checklist = load_checklist("general")
    scores = {run.session_id: score_run(run, checklist) for run in runs}

    expected = json.loads((_FIXTURES / "expected_scores_general.json").read_text())
    assert scores == expected


def test_load_runs_reads_turns_and_completion() -> None:
    runs = {r.session_id: r for r in load_runs(_FIXTURES / "logs")}
    alpha = runs["sess-alpha"]
    assert alpha.outcome == "success"
    assert alpha.turns_to_complete == 4
    assert alpha.agent_model == "claude-opus-4-8"
    # tokens exclude cache-read (alpha turn 1 has cache=200 that must not count)
    assert alpha.total_tokens == 810
