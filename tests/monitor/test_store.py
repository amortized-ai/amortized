"""Tests for the SQLite store: ingest idempotency, cross-axis queries, export."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from monitor.scorer import store
from monitor.scorer.matchers import score_run
from monitor.scorer.model import load_checklist, load_runs

_FIXTURES = Path(__file__).parent / "fixtures"


def _ingest(db: Path):
    runs = load_runs(_FIXTURES / "logs")
    checklist = load_checklist("general")
    scores = {r.session_id: score_run(r, checklist) for r in runs}
    store.ingest(db, runs, use_case="general", checklist=checklist, scores=scores)
    return runs, checklist, scores


def test_ingest_populates_tables(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    _ingest(db)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM row_scores").fetchone()[0] > 0
        model = conn.execute("SELECT model FROM runs WHERE session_id = 'sess-alpha'").fetchone()[0]
        assert model == "claude-opus-4-8"
    finally:
        conn.close()


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    _ingest(db)
    _ingest(db)  # second pass must not duplicate
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        # tool_calls keyed by (session, turn, call) -> no dupes
        n = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM tool_calls "
            "GROUP BY session_id, turn_idx, call_idx HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        assert n == 0
    finally:
        conn.close()


def test_cross_axis_query(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    _ingest(db)
    conn = sqlite3.connect(db)
    try:
        # per-model met-rate across the use case (axis E query shape)
        rows = conn.execute(
            "SELECT model, "
            "SUM(status='met') AS met, "
            "SUM(status IN ('met','wrong','missed')) AS scored "
            "FROM row_scores GROUP BY model ORDER BY model"
        ).fetchall()
        models = {r[0] for r in rows}
        assert "claude-opus-4-8" in models
        assert "claude-sonnet-5-5" in models
        # the failing beta run has 0 met
        beta = next(r for r in rows if r[0] == "claude-sonnet-5-5")
        assert beta[1] == 0
        # per-role tool_calls breakdown
        roles = {r[0] for r in conn.execute("SELECT DISTINCT role FROM tool_calls").fetchall()}
        assert {"sdg", "training", "eval"} <= roles
    finally:
        conn.close()


def test_full_input_persisted_as_json(tmp_path: Path) -> None:
    # synthesise a run whose tool call carries a full redacted input
    from monitor.scorer.model import Run

    run = Run(
        session_id="s-input",
        turns=[
            {
                "ts": "2026-01-01T00:00:00",
                "role": "sdg",
                "tokens": {},
                "tool_calls": [
                    {
                        "tool": "validate_sdg_job",
                        "role": "sdg",
                        "input": {"mode": "preview", "config": {"n": 5}},
                    }
                ],
            }
        ],
    )
    db = tmp_path / "x.sqlite"
    store.ingest(db, [run], use_case="general", checklist={"rows": []}, scores={})
    conn = sqlite3.connect(db)
    try:
        raw = conn.execute(
            "SELECT input_json FROM tool_calls WHERE session_id = 's-input'"
        ).fetchone()[0]
        assert json.loads(raw)["config"]["n"] == 5
    finally:
        conn.close()


def test_export_json_and_csv(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    _ingest(db)

    jdir = tmp_path / "json"
    store.export(db, "json", jdir)
    runs = json.loads((jdir / "runs.json").read_text())
    assert len(runs) == 2
    # job_created records from the fixture land in the jobs table
    assert len(json.loads((jdir / "jobs.json").read_text())) == 3
    # empty table still exports (no rubric scores ingested here)
    assert json.loads((jdir / "rubric_scores.json").read_text()) == []

    cdir = tmp_path / "csv"
    store.export(db, "csv", cdir)
    text = (cdir / "row_scores.csv").read_text()
    assert "status" in text.splitlines()[0]
    # empty rubric_scores table still writes a header row
    assert (cdir / "rubric_scores.csv").read_text().splitlines()[0].startswith("session_id")
