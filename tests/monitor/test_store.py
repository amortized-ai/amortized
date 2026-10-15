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


def test_ingest_preserves_other_use_case(tmp_path: Path) -> None:
    # Re-ingesting one use case must not drop the same session's other-use-case rows.
    db = tmp_path / "runs.sqlite"
    runs = load_runs(_FIXTURES / "logs")
    general = load_checklist("general")
    rfe = load_checklist("rfe_assess")
    gscores = {r.session_id: score_run(r, general) for r in runs}
    rscores = {r.session_id: score_run(r, rfe) for r in runs}

    store.ingest(db, runs, use_case="general", checklist=general, scores=gscores)
    store.ingest(db, runs, use_case="rfe_assess", checklist=rfe, scores=rscores)
    # re-ingest general again: must keep the rfe_assess rows
    store.ingest(db, runs, use_case="general", checklist=general, scores=gscores)

    conn = sqlite3.connect(db)
    try:
        use_cases = {
            r[0] for r in conn.execute("SELECT DISTINCT use_case FROM row_scores").fetchall()
        }
        assert use_cases == {"general", "rfe_assess"}
        # runs now holds both use cases for the same session
        n_runs = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE session_id = 'sess-alpha'"
        ).fetchone()[0]
        assert n_runs == 2
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
        # the failing beta run scores far worse than the clean alpha run
        beta = next(r for r in rows if r[0] == "claude-sonnet-5-5")
        alpha = next(r for r in rows if r[0] == "claude-opus-4-8")
        assert beta[1] < alpha[1]  # fewer met
        assert beta[1] <= 1  # at most the no-stall freebie (it errored out immediately)
        # per-role tool_calls breakdown
        roles = {r[0] for r in conn.execute("SELECT DISTINCT role FROM tool_calls").fetchall()}
        assert {"sdg", "training", "eval"} <= roles
    finally:
        conn.close()


def test_promoted_fields_persisted(tmp_path: Path) -> None:
    # we store only allowlisted promoted fields (no raw input dump)
    from monitor.scorer.model import Run

    run = Run(
        session_id="s-input",
        turns=[
            {
                "ts": "2026-01-01T00:00:00",
                "role": "sdg",
                "tokens": {},
                "tool_calls": [
                    {"tool": "validate_sdg_job", "role": "sdg", "mode": "preview"},
                    {"tool": "skill", "role": "sdg", "skill": "sdg-task-distillation"},
                    {"tool": "get_job", "role": "sdg", "job_id": "j-1"},
                ],
            }
        ],
    )
    db = tmp_path / "x.sqlite"
    store.ingest(db, [run], use_case="general", checklist={"rows": []}, scores={})
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT tool, mode, skill, job_id FROM tool_calls "
            "WHERE session_id = 's-input' ORDER BY call_idx"
        ).fetchall()
        assert rows == [
            ("validate_sdg_job", "preview", None, None),
            ("skill", None, "sdg-task-distillation", None),
            ("get_job", None, None, "j-1"),
        ]
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
