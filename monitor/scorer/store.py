"""Structured SQLite store + flat export for comparative analysis.

JSONL stays the write-ahead capture format (the online path must never block on a
DB). This module ingests those logs — plus the scored rows — into a queryable
SQLite database so cross-run / cross-model / cross-use-case / per-role comparison
is plain SQL, and exports flat JSON/CSV (and parquet when pandas is available) for
notebooks.

Ingest is idempotent: re-ingesting a session replaces its rows, so pointing it at
a growing log dir repeatedly is safe.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from monitor.scorer.model import Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    session_id TEXT PRIMARY KEY,
    use_case TEXT, model TEXT, outcome TEXT,
    turns INTEGER, tokens INTEGER, cost REAL,
    wall_clock_s REAL, avg_latency_s REAL
);
CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT, turn_idx INTEGER, role TEXT, model TEXT, provider TEXT,
    input_tok INTEGER, output_tok INTEGER, reasoning_tok INTEGER, cache_tok INTEGER,
    cost REAL, duration_ms INTEGER,
    PRIMARY KEY (session_id, turn_idx)
);
CREATE TABLE IF NOT EXISTS tool_calls (
    session_id TEXT, turn_idx INTEGER, call_idx INTEGER,
    role TEXT, tool TEXT, status TEXT,
    target TEXT, mode TEXT, parent_job_id TEXT, judge TEXT, input_json TEXT,
    PRIMARY KEY (session_id, turn_idx, call_idx)
);
CREATE TABLE IF NOT EXISTS texts (
    session_id TEXT, turn_idx INTEGER, text_idx INTEGER, role TEXT, text TEXT,
    PRIMARY KEY (session_id, turn_idx, text_idx)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, session_id TEXT, type TEXT, status TEXT,
    parent_job_id TEXT, mlflow_run_id TEXT, user_id TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS row_scores (
    session_id TEXT, row_id TEXT, use_case TEXT,
    model TEXT, aspect TEXT, role TEXT, adjudicate TEXT, status TEXT, reason TEXT,
    PRIMARY KEY (session_id, row_id, use_case)
);
CREATE TABLE IF NOT EXISTS rubric_scores (
    session_id TEXT, use_case TEXT, role TEXT, skill TEXT, step_id TEXT,
    status TEXT, weight REAL, reason TEXT, adherence_pct REAL,
    PRIMARY KEY (session_id, use_case, skill, step_id)
);
"""

_RAW_TABLES = ("turns", "tool_calls", "texts")
_EXPORT_TABLES = (
    "runs",
    "turns",
    "tool_calls",
    "texts",
    "jobs",
    "row_scores",
    "rubric_scores",
)


def _row_role(row: dict[str, Any]) -> str | None:
    """Best-effort subagent role a checklist row pertains to (for per-role
    analysis): the `requires.role` gate, else `match.where.role`, else None."""
    req = row.get("requires") or {}
    if req.get("role"):
        return str(req["role"])
    where = (row.get("match") or {}).get("where") or {}
    if where.get("role"):
        return str(where["role"])
    return None


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def _delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    for table in ("runs", *_RAW_TABLES, "row_scores", "rubric_scores"):
        conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
    # jobs are correlated by session too
    conn.execute("DELETE FROM jobs WHERE session_id = ?", (session_id,))


def ingest(
    db_path: Path,
    runs: list[Run],
    *,
    use_case: str,
    checklist: dict[str, Any],
    scores: dict[str, dict[str, str]],
    reasons: dict[tuple[str, str], str] | None = None,
) -> None:
    """Write (or replace) the raw + scored rows for each run. Idempotent."""
    reasons = reasons or {}
    rows_by_id = {r["id"]: r for r in checklist["rows"]}
    conn = connect(db_path)
    try:
        for run in runs:
            _delete_session(conn, run.session_id)
            model = run.agent_model
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run.session_id,
                    use_case,
                    model,
                    run.outcome,
                    run.turns_to_complete,
                    run.total_tokens,
                    run.total_cost,
                    run.wall_clock_s,
                    run.avg_latency_s,
                ),
            )
            for ti, turn in enumerate(run.counted_turns):
                tk = turn.get("tokens") or {}
                conn.execute(
                    "INSERT OR REPLACE INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run.session_id,
                        ti,
                        turn.get("role"),
                        turn.get("model"),
                        turn.get("provider"),
                        int(tk.get("input", 0)),
                        int(tk.get("output", 0)),
                        int(tk.get("reasoning", 0)),
                        int(tk.get("cache", 0)),
                        float(turn.get("cost") or 0),
                        turn.get("duration_ms"),
                    ),
                )
                for ci, call in enumerate(turn.get("tool_calls") or []):
                    inp = call.get("input")
                    conn.execute(
                        "INSERT OR REPLACE INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            run.session_id,
                            ti,
                            ci,
                            call.get("role"),
                            call.get("tool"),
                            call.get("status"),
                            call.get("target"),
                            call.get("mode"),
                            call.get("parent_job_id"),
                            call.get("judge"),
                            json.dumps(inp) if inp is not None else None,
                        ),
                    )
                for xi, txt in enumerate(turn.get("texts") or []):
                    conn.execute(
                        "INSERT OR REPLACE INTO texts VALUES (?,?,?,?,?)",
                        (run.session_id, ti, xi, txt.get("role"), txt.get("text")),
                    )
            for row_id, status in scores.get(run.session_id, {}).items():
                row = rows_by_id.get(row_id, {})
                conn.execute(
                    "INSERT OR REPLACE INTO row_scores VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        run.session_id,
                        row_id,
                        use_case,
                        model,
                        row.get("aspect"),
                        _row_role(row),
                        row.get("adjudicate"),
                        status,
                        reasons.get((run.session_id, row_id), ""),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def _fetch_all(db_path: Path, table: str) -> list[dict[str, Any]]:
    conn = connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(f"SELECT * FROM {table}")
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def export(db_path: Path, fmt: str, out: Path) -> None:
    """Export every table to `out/` as `<table>.<fmt>`. fmt: json | csv | parquet."""
    out.mkdir(parents=True, exist_ok=True)
    for table in _EXPORT_TABLES:
        rows = _fetch_all(db_path, table)
        if fmt == "json":
            (out / f"{table}.json").write_text(json.dumps(rows, indent=2) + "\n")
        elif fmt == "csv":
            _write_csv(out / f"{table}.csv", rows)
        elif fmt == "parquet":
            _write_parquet(out / f"{table}.parquet", rows)
        else:
            raise ValueError(f"unknown export format: {fmt!r}")
    print(f"Exported {len(_EXPORT_TABLES)} tables to {out} ({fmt})")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    conn_cols = _EXPORT_COLUMNS.get(path.stem)
    cols = list(rows[0].keys()) if rows else (conn_cols or [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "parquet export needs pandas/pyarrow: uv pip install 'amortized[monitor]'"
        ) from exc
    pd.DataFrame(rows).to_parquet(path, index=False)


# Column order for empty CSV tables (so a table with no rows still emits a header).
_EXPORT_COLUMNS = {
    "jobs": [
        "job_id",
        "session_id",
        "type",
        "status",
        "parent_job_id",
        "mlflow_run_id",
        "user_id",
        "ts",
    ],
    "rubric_scores": [
        "session_id",
        "use_case",
        "role",
        "skill",
        "step_id",
        "status",
        "weight",
        "reason",
        "adherence_pct",
    ],
}
