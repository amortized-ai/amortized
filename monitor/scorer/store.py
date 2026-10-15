"""Structured SQLite store + flat export for comparative analysis.

JSONL stays the write-ahead capture format (the online path must never block on a
DB). This module ingests those logs — plus the scored rows — into a queryable
SQLite database so cross-run / cross-model / cross-use-case / per-role comparison
is plain SQL, and exports flat JSON/CSV (and parquet when pandas is available) for
notebooks.

Ingest is idempotent *per (session, use_case)*: re-ingesting a session under one
use case replaces only that use case's scored rows, so scoring the same logs under
``general`` and ``rfe_assess`` both persist.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from monitor.scorer.model import Run

# Column order per table (single source of truth for schema, INSERTs, and the
# empty-CSV header fallback — so a schema reorder can't silently misalign).
_COLUMNS: dict[str, list[str]] = {
    "runs": [
        "session_id",
        "use_case",
        "model",
        "outcome",
        "turns",
        "tokens",
        "cost",
        "wall_clock_s",
        "avg_latency_s",
    ],
    "turns": [
        "session_id",
        "turn_idx",
        "role",
        "model",
        "provider",
        "input_tok",
        "output_tok",
        "reasoning_tok",
        "cache_tok",
        "cost",
        "duration_ms",
    ],
    "tool_calls": [
        "session_id",
        "turn_idx",
        "call_idx",
        "role",
        "tool",
        "status",
        "target",
        "mode",
        "parent_job_id",
        "judge",
        "input_json",
    ],
    "texts": ["session_id", "turn_idx", "text_idx", "role", "text"],
    "jobs": [
        "job_id",
        "session_id",
        "job_type",
        "status",
        "parent_job_id",
        "mlflow_run_id",
        "user_id",
        "ts",
    ],
    "row_scores": [
        "session_id",
        "row_id",
        "use_case",
        "model",
        "aspect",
        "role",
        "adjudicate",
        "status",
        "reason",
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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    session_id TEXT, use_case TEXT, model TEXT, outcome TEXT,
    turns INTEGER, tokens INTEGER, cost REAL, wall_clock_s REAL, avg_latency_s REAL,
    PRIMARY KEY (session_id, use_case)
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
    job_id TEXT PRIMARY KEY, session_id TEXT, job_type TEXT, status TEXT,
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

_EXPORT_TABLES = tuple(_COLUMNS)
# Raw (use-case-independent) tables: replaced wholesale per session on re-ingest.
_RAW_TABLES = ("turns", "tool_calls", "texts", "jobs")


def row_role(row: dict[str, Any]) -> str | None:
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


def _insert(conn: sqlite3.Connection, table: str, values: dict[str, Any]) -> None:
    cols = _COLUMNS[table]
    placeholders = ",".join("?" * len(cols))
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
        tuple(values.get(c) for c in cols),
    )


def _delete_session(conn: sqlite3.Connection, session_id: str, use_case: str) -> None:
    # Scored tables are per (session, use_case): replacing a `general` ingest must
    # not drop the same session's `rfe_assess` rows.
    conn.execute("DELETE FROM runs WHERE session_id = ? AND use_case = ?", (session_id, use_case))
    conn.execute(
        "DELETE FROM row_scores WHERE session_id = ? AND use_case = ?", (session_id, use_case)
    )
    # Raw tables are use-case-independent — replace wholesale for the session.
    for table in _RAW_TABLES:
        conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))


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
            _delete_session(conn, run.session_id, use_case)
            model = run.agent_model
            _insert(
                conn,
                "runs",
                {
                    "session_id": run.session_id,
                    "use_case": use_case,
                    "model": model,
                    "outcome": run.outcome,
                    "turns": run.turns_to_complete,
                    "tokens": run.total_tokens,
                    "cost": run.total_cost,
                    "wall_clock_s": run.wall_clock_s,
                    "avg_latency_s": run.avg_latency_s,
                },
            )
            for ti, turn in enumerate(run.counted_turns):
                tk = turn.get("tokens") or {}
                _insert(
                    conn,
                    "turns",
                    {
                        "session_id": run.session_id,
                        "turn_idx": ti,
                        "role": turn.get("role"),
                        "model": turn.get("model"),
                        "provider": turn.get("provider"),
                        "input_tok": int(tk.get("input", 0)),
                        "output_tok": int(tk.get("output", 0)),
                        "reasoning_tok": int(tk.get("reasoning", 0)),
                        "cache_tok": int(tk.get("cache", 0)),
                        "cost": float(turn.get("cost") or 0),
                        "duration_ms": turn.get("duration_ms"),
                    },
                )
                for ci, call in enumerate(turn.get("tool_calls") or []):
                    inp = call.get("input")
                    _insert(
                        conn,
                        "tool_calls",
                        {
                            "session_id": run.session_id,
                            "turn_idx": ti,
                            "call_idx": ci,
                            "role": call.get("role"),
                            "tool": call.get("tool"),
                            "status": call.get("status"),
                            "target": call.get("target"),
                            "mode": call.get("mode"),
                            "parent_job_id": call.get("parent_job_id"),
                            "judge": call.get("judge"),
                            "input_json": json.dumps(inp) if inp is not None else None,
                        },
                    )
                for xi, txt in enumerate(turn.get("texts") or []):
                    _insert(
                        conn,
                        "texts",
                        {
                            "session_id": run.session_id,
                            "turn_idx": ti,
                            "text_idx": xi,
                            "role": txt.get("role"),
                            "text": txt.get("text"),
                        },
                    )
            for row_id, status in scores.get(run.session_id, {}).items():
                row = rows_by_id.get(row_id, {})
                _insert(
                    conn,
                    "row_scores",
                    {
                        "session_id": run.session_id,
                        "row_id": row_id,
                        "use_case": use_case,
                        "model": model,
                        "aspect": row.get("aspect"),
                        "role": row_role(row),
                        "adjudicate": row.get("adjudicate"),
                        "status": status,
                        "reason": reasons.get((run.session_id, row_id), ""),
                    },
                )
            for job in run.jobs:
                _insert(
                    conn,
                    "jobs",
                    {
                        "job_id": job.get("job_id"),
                        "session_id": run.session_id,
                        "job_type": job.get("job_type"),
                        "status": job.get("status"),
                        "parent_job_id": job.get("parent_job_id"),
                        "mlflow_run_id": job.get("mlflow_run_id"),
                        "user_id": job.get("user_id"),
                        "ts": job.get("ts"),
                    },
                )
        conn.commit()
    finally:
        conn.close()


def ingest_rubrics(db_path: Path, use_case: str, results: list[dict[str, Any]]) -> None:
    """Write (or replace) graded rubric step rows. `results` is score_all()'s output."""
    conn = connect(db_path)
    try:
        for sid in {r["session_id"] for r in results}:
            conn.execute(
                "DELETE FROM rubric_scores WHERE session_id = ? AND use_case = ?", (sid, use_case)
            )
        for rb in results:
            for step in rb["steps"]:
                _insert(
                    conn,
                    "rubric_scores",
                    {
                        "session_id": rb["session_id"],
                        "use_case": use_case,
                        "role": rb["role"],
                        "skill": rb["skill"],
                        "step_id": step["step_id"],
                        "status": step["status"],
                        "weight": step["weight"],
                        "reason": step.get("reason", ""),
                        "adherence_pct": rb["adherence_pct"],
                    },
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
            _write_csv(table, out / f"{table}.csv", rows)
        elif fmt == "parquet":
            _write_parquet(out / f"{table}.parquet", rows)
        else:
            raise ValueError(f"unknown export format: {fmt!r}")
    print(f"Exported {len(_EXPORT_TABLES)} tables to {out} ({fmt})")


def _write_csv(table: str, path: Path, rows: list[dict[str, Any]]) -> None:
    # Always emit a header (from the schema columns) even for an empty table, so
    # header-keyed loaders don't choke.
    cols = list(rows[0].keys()) if rows else _COLUMNS[table]
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
