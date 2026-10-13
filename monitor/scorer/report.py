"""Human-readable output (per-run block + comparison table) and review CSV I/O."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from monitor.scorer.model import Run


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "s":
        return f"{value:.0f}s"
    if kind == "$":
        return f"${value:.4f}"
    if kind == "int":
        return f"{int(value):,}"
    return str(value)


def emit_review_csv(path: Path, runs: list[Run], checklist: dict[str, Any]) -> None:
    review_rows = [r for r in checklist["rows"] if r.get("adjudicate") in ("human", "llm_judge")]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["session_id", "row_id", "aspect", "expected", "status"])
        for run in runs:
            for row in review_rows:
                writer.writerow([run.session_id, row["id"], row["aspect"], row["expected"], ""])
    print(f"Wrote review template: {path}  ({len(runs)} runs x {len(review_rows)} rows)")


def load_review_csv(path: Path) -> dict[tuple[str, str], str]:
    overrides: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            status = (row.get("status") or "").strip()
            if status:
                overrides[(row["session_id"], row["row_id"])] = status
    return overrides


def print_per_run(
    runs: list[Run],
    scores: dict[str, dict[str, str]],
    checklist: dict[str, Any],
    reasons: dict[tuple[str, str], str] | None = None,
) -> None:
    reasons = reasons or {}
    print("## Per-run\n")
    for run in runs:
        sc = scores[run.session_id]
        print(f"### {run.session_id}")
        print(f"- model: `{run.agent_model or '-'}`   outcome: {run.outcome or '(not marked)'}")
        print(
            f"- turns-to-complete: **{run.turns_to_complete}**   "
            f"tokens excl. cache: **{_fmt(run.total_tokens, 'int')}**   "
            f"cost: {_fmt(run.total_cost, '$')}   "
            f"wall-clock: {_fmt(run.wall_clock_s, 's')}"
        )
        print(
            f"- response latency: avg {_fmt(run.avg_latency_s, 's')}   "
            f"max {_fmt(run.max_latency_s, 's')}   (send received -> response ready, per message)"
        )
        for row in checklist["rows"]:
            reason = reasons.get((run.session_id, row["id"]), "")
            suffix = f"  |  {reason}" if reason else ""
            print(
                f"    - [{sc[row['id']]:>6}] {row['aspect']} · {row['id']} — {row['stage']}{suffix}"
            )
        print()


def print_rubrics(results: list[dict[str, Any]]) -> None:
    """Per-run graded skill-adherence: % + each step's status, skipped steps listed."""
    if not results:
        return
    print("\n## Skill adherence (graded rubric)\n")
    by_session: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_session.setdefault(r["session_id"], []).append(r)
    for session_id, rubrics in by_session.items():
        print(f"### {session_id}")
        for rb in rubrics:
            pct = rb["adherence_pct"]
            pct_str = f"{pct:.0f}%" if pct is not None else "n/a"
            print(f"- **{rb['skill']}** ({rb['role']}): adherence **{pct_str}**")
            for step in rb["steps"]:
                reason = f"  |  {step['reason']}" if step.get("reason") else ""
                print(
                    f"    - [{step['status']:>6}] {step['step_id']} (w{step['weight']:g}){reason}"
                )
            if rb["skipped"]:
                print(f"    skipped: {', '.join(rb['skipped'])}")
        print()


def print_comparison(
    runs: list[Run], scores: dict[str, dict[str, str]], checklist: dict[str, Any]
) -> None:
    aspects = sorted({row["aspect"] for row in checklist["rows"]})
    header = [
        "model",
        "session",
        "outcome",
        "turns",
        "tokens (excl cache)",
        "cost",
        "time",
        "latency",
    ] + [f"asp {a}" for a in aspects]
    print("## Per-run comparison\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")

    # One row per run (no averaging), grouped by model for readability.
    for run in sorted(runs, key=lambda r: (r.agent_model or "(unknown)", r.session_id)):
        sc = scores[run.session_id]
        aspect_cells: list[str] = []
        for aspect in aspects:
            met = total = 0
            pending = False  # any `review` row (judge not run / failed / deferred)
            for row in checklist["rows"]:
                if row["aspect"] != aspect:
                    continue
                status = sc[row["id"]]
                if status in ("met", "wrong", "missed"):
                    total += 1
                    met += status == "met"
                elif status == "review":
                    pending = True
            cell = f"{met}/{total}" if total else "-"
            if pending:
                cell += "*"  # fraction omits unadjudicated rows
            aspect_cells.append(cell)

        row_cells = [
            run.agent_model or "(unknown)",
            run.session_id[:8],
            run.outcome or "(not marked)",
            str(run.turns_to_complete),
            _fmt(run.total_tokens, "int"),
            _fmt(run.total_cost, "$"),
            _fmt(run.wall_clock_s, "s"),
            _fmt(run.avg_latency_s, "s"),
            *aspect_cells,
        ]
        print("| " + " | ".join(row_cells) + " |")
    print()
    print(
        "_One row per run. Aspect cells = met / (met+wrong+missed); "
        "`review` rows excluded until adjudicated. `*` = the aspect has "
        "unadjudicated (`review`) rows, so its fraction is auto-only and not "
        "comparable to a fully-scored run._"
    )
