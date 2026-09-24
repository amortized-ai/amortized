#!/usr/bin/env python3
"""Process monitor JSONL logs into vibe-testing metrics.

Reads per-session logs written by the amortized proxy (`<session>.jsonl`,
default `~/.amortized/monitor/`), scores each run against a use-case
expected-aspect checklist, and prints:

  (a) per-run efficiency + checklist status, and
  (b) a per-model comparison table (the LLM comparison).

Efficiency (turns-to-complete, total tokens, cost, wall-clock) is measured up to
the human-declared completion turn. Objective checklist rows are scored
autonomously from the log; `human` / `llm_judge` rows are emitted as `review` and
can be filled via `--review <csv>`.

Usage:
  process_monitor_logs.py <log-dir> --use-case rfe_assess
  process_monitor_logs.py <log-dir> --use-case rfe_assess --emit-review review.csv
  process_monitor_logs.py <log-dir> --use-case rfe_assess --review review.filled.csv

Status values: met | wrong | missed | n/a | review
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")

CHECKLIST_DIR = Path(__file__).resolve().parent.parent / "use_cases"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    session_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    completion: dict[str, Any] | None = None

    @property
    def counted_turns(self) -> list[dict[str, Any]]:
        """Turns up to (and including) the completion boundary."""
        if not self.completion:
            return self.turns
        cutoff = self.completion.get("ts", "")
        return [t for t in self.turns if str(t.get("ts", "")) <= cutoff] or self.turns

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for turn in self.counted_turns:
            calls.extend(turn.get("tool_calls") or [])
        return calls

    @property
    def total_tokens(self) -> int:
        return sum(int((t.get("tokens") or {}).get("total", 0)) for t in self.counted_turns)

    @property
    def total_cost(self) -> float:
        return sum(float(t.get("cost") or 0) for t in self.counted_turns)

    @property
    def turns_to_complete(self) -> int:
        return len(self.counted_turns)

    @property
    def wall_clock_s(self) -> float | None:
        stamps = [
            (t.get("started_at"), t.get("finished_at"))
            for t in self.counted_turns
            if t.get("started_at") and t.get("finished_at")
        ]
        if not stamps:
            return None
        starts = [_parse_dt(s) for s, _ in stamps if _parse_dt(s)]
        ends = [_parse_dt(e) for _, e in stamps if _parse_dt(e)]
        if not starts or not ends:
            return None
        return (max(ends) - min(starts)).total_seconds()

    @property
    def agent_model(self) -> str | None:
        models = [t.get("model") for t in self.counted_turns if t.get("role") == "orchestrator"]
        models = [m for m in models if m]
        if not models:
            models = [t.get("model") for t in self.counted_turns if t.get("model")]
        return models[-1] if models else None

    @property
    def outcome(self) -> str | None:
        return self.completion.get("outcome") if self.completion else None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def load_runs(log_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        run = Run(session_id=path.stem)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("kind")
            if kind == "turn":
                run.turns.append(rec)
            elif kind == "completion":
                run.completion = rec  # last one wins
        run.turns.sort(key=lambda t: str(t.get("ts", "")))
        runs.append(run)
    return runs


def load_checklist(use_case: str) -> dict[str, Any]:
    path = CHECKLIST_DIR / use_case / "checklist.yaml"
    if not path.exists():
        sys.exit(f"No checklist for use-case {use_case!r} at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Matchers (autonomous scoring)
# --------------------------------------------------------------------------- #

_ERROR_MARKERS = ('"errors"', "validation_error", "is required", "invalid", "must be")


def _is_ok_output(status: str | None, output: str | None) -> bool:
    if status == "error":
        return False
    text = (output or "").lower()
    return not any(marker in text for marker in _ERROR_MARKERS)


def _calls(run: Run, tool: str) -> list[dict[str, Any]]:
    return [c for c in run.tool_calls if c.get("tool") == tool]


def score_auto(run: Run, match: dict[str, Any]) -> str:
    mtype = match.get("type")

    if mtype == "tool_called":
        where = match.get("where") or {}
        for c in _calls(run, match["tool"]):
            if all(c.get(k) == v for k, v in where.items()):
                if match.get("status") and c.get("status") != match["status"]:
                    continue
                return "met"
        return "missed"

    if mtype == "validate_ok":
        calls = _calls(run, match["tool"])
        if not calls:
            return "missed"
        return (
            "met"
            if any(_is_ok_output(c.get("status"), c.get("output")) for c in calls)
            else "wrong"
        )

    if mtype == "tool_before":
        before = [i for i, c in enumerate(run.tool_calls) if c.get("tool") == match["before"]]
        after = [i for i, c in enumerate(run.tool_calls) if c.get("tool") == match["after"]]
        if not after:
            return "missed"
        if not before or min(before) > min(after):
            return "wrong"
        return "met"

    if mtype == "recovery":
        calls = run.tool_calls
        error_idx = next((i for i, c in enumerate(calls) if c.get("status") == "error"), None)
        if error_idx is None:
            return "n/a"  # nothing to recover from
        recovered = any(c.get("status") == "completed" for c in calls[error_idx + 1 :])
        return "met" if recovered else "wrong"

    if mtype == "completion":
        if not run.completion:
            return "missed"
        want = match.get("outcome")
        if want and run.outcome != want:
            return "wrong"
        return "met"

    return "review"


def score_run(run: Run, checklist: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in checklist["rows"]:
        mode = row.get("adjudicate", "human")
        if mode == "auto" and row.get("match"):
            result[row["id"]] = score_auto(run, row["match"])
        else:
            result[row["id"]] = "review"  # human / llm_judge -> offline
    return result


# --------------------------------------------------------------------------- #
# Review CSV merge
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


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


def print_per_run(
    runs: list[Run], scores: dict[str, dict[str, str]], checklist: dict[str, Any]
) -> None:
    print("## Per-run\n")
    for run in runs:
        sc = scores[run.session_id]
        print(f"### {run.session_id}")
        print(f"- model: `{run.agent_model or '-'}`   outcome: {run.outcome or '(not marked)'}")
        print(
            f"- turns-to-complete: **{run.turns_to_complete}**   "
            f"total tokens: **{_fmt(run.total_tokens, 'int')}**   "
            f"cost: {_fmt(run.total_cost, '$')}   "
            f"wall-clock: {_fmt(run.wall_clock_s, 's')}"
        )
        for row in checklist["rows"]:
            print(f"    - [{sc[row['id']]:>6}] {row['aspect']} · {row['id']} — {row['stage']}")
        print()


def print_comparison(
    runs: list[Run], scores: dict[str, dict[str, str]], checklist: dict[str, Any]
) -> None:
    by_model: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_model[run.agent_model or "(unknown)"].append(run)

    aspects = sorted({row["aspect"] for row in checklist["rows"]})
    header = ["model", "runs", "avg turns", "avg tokens", "avg cost", "avg time"] + [
        f"asp {a}" for a in aspects
    ]
    print("## Model comparison\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")

    for model, model_runs in sorted(by_model.items()):
        n = len(model_runs)
        avg_turns = sum(r.turns_to_complete for r in model_runs) / n
        avg_tokens = sum(r.total_tokens for r in model_runs) / n
        avg_cost = sum(r.total_cost for r in model_runs) / n
        times = [r.wall_clock_s for r in model_runs if r.wall_clock_s is not None]
        avg_time = sum(times) / len(times) if times else None

        aspect_cells: list[str] = []
        for aspect in aspects:
            met = total = 0
            for run in model_runs:
                for row in checklist["rows"]:
                    if row["aspect"] != aspect:
                        continue
                    status = scores[run.session_id][row["id"]]
                    if status in ("met", "wrong", "missed"):
                        total += 1
                        met += status == "met"
            aspect_cells.append(f"{met}/{total}" if total else "-")

        row_cells = [
            model,
            str(n),
            f"{avg_turns:.1f}",
            _fmt(avg_tokens, "int"),
            _fmt(avg_cost, "$"),
            _fmt(avg_time, "s"),
            *aspect_cells,
        ]
        print("| " + " | ".join(row_cells) + " |")
    print()
    print(
        "_Aspect cells = met / (met+wrong+missed) across runs; "
        "`review` rows excluded until adjudicated._"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="Directory of <session>.jsonl monitor logs")
    parser.add_argument("--use-case", required=True, help="Checklist under monitor/use_cases/")
    parser.add_argument("--emit-review", type=Path, help="Write a blank review CSV and exit")
    parser.add_argument(
        "--review", type=Path, help="Merge a filled review CSV of human/LLM verdicts"
    )
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        sys.exit(f"Not a directory: {args.log_dir}")

    checklist = load_checklist(args.use_case)
    runs = load_runs(args.log_dir)
    if not runs:
        sys.exit(f"No *.jsonl logs found in {args.log_dir}")

    if args.emit_review:
        emit_review_csv(args.emit_review, runs, checklist)
        return

    scores = {run.session_id: score_run(run, checklist) for run in runs}

    if args.review:
        overrides = load_review_csv(args.review)
        for run in runs:
            for row in checklist["rows"]:
                key = (run.session_id, row["id"])
                if key in overrides:
                    scores[run.session_id][row["id"]] = overrides[key]

    print_per_run(runs, scores, checklist)
    print_comparison(runs, scores, checklist)


if __name__ == "__main__":
    main()
