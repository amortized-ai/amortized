#!/usr/bin/env python3
"""Process monitor JSONL logs into testing metrics.

Reads per-session logs written by the amortized proxy (`<session>.jsonl`,
default `~/.amortized/monitor/`), scores each run against a use-case
expected-aspect checklist, and prints:

  (a) per-run efficiency + checklist status, and
  (b) a per-model comparison table (the LLM comparison).

Efficiency (turns-to-complete, tokens excl. cache, cost, wall-clock) is measured up to
the human-declared completion turn. Objective checklist rows are scored
autonomously from the log. `llm_judge` rows can be scored by --llm-judge (an LLM
pass over the transcript); otherwise `llm_judge` / `human` rows are emitted as
`review` and can be filled via `--review <csv>`.

Usage:
  process_monitor_logs.py <log-dir> --use-case general
  process_monitor_logs.py <log-dir> --use-case general --emit-review review.csv
  process_monitor_logs.py <log-dir> --use-case general --review review.filled.csv
  process_monitor_logs.py <log-dir> --use-case general --llm-judge

--llm-judge uses claude-opus-4-8 (what Morty runs on; override with --judge-model)
and reads creds from the env: ANTHROPIC_VERTEX_PROJECT_ID (+ CLOUD_ML_REGION) for
Vertex, else ANTHROPIC_API_KEY for the direct API. A --review CSV overrides it.

Status values: met | wrong | missed | n/a | review

The scoring logic lives in the importable `monitor.scorer` package; this file is
only the CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so `monitor.scorer` resolves when this script is
# run directly (e.g. `uv run python monitor/scripts/process_monitor_logs.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from monitor.scorer.judge import JUDGE_MODEL_DEFAULT, llm_judge_runs  # noqa: E402
from monitor.scorer.matchers import score_run  # noqa: E402
from monitor.scorer.model import load_checklist, load_runs  # noqa: E402
from monitor.scorer.report import (  # noqa: E402
    emit_review_csv,
    load_review_csv,
    print_comparison,
    print_per_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="Directory of <session>.jsonl monitor logs")
    parser.add_argument("--use-case", required=True, help="Checklist under monitor/use_cases/")
    parser.add_argument("--emit-review", type=Path, help="Write a blank review CSV and exit")
    parser.add_argument(
        "--review", type=Path, help="Merge a filled review CSV of human/LLM verdicts"
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Score `llm_judge` rows with an LLM pass (one call per run)",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL_DEFAULT,
        help=f"Model for --llm-judge (default: {JUDGE_MODEL_DEFAULT}, what Morty runs on)",
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

    judge_reasons: dict[tuple[str, str], str] = {}
    if args.llm_judge:
        verdicts, judge_reasons = llm_judge_runs(runs, checklist, args.judge_model)
        for (sid, row_id), status in verdicts.items():
            scores[sid][row_id] = status

    # A filled review CSV wins over the LLM (human overrides the machine).
    if args.review:
        overrides = load_review_csv(args.review)
        for run in runs:
            for row in checklist["rows"]:
                key = (run.session_id, row["id"])
                if key in overrides:
                    scores[run.session_id][row["id"]] = overrides[key]

    print_per_run(runs, scores, checklist, judge_reasons)
    print_comparison(runs, scores, checklist)


if __name__ == "__main__":
    main()
