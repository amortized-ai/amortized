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
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

# Default judge model = what Morty itself runs on.
JUDGE_MODEL_DEFAULT = "claude-opus-4-8"
# Per-tool-output cap in the judge PROMPT only (the log keeps full output). Large
# enough for eval-results/config payloads; bounds pathological dumps.
_JUDGE_OUTPUT_CAP = 8000


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
        # Headline efficiency EXCLUDES cache-read tokens (cheap, dominated by the
        # cached system prompt): input + output + reasoning only. The raw log
        # keeps the full breakdown incl. cache under `tokens`.
        total = 0
        for t in self.counted_turns:
            tk = t.get("tokens") or {}
            total += (
                int(tk.get("input", 0)) + int(tk.get("output", 0)) + int(tk.get("reasoning", 0))
            )
        return total

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
    def _latencies_s(self) -> list[float]:
        # Per-message response latency: send received -> response ready
        # (turn.duration_ms). This is the assistant response time per user message.
        out: list[float] = []
        for t in self.counted_turns:
            ms = t.get("duration_ms")
            if isinstance(ms, (int, float)):
                out.append(float(ms) / 1000.0)
        return out

    @property
    def avg_latency_s(self) -> float | None:
        lat = self._latencies_s
        return sum(lat) / len(lat) if lat else None

    @property
    def max_latency_s(self) -> float | None:
        lat = self._latencies_s
        return max(lat) if lat else None

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
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # A use-case may `extends: general` — a task overlay on top of the generic
    # workflow checklist. Merge: base rows in order (overlay overrides by id),
    # then overlay-only rows appended.
    parent = data.get("extends")
    if parent:
        base_rows = load_checklist(parent).get("rows", [])
        overlay_rows = data.get("rows", [])
        overlay_by_id = {r["id"]: r for r in overlay_rows}
        base_ids = {r["id"] for r in base_rows}
        merged = [overlay_by_id.get(r["id"], r) for r in base_rows]
        merged += [r for r in overlay_rows if r["id"] not in base_ids]
        data["rows"] = merged
    return data


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


def _spec_match(call: dict[str, Any], spec: Any) -> bool:
    """A tool_before endpoint is either a bare tool name (str) or a
    ``{tool, where}`` dict that also matches on logged arg fields (e.g. mode)."""
    if isinstance(spec, str):
        return call.get("tool") == spec
    if call.get("tool") != spec.get("tool"):
        return False
    return all(call.get(k) == v for k, v in (spec.get("where") or {}).items())


def _spec_field_missing(run: Run, spec: Any) -> bool:
    """logging-dependency guard: True when a `where` field is required but no
    call of that tool carries it (log predates the signal -> defer to review).
    False when the tool was never called (let the normal missed/wrong path run)."""
    if isinstance(spec, str) or not spec.get("where"):
        return False
    keys = spec["where"].keys()
    calls = _calls(run, spec.get("tool"))
    return bool(calls) and not any(any(k in c for k in keys) for c in calls)


def score_auto(run: Run, match: dict[str, Any]) -> str:
    mtype = match.get("type")

    if mtype == "tool_called":
        where = match.get("where") or {}
        tools = match.get("tools") or [match["tool"]]  # `tools` = any-of tool names
        needle = match.get("output_contains")  # optional substring test on tool output
        for c in run.tool_calls:
            if c.get("tool") not in tools:
                continue
            if not all(c.get(k) == v for k, v in where.items()):
                continue
            if match.get("status") and c.get("status") != match["status"]:
                continue
            if needle and needle not in (c.get("output") or ""):
                continue
            return "met"
        return "missed"

    if mtype == "chained":
        # A validate_* call carries a non-empty field from `any_of` — used for
        # pipeline wiring (parent_job_id / data_run_id / eval_data_run_id) and
        # for other required config refs (e.g. the eval `judge`). Absent -> missed.
        keys = match.get("any_of") or ["parent_job_id", "data_run_id"]
        for c in _calls(run, match["tool"]):
            if any(c.get(k) for k in keys):
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
        before_spec, after_spec = match["before"], match["after"]
        if _spec_field_missing(run, before_spec) or _spec_field_missing(run, after_spec):
            return "review"  # log predates the arg the `where` filters on
        before = [i for i, c in enumerate(run.tool_calls) if _spec_match(c, before_spec)]
        after = [i for i, c in enumerate(run.tool_calls) if _spec_match(c, after_spec)]
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

    if mtype == "scores_present":
        # Outcome (not mechanic): the eval job returned usable rubric scores.
        # get_eval_results can succeed as a call yet carry all-null `scores`
        # (judge scored nothing) -> the eval produced no signal. missed when the
        # results were never fetched; wrong when fetched but every score is null.
        calls = _calls(run, match.get("tool", "get_eval_results"))
        if not calls:
            return "missed"
        for c in calls:
            # output is truncated, so match the `"scores": { ... }` block
            # non-greedily up to its closing brace or the truncation edge.
            m = re.search(r'"scores"\s*:\s*\{(.*?)(?:\}|$)', c.get("output") or "", re.S)
            if m and re.search(r":\s*-?\d+(?:\.\d+)?", m.group(1)):
                return "met"  # at least one criterion has a real numeric score
        return "wrong"

    if mtype == "solo_delegation":
        # Every delegating message was ONLY the delegate call (no text, no other
        # tool). `solo` is captured per delegate call; absent -> log predates the
        # signal, defer to review (logging-dependency rule).
        calls = _calls(run, "delegate_to_subagent")
        target = match.get("target")
        if target:
            calls = [c for c in calls if c.get("target") == target]
        if not calls:
            return "missed"
        if not any("solo" in c for c in calls):
            return "review"
        return "met" if all(c.get("solo") for c in calls) else "wrong"

    if mtype == "no_error_calls":
        # Robustness: no tool call ended in an error status (optionally limited
        # to `tools`). Distinct from `recovery`, which only asks whether a later
        # call succeeded after an error — this penalises the error itself.
        tools = match.get("tools")
        for c in run.tool_calls:
            if tools and c.get("tool") not in tools:
                continue
            if c.get("status") == "error":
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
# LLM-judge pass (fills `llm_judge` rows from the log; default model = Morty's)
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM = (
    'You are a strict adjudicator for an ML-agent ("Morty") pipeline monitor. '
    "You are given ONE run's transcript — the agent's assistant messages and its "
    "tool calls with outputs — and a rubric of behavioral checklist rows. For each "
    "row decide whether its expectation held, judging ONLY from the transcript.\n"
    'Verdicts: "met" (clearly satisfied), "wrong" (clearly violated), '
    '"n/a" (the situation the row targets never arose), '
    '"unknown" (the transcript lacks the evidence to decide — e.g. the assistant '
    "messages were not captured). Prefer \"unknown\" over guessing. For grounding "
    "rows, a claim is grounded only if the cited ids/numbers actually appear in a "
    "tool output. Respond with ONLY a JSON object mapping each row_id to "
    '{"verdict": <one of the four>, "reason": "<=200 chars"}. No prose outside JSON.'
)


def _judge_evidence(run: Run) -> str:
    """Chronological transcript for the judge: assistant prose + tool calls with
    (prompt-capped) outputs, in turn order."""
    lines: list[str] = []
    for turn in run.counted_turns:
        trole = turn.get("role", "?")
        for t in turn.get("texts") or []:
            lines.append(f"[{t.get('role', trole)} says] {t.get('text', '')}")
        for c in turn.get("tool_calls") or []:
            out = c.get("output") or ""
            if len(out) > _JUDGE_OUTPUT_CAP:
                out = out[:_JUDGE_OUTPUT_CAP] + f"…[+{len(out) - _JUDGE_OUTPUT_CAP} chars]"
            meta = {k: c.get(k) for k in ("status", "target", "mode") if c.get(k)}
            lines.append(f"[{c.get('role', trole)} tool] {c.get('tool')} {meta} -> {out}")
    return "\n".join(lines) or "(no assistant text or tool calls were captured)"


def _norm_verdict(value: Any) -> str:
    v = str(value or "").strip().lower().replace("_", "/")
    if v in {"met", "wrong"}:
        return v
    if v in {"n/a", "na"}:
        return "n/a"
    return "review"  # "unknown" or unparseable -> stays for a human


def _parse_judge_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.lstrip("`")
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _make_judge_call(model: str):
    """Return a `call(system, prompt) -> str`. Uses Vertex when
    ANTHROPIC_VERTEX_PROJECT_ID is set (matches this env), else the direct
    Anthropic API (ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL)."""
    try:
        import anthropic
    except ImportError:
        sys.exit("--llm-judge needs the `anthropic` package: uv pip install anthropic")
    import os

    project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    if project:
        region = os.environ.get("CLOUD_ML_REGION") or os.environ.get(
            "ANTHROPIC_VERTEX_REGION", "global"
        )
        client: Any = anthropic.AnthropicVertex(project_id=project, region=region)
    else:
        client = anthropic.Anthropic()

    def call(system: str, prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    return call


def llm_judge_runs(
    runs: list[Run], checklist: dict[str, Any], model: str
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Judge every `llm_judge` row for every run in one call per run. Returns
    (verdicts, reasons) keyed by (session_id, row_id)."""
    rows = [r for r in checklist["rows"] if r.get("adjudicate") == "llm_judge"]
    verdicts: dict[tuple[str, str], str] = {}
    reasons: dict[tuple[str, str], str] = {}
    if not rows:
        return verdicts, reasons
    call = _make_judge_call(model)
    rubric = json.dumps([{"row_id": r["id"], "expected": r["expected"]} for r in rows], indent=2)
    for run in runs:
        prompt = f"RUBRIC (judge each row_id):\n{rubric}\n\nTRANSCRIPT:\n{_judge_evidence(run)}"
        try:
            data = _parse_judge_json(call(_JUDGE_SYSTEM, prompt))
        except Exception as exc:  # noqa: BLE001 - report and leave rows as review
            print(f"  ! llm-judge failed for {run.session_id}: {exc}", file=sys.stderr)
            data = {}
        for r in rows:
            entry = data.get(r["id"]) or {}
            verdicts[(run.session_id, r["id"])] = _norm_verdict(entry.get("verdict"))
            reasons[(run.session_id, r["id"])] = str(entry.get("reason", "")).strip()
    return verdicts, reasons


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
            f"tokens excl. cache: **{_fmt(run.total_tokens, 'int')}**   "
            f"cost: {_fmt(run.total_cost, '$')}   "
            f"wall-clock: {_fmt(run.wall_clock_s, 's')}"
        )
        print(
            f"- response latency: avg {_fmt(run.avg_latency_s, 's')}   "
            f"max {_fmt(run.max_latency_s, 's')}   (send received -> response ready, per message)"
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
    header = [
        "model",
        "runs",
        "avg turns",
        "avg tokens (excl cache)",
        "avg cost",
        "avg time",
        "avg latency",
    ] + [f"asp {a}" for a in aspects]
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
        lats = [r.avg_latency_s for r in model_runs if r.avg_latency_s is not None]
        avg_latency = sum(lats) / len(lats) if lats else None

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
            _fmt(avg_latency, "s"),
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


def print_judge_rationale(
    runs: list[Run],
    checklist: dict[str, Any],
    scores: dict[str, dict[str, str]],
    reasons: dict[tuple[str, str], str],
    model: str,
) -> None:
    rows = [r for r in checklist["rows"] if r.get("adjudicate") == "llm_judge"]
    print(f"\n## LLM-judge rationale  (model: `{model}`)")
    for run in runs:
        print(f"\n### {run.session_id}")
        for r in rows:
            status = scores[run.session_id].get(r["id"], "review")
            reason = reasons.get((run.session_id, r["id"]), "")
            print(f"- [{status:>6}] {r['id']} — {reason or '(no reason)'}")


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

    print_per_run(runs, scores, checklist)
    if judge_reasons:
        print_judge_rationale(runs, checklist, scores, judge_reasons, args.judge_model)
    print_comparison(runs, scores, checklist)


if __name__ == "__main__":
    main()
