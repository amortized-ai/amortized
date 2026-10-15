"""The ``Run`` aggregate and log/checklist loading.

A ``Run`` is one session's turns plus its completion record; its properties are
the efficiency metrics and the flattened, chronological ``tool_calls`` every
matcher scans. Loading parses ``<session>.jsonl`` files and merges ``extends``
checklist overlays.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")

CHECKLIST_DIR = Path(__file__).resolve().parent.parent / "use_cases"


@dataclass
class Run:
    session_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    completion: dict[str, Any] | None = None
    jobs: list[dict[str, Any]] = field(default_factory=list)

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

    @property
    def models(self) -> list[str]:
        """The per-turn orchestrator model series, chronological (drops blanks).
        `agent_model` is the last of these; this keeps the whole sequence so a
        mid-conversation model switch is observable."""
        return [
            t["model"]
            for t in self.counted_turns
            if t.get("role") == "orchestrator" and t.get("model")
        ]

    @property
    def model_switches(self) -> int:
        """How many times the orchestrator model changed between consecutive turns."""
        seq = self.models
        return sum(1 for a, b in pairwise(seq) if a != b)

    @property
    def model_switch_index(self) -> int | None:
        """Index into `counted_turns` of the first turn whose orchestrator model
        differs from the previous one (the switch boundary), or None if no switch."""
        prev: str | None = None
        for i, t in enumerate(self.counted_turns):
            if t.get("role") != "orchestrator":
                continue
            m = t.get("model")
            if not m:
                continue
            if prev is not None and m != prev:
                return i
            prev = m
        return None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _calls(run: Run, tool: str) -> list[dict[str, Any]]:
    """Every tool call of a given name in the run (chronological)."""
    return [c for c in run.tool_calls if c.get("tool") == tool]


def load_runs(log_dir: Path) -> list[Run]:
    import json

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
            elif kind == "job_created":
                run.jobs.append(rec)
        run.turns.sort(key=lambda t: str(t.get("ts", "")))
        runs.append(run)
    return runs


def load_checklist(use_case: str, _seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    if use_case in _seen:
        sys.exit(f"checklist `extends` cycle involving {use_case!r}")
    path = CHECKLIST_DIR / use_case / "checklist.yaml"
    if not path.exists():
        sys.exit(f"No checklist for use-case {use_case!r} at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # A use-case may `extends: general` — a task overlay on top of the generic
    # workflow checklist. Merge: base rows in order (overlay overrides by id),
    # then overlay-only rows appended.
    parent = data.get("extends")
    if parent:
        base_rows = load_checklist(parent, _seen | {use_case}).get("rows", [])
        overlay_rows = data.get("rows", [])
        overlay_by_id = {r["id"]: r for r in overlay_rows}
        base_ids = {r["id"] for r in base_rows}
        merged = [overlay_by_id.get(r["id"], r) for r in base_rows]
        merged += [r for r in overlay_rows if r["id"] not in base_ids]
        data["rows"] = merged
    return data
