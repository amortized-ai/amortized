"""The optional ``--llm-judge`` pass.

Scores ``adjudicate: llm_judge`` rows with one LLM call per run, over a
transcript built from the logged assistant prose + tool outputs. Verdicts are
normalised to the scorer's status vocabulary; ``unknown``/unparseable stays
``review`` for a human.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from monitor.scorer.model import Run
from monitor.scorer.requires import _requires_met

# Default judge model = what Morty itself runs on.
JUDGE_MODEL_DEFAULT = "claude-opus-4-8"
# Per-tool-output cap in the judge PROMPT only (the log keeps full output). Large
# enough for eval-results/config payloads; bounds pathological dumps.
_JUDGE_OUTPUT_CAP = 8000

_JUDGE_SYSTEM = (
    'You are a strict adjudicator for an ML-agent ("Morty") pipeline monitor. '
    "You are given ONE run's transcript — the agent's assistant messages and its "
    "tool calls with outputs — and a rubric of behavioral checklist rows. For each "
    "row decide whether its expectation held, judging ONLY from the transcript.\n"
    'Verdicts: "met" (clearly satisfied), "wrong" (clearly violated), '
    '"n/a" (the situation the row targets never arose), '
    '"unknown" (the transcript lacks the evidence to decide — e.g. the assistant '
    'messages were not captured). Prefer "unknown" over guessing. For grounding '
    "rows, a claim is grounded only if the cited ids/numbers actually appear in a "
    "tool output.\n"
    "For rows about a specific workflow step: if the run ended before that step "
    "could occur (it was abandoned or errored out earlier in the pipeline), answer "
    '"n/a" — do NOT penalise a step the run never reached. Reserve "wrong" for a '
    "step the run actually reached but did incorrectly, or clearly skipped despite "
    "progressing past the point where it should have happened.\n"
    "Respond with ONLY a JSON object mapping each row_id to "
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
    for run in runs:
        # Only judge rows whose precondition holds for this run; gated-out rows
        # keep the `n/a` score_run already assigned (skill not loaded here).
        active = [r for r in rows if _requires_met(run, r.get("requires"))]
        if not active:
            continue
        rubric = json.dumps(
            [{"row_id": r["id"], "expected": r["expected"]} for r in active], indent=2
        )
        prompt = f"RUBRIC (judge each row_id):\n{rubric}\n\nTRANSCRIPT:\n{_judge_evidence(run)}"
        try:
            data = _parse_judge_json(call(_JUDGE_SYSTEM, prompt))
        except Exception as exc:
            print(f"  ! llm-judge failed for {run.session_id}: {exc}", file=sys.stderr)
            data = {}
        for r in active:
            entry = data.get(r["id"])
            # The judge is asked for {"verdict","reason"} per row, but sometimes
            # returns a bare verdict string ({"row_id": "met"}); tolerate both so
            # one malformed entry doesn't crash the whole scoring pass.
            if isinstance(entry, dict):
                verdict, reason = entry.get("verdict"), entry.get("reason", "")
            elif isinstance(entry, str):
                verdict, reason = entry, ""
            else:
                verdict, reason = None, ""
            verdicts[(run.session_id, r["id"])] = _norm_verdict(verdict)
            reasons[(run.session_id, r["id"])] = str(reason).strip()
    return verdicts, reasons
