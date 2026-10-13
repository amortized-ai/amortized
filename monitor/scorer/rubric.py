"""Graded per-step skill-adherence rubrics.

Each skill guide is decomposed into an ordered, weighted rubric
(``monitor/skills/<agent>/<skill>/rubric.yaml``). A rubric applies to a run only
when its guide was loaded (the ``gate``), so a classification run is never scored
against the task-distillation rubric. Steps are scored met/missed/wrong/n-a/review
— ``auto`` steps via the matcher registry, ``llm_judge`` steps via one judge call
per (run, skill) — and rolled up into a weighted adherence %:

    adherence% = Σ weight(met) / Σ weight(met + wrong + missed)   (n/a, review excluded)

Dropping a new ``rubric.yaml`` in the skills tree adds a graded skill with no code
change — the discovery here scans the tree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from monitor.scorer.judge import (
    _JUDGE_SYSTEM,
    _judge_evidence,
    _make_judge_call,
    _norm_verdict,
    _parse_judge_json,
)
from monitor.scorer.matchers import score_auto
from monitor.scorer.model import Run
from monitor.scorer.requires import _requires_met

RUBRICS_DIR = Path(__file__).resolve().parent.parent / "skills"


def load_rubrics(rubrics_dir: Path = RUBRICS_DIR) -> list[dict[str, Any]]:
    """Discover every rubric.yaml under the skills tree (sorted by skill id)."""
    rubrics: list[dict[str, Any]] = []
    for path in sorted(rubrics_dir.rglob("rubric.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("steps"):
            rubrics.append(data)
    rubrics.sort(key=lambda r: str(r.get("skill", "")))
    return rubrics


def applies(run: Run, rubric: dict[str, Any]) -> bool:
    return _requires_met(run, rubric.get("gate"))


def _adherence_pct(steps: list[dict[str, Any]]) -> float | None:
    num = sum(float(s["weight"]) for s in steps if s["status"] == "met")
    den = sum(float(s["weight"]) for s in steps if s["status"] in ("met", "wrong", "missed"))
    return round(100.0 * num / den, 1) if den else None


def score_rubric(
    run: Run,
    rubric: dict[str, Any],
    judge_map: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Score one rubric for one run. `judge_map` maps step_id -> (status, reason)
    for the llm_judge steps (empty -> those steps stay `review`)."""
    judge_map = judge_map or {}
    steps_out: list[dict[str, Any]] = []
    for step in sorted(rubric["steps"], key=lambda s: s.get("order", 0)):
        detect = step.get("detect") or {}
        weight = float(step.get("weight", 1))
        if detect.get("type") == "llm_judge":
            status, reason = judge_map.get(step["id"], ("review", ""))
        else:
            status, reason = score_auto(run, detect), ""
        steps_out.append(
            {"step_id": step["id"], "status": status, "weight": weight, "reason": reason}
        )
    skipped = [s["step_id"] for s in steps_out if s["status"] == "missed"]
    return {
        "session_id": run.session_id,
        "skill": rubric["skill"],
        "role": rubric.get("role"),
        "adherence_pct": _adherence_pct(steps_out),
        "steps": steps_out,
        "skipped": skipped,
    }


def judge_rubrics(
    runs: list[Run], rubrics: list[dict[str, Any]], model: str
) -> dict[tuple[str, str], dict[str, tuple[str, str]]]:
    """Judge the llm_judge steps of every applicable rubric: one call per
    (run, skill). Returns {(session_id, skill): {step_id: (status, reason)}}."""
    out: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
    pending = [
        (run, rb)
        for run in runs
        for rb in rubrics
        if applies(run, rb)
        and any((s.get("detect") or {}).get("type") == "llm_judge" for s in rb["steps"])
    ]
    if not pending:
        return out
    call = _make_judge_call(model)
    for run, rb in pending:
        llm_steps = [s for s in rb["steps"] if (s.get("detect") or {}).get("type") == "llm_judge"]
        rubric_payload = json.dumps(
            [{"row_id": s["id"], "expected": s["description"]} for s in llm_steps],
            indent=2,
        )
        prompt = (
            f"SKILL: {rb['skill']} — judge each row_id against the transcript.\n"
            f"RUBRIC:\n{rubric_payload}\n\nTRANSCRIPT:\n{_judge_evidence(run)}"
        )
        try:
            data = _parse_judge_json(call(_JUDGE_SYSTEM, prompt))
        except Exception as exc:
            print(
                f"  ! rubric judge failed for {run.session_id}/{rb['skill']}: {exc}",
                file=sys.stderr,
            )
            data = {}
        step_map: dict[str, tuple[str, str]] = {}
        for s in llm_steps:
            entry = data.get(s["id"])
            if isinstance(entry, dict):
                verdict, reason = entry.get("verdict"), entry.get("reason", "")
            elif isinstance(entry, str):
                verdict, reason = entry, ""
            else:
                verdict, reason = None, ""
            step_map[s["id"]] = (_norm_verdict(verdict), str(reason).strip())
        out[(run.session_id, rb["skill"])] = step_map
    return out


def score_all(
    runs: list[Run],
    rubrics: list[dict[str, Any]],
    judge: dict[tuple[str, str], dict[str, tuple[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    """Score every applicable (run, rubric) pair."""
    judge = judge or {}
    results: list[dict[str, Any]] = []
    for run in runs:
        for rb in rubrics:
            if not applies(run, rb):
                continue
            results.append(score_rubric(run, rb, judge.get((run.session_id, rb["skill"]))))
    return results
