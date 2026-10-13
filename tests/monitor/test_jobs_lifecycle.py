"""Tests for the job created/monitored matchers (job-lifecycle failure point)."""

from __future__ import annotations

from typing import Any

from monitor.scorer.matchers import score_auto
from monitor.scorer.model import Run


def _run(tool_calls: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> Run:
    turn = {"ts": "2026-01-01T00:00:00", "role": "training", "tool_calls": tool_calls}
    return Run(session_id="t", turns=[turn], jobs=jobs)


def _job(job_type: str, job_id: str) -> dict[str, Any]:
    return {"kind": "job_created", "job_type": job_type, "job_id": job_id}


def _get_job(job_id: str, status: str) -> dict[str, Any]:
    return {
        "tool": "get_job",
        "role": "training",
        "status": "completed",
        "input": {"job_id": job_id},
        "output": f'{{"id":"{job_id}","status":"{status}"}}',
    }


# --- job_created -----------------------------------------------------------


def test_job_created_met() -> None:
    run = _run([], [_job("training", "t-1")])
    assert score_auto(run, {"type": "job_created", "job_type": "training"}) == "met"


def test_job_created_missed_when_absent() -> None:
    run = _run([], [_job("sdg", "s-1")])
    assert score_auto(run, {"type": "job_created", "job_type": "training"}) == "missed"


# --- job_monitored ---------------------------------------------------------


def test_job_monitored_na_when_not_created() -> None:
    run = _run([_get_job("t-1", "succeeded")], [])
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "n/a"


def test_job_monitored_met_when_terminal_polled() -> None:
    run = _run([_get_job("t-1", "succeeded")], [_job("training", "t-1")])
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "met"


def test_job_monitored_missed_when_never_polled() -> None:
    run = _run([], [_job("training", "t-1")])
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "missed"


def test_job_monitored_wrong_when_polled_but_not_terminal() -> None:
    run = _run([_get_job("t-1", "running")], [_job("training", "t-1")])
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "wrong"


def test_job_monitored_missed_when_polls_other_job() -> None:
    run = _run([_get_job("other", "succeeded")], [_job("training", "t-1")])
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "missed"
