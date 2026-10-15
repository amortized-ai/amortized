"""Tests for the job created/monitored matchers (job-lifecycle failure point)."""

from __future__ import annotations

from typing import Any

from monitor.scorer.matchers import score_auto
from monitor.scorer.model import Run


def _run(
    tool_calls: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    turn_ts: str = "2026-01-01T00:00:30",
) -> Run:
    turn = {"ts": turn_ts, "role": "training", "tool_calls": tool_calls}
    return Run(session_id="t", turns=[turn], jobs=jobs)


def _job(job_type: str, job_id: str, ts: str | None = None) -> dict[str, Any]:
    return {"kind": "job_created", "job_type": job_type, "job_id": job_id, "ts": ts}


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


def test_job_monitored_missed_when_poll_before_create() -> None:
    # poll at 00:10, job created at 00:20 -> the poll predates creation -> not counted
    run = Run(
        session_id="t",
        turns=[
            {
                "ts": "2026-01-01T00:00:10",
                "role": "training",
                "tool_calls": [_get_job("t-1", "succeeded")],
            }
        ],
        jobs=[_job("training", "t-1", ts="2026-01-01T00:00:20")],
    )
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "missed"


def test_job_monitored_met_when_poll_after_create() -> None:
    run = Run(
        session_id="t",
        turns=[
            {
                "ts": "2026-01-01T00:00:30",
                "role": "training",
                "tool_calls": [_get_job("t-1", "succeeded")],
            }
        ],
        jobs=[_job("training", "t-1", ts="2026-01-01T00:00:20")],
    )
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "met"


def test_job_monitored_status_field_not_substring() -> None:
    # a running job whose output merely mentions "a step failed" is NOT terminal
    poll = {
        "tool": "get_job",
        "role": "training",
        "input": {"job_id": "t-1"},
        "output": '{"id":"t-1","status":"running","last_error":"a step failed earlier"}',
    }
    run = _run([poll], [_job("training", "t-1")])
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "wrong"


def test_job_monitored_review_when_no_job_id_captured() -> None:
    # get_job called but input/job_id not captured (old log) -> can't correlate
    poll = {"tool": "get_job", "role": "training", "output": '{"status":"succeeded"}'}
    run = _run([poll], [_job("training", "t-1")])
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "review"


def test_job_monitored_requires_all_jobs_of_type() -> None:
    # two training jobs, only one polled -> not fully monitored
    run = _run(
        [_get_job("t-1", "succeeded")],
        [_job("training", "t-1"), _job("training", "t-2")],
    )
    assert score_auto(run, {"type": "job_monitored", "job_type": "training"}) == "missed"
