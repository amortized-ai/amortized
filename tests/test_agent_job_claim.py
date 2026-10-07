"""Tests for the job-claim verification gate in the agent proxy.

The gate checks that a job the assistant reports to the user actually MATCHES its
DB record: (A) a job reported done must really be succeeded, (B) a job reported as
freshly generated must not be an undisclosed stale reuse from an earlier
conversation, and (C) a training/eval dispatch this turn must not chain onto a stale,
undisclosed parent dataset. It force-corrects once, then appends a caveat.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from amortized.api import agent


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


def _body() -> agent.MessageRequest:
    return agent.MessageRequest(parts=[agent.MessagePart(type="text", text="hi")])


def _text_result(text: str) -> dict[str, Any]:
    return {"info": {"id": "x"}, "parts": [{"type": "text", "text": text}]}


def _dispatch_result(text: str, tool: str, parent_job_id: str) -> dict[str, Any]:
    """A reply carrying a training/eval validate dispatch tool call."""
    return {
        "info": {"id": "x"},
        "parts": [
            {"type": "text", "text": text},
            {"type": "tool", "tool": tool, "input": {"parent_job_id": parent_job_id}},
        ],
    }


def _job(**kw: Any) -> dict[str, Any]:
    base = {
        "id": "ad472e75-2d12-4819-8ed6-4a8bdb3321ad",
        "status": "succeeded",
        "type": "sdg",
        "config": {"num_records": 500},
        "created_at": datetime.now(UTC) - timedelta(days=1),  # earlier conversation
    }
    base.update(kw)
    return base


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    jobs: dict[str, dict[str, Any]],
    *,
    has_validate: bool = False,
    reply: str = "clean",
) -> list[str]:
    async def _fetch(token: str, job_cache: Any = None) -> dict[str, Any] | None:
        return jobs.get(token)

    async def _has_validate(state: Any, jtype: str, cache: Any) -> bool:
        return has_validate

    sent: list[str] = []

    async def _send(sid: str, text: str, **k: Any) -> dict[str, Any]:
        sent.append(text)
        return _text_result(reply)

    monkeypatch.setattr(agent, "_fetch_job_by_token", _fetch)
    monkeypatch.setattr(agent, "_session_has_validate", _has_validate)
    monkeypatch.setattr(agent, "_proxy_send_message", _send)
    return sent


class TestStateTruth:
    def test_succeeded_job_reported_done_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        # job created AFTER the session starts, so the reuse check can't fire either
        job = _job(status="succeeded", created_at=datetime.now(UTC) + timedelta(seconds=5))
        sent = _patch(monkeypatch, {"ad472e75": job})
        out = _run(
            agent._apply_job_claim_gate(
                state, _text_result("Your job ad472e75 finished and is ready."), _body()
            )
        )
        assert sent == []  # no correction
        assert "ad472e75 finished" in agent._result_text(out)

    def test_running_job_reported_done_corrects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(
            monkeypatch,
            {"ad472e75": _job(status="running")},
            reply="Your job ad472e75 is still running.",
        )
        out = _run(
            agent._apply_job_claim_gate(
                state, _text_result("Your job ad472e75 finished — 500 records ready."), _body()
            )
        )
        assert len(sent) == 1
        assert "status is 'running'" in sent[0]
        assert "still running" in agent._result_text(out)  # corrected reply kept


class TestStaleReuse:
    def test_stale_reused_job_corrects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The real ad472e75 case: user asked for a fresh run, a stale succeeded job
        # from an earlier conversation is reported as "just finished".
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(
            monkeypatch,
            {"ad472e75": _job()},
            has_validate=False,
            reply="Reusing the existing dataset ad472e75 (500 records) — confirm it fits.",
        )
        out = _run(
            agent._apply_job_claim_gate(
                state,
                _text_result("The SDG run finished — 500 examples ready (job ad472e75)."),
                _body(),
            )
        )
        assert len(sent) == 1
        assert "REUSING existing job ad472e75" in sent[0]
        assert "existing dataset" in agent._result_text(out)

    def test_disclosed_reuse_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {"ad472e75": _job()})
        result = _text_result("Reusing your existing dataset ad472e75 (500 records).")
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # reuse disclosed -> no correction
        assert out is result

    def test_validate_this_session_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {"ad472e75": _job()}, has_validate=True)
        result = _text_result("The SDG run finished — ready (job ad472e75).")
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # a fresh job really was submitted this session
        assert out is result

    def test_fresh_job_this_session_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        job = _job(created_at=datetime.now(UTC) + timedelta(seconds=5))
        sent = _patch(monkeypatch, {"ad472e75": job})
        result = _text_result("The SDG run finished — ready (job ad472e75).")
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # job created within this conversation
        assert out is result


class TestRemediation:
    def test_persistent_contradiction_gets_caveat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        # Model doubles down with the same stale-reuse claim on the correction.
        sent = _patch(
            monkeypatch,
            {"ad472e75": _job()},
            reply="The SDG run finished — ready (job ad472e75).",
        )
        out = _run(
            agent._apply_job_claim_gate(
                state,
                _text_result("The SDG run finished — ready (job ad472e75)."),
                _body(),
            )
        )
        assert len(sent) == agent.MAX_PROVENANCE_RETRIES == 1
        text = agent._result_text(out)
        assert "Platform records don't match" in text
        assert "ad472e75" in text

    def test_unknown_token_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {})  # token resolves to no job
        result = _text_result("Your job ad472e75 finished.")
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []
        assert out is result  # untouched; provenance gate owns fabrication


class TestDispatchProvenance:
    PARENT = "ad472e75-2d12-4819-8ed6-4a8bdb3321ad"

    def test_stale_parent_training_corrects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The real harm: training chained onto a stale SDG from an earlier
        # conversation, no SDG run this session, no reuse disclosure.
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(
            monkeypatch,
            {self.PARENT: _job()},  # succeeded sdg, created yesterday
            has_validate=False,
            reply="I'll reuse the existing dataset ad472e75 (500 records) — confirm it fits.",
        )
        out = _run(
            agent._apply_job_claim_gate(
                state,
                _dispatch_result("Setting up training.", "validate_training_job", self.PARENT),
                _body(),
            )
        )
        assert len(sent) == 1
        assert "REUSING existing dataset ad472e75" in sent[0]
        assert "train on dataset ad472e75" in sent[0]
        assert "existing dataset" in agent._result_text(out)  # corrected reply kept

    def test_fresh_parent_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        job = _job(created_at=datetime.now(UTC) + timedelta(seconds=5))  # this conversation
        sent = _patch(monkeypatch, {self.PARENT: job}, has_validate=False)
        result = _dispatch_result("Setting up training.", "validate_training_job", self.PARENT)
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # parent produced this conversation
        assert out is result

    def test_sdg_validated_this_session_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {self.PARENT: _job()}, has_validate=True)
        result = _dispatch_result("Setting up training.", "validate_training_job", self.PARENT)
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # a fresh SDG really was set up this session
        assert out is result

    def test_reuse_disclosed_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {self.PARENT: _job()}, has_validate=False)
        result = _dispatch_result(
            "Reusing the existing dataset to fine-tune.", "validate_training_job", self.PARENT
        )
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # reuse disclosed in the reply
        assert out is result

    def test_no_parent_id_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {self.PARENT: _job()}, has_validate=False)
        result = _dispatch_result("Training on an uploaded file.", "validate_training_job", "")
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # data_path / data_run_id dispatch, not an SDG chain
        assert out is result

    def test_eval_dispatch_stale_corrects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(
            monkeypatch, {self.PARENT: _job()}, has_validate=False, reply="Reusing dataset."
        )
        out = _run(
            agent._apply_job_claim_gate(
                state,
                _dispatch_result("Scoring the model.", "validate_eval_job", self.PARENT),
                _body(),
            )
        )
        assert len(sent) == 1
        assert "evaluate on dataset ad472e75" in sent[0]
        assert "Reusing dataset" in agent._result_text(out)  # corrected reply kept

    def test_non_dispatch_tool_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {self.PARENT: _job()}, has_validate=False)
        result = _dispatch_result("Listing jobs.", "list_jobs", self.PARENT)
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # not a dispatch tool
        assert out is result

    def test_persistent_dispatch_gets_caveat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Model re-sends the same stale dispatch on the correction -> caveat.
        state = agent.SessionState(orchestrator_id="orch")

        async def _fetch(token: str, job_cache: Any = None) -> dict[str, Any] | None:
            return _job() if token == self.PARENT else None

        async def _has_validate(st: Any, jt: str, c: Any) -> bool:
            return False

        sent: list[str] = []

        async def _send(sid: str, text: str, **k: Any) -> dict[str, Any]:
            sent.append(text)
            return _dispatch_result("Setting up training.", "validate_training_job", self.PARENT)

        monkeypatch.setattr(agent, "_fetch_job_by_token", _fetch)
        monkeypatch.setattr(agent, "_session_has_validate", _has_validate)
        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        out = _run(
            agent._apply_job_claim_gate(
                state,
                _dispatch_result("Setting up training.", "validate_training_job", self.PARENT),
                _body(),
            )
        )
        assert len(sent) == 1
        assert "existing dataset ad472e75" in agent._result_text(out)


class TestHelpers:
    def test_sentences_with_isolates_the_mention(self) -> None:
        text = "Job aaaa1111 finished. Job bbbb2222 is still running."
        assert "finished" in agent._sentences_with(text, "aaaa1111")
        assert "running" not in agent._sentences_with(text, "aaaa1111")

    def test_as_utc_handles_naive_and_str(self) -> None:
        assert agent._as_utc(datetime(2026, 1, 1)).tzinfo is UTC
        assert agent._as_utc("2026-01-01T00:00:00").tzinfo is UTC
        assert agent._as_utc(None) is None
