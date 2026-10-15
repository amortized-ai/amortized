"""Tests for the Layer-2 stage-gate in the agent proxy.

The stage-gate enforces the dependency invariant at the proxy: a committing dispatch
must not advance while an upstream job it references (the SDG dataset, or the model
under eval) is not yet 'succeeded' — a validate-time 'not_ready' violation that
force-corrects Morty to report status and wait. It is field-driven, so no job type is
special-cased. Also covered here: the data-availability fork at the delegation boundary
(_no_data_advisory).
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


def _dispatch_result(
    tool: str, inp: dict[str, str], text: str = "Here's the setup."
) -> dict[str, Any]:
    """A reply carrying a training/eval validate dispatch tool call with raw inputs."""
    return {
        "info": {"id": "x"},
        "parts": [
            {"type": "text", "text": text},
            {"type": "tool", "tool": tool, "input": inp},
        ],
    }


def _job(status: str = "succeeded", jtype: str = "training", **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "job",
        "status": status,
        "type": jtype,
        "config": {"num_records": 500},
        "created_at": datetime.now(UTC),
    }
    base.update(kw)
    return base


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    jobs: dict[str, dict[str, Any]],
    *,
    has_validate: bool = False,
    reply: str = "It's still running — I'll continue when it finishes.",
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


class TestNotReadyGate:
    def test_eval_on_inflight_training_corrects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(
            monkeypatch,
            {
                "trainjob01": _job(status="running", jtype="training"),
                "sdgjob0001": _job(status="succeeded", jtype="sdg"),
            },
        )
        out = _run(
            agent._apply_job_claim_gate(
                state,
                _dispatch_result(
                    "validate_eval_job",
                    {"training_job_id": "trainjob01", "parent_job_id": "sdgjob0001"},
                ),
                _body(),
            )
        )
        assert len(sent) == 1
        assert "the model's training job" in sent[0]
        assert "'running'" in sent[0]
        assert "Do NOT create, confirm" in sent[0]
        assert "still running" in agent._result_text(out)  # corrected reply kept

    def test_training_on_inflight_sdg_corrects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {"sdgjob0001": _job(status="queued", jtype="sdg")})
        _run(
            agent._apply_job_claim_gate(
                state,
                _dispatch_result("validate_training_job", {"parent_job_id": "sdgjob0001"}),
                _body(),
            )
        )
        assert len(sent) == 1
        assert "the upstream job that produced this job's input" in sent[0]
        assert "'queued'" in sent[0]

    def test_succeeded_upstream_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(
            monkeypatch,
            {
                "trainjob01": _job(status="succeeded", jtype="training"),
                "sdgjob0001": _job(status="succeeded", jtype="sdg"),
            },
        )
        result = _dispatch_result(
            "validate_eval_job",
            {"training_job_id": "trainjob01", "parent_job_id": "sdgjob0001"},
        )
        out = _run(agent._apply_job_claim_gate(state, result, _body()))
        assert sent == []  # every upstream succeeded -> no gate
        assert out is result

    def test_sdg_dispatch_is_field_driven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SDG gets no special case: an SDG dispatch that chains from an in-flight parent
        # is gated like any other, and one with no upstream reference is not.
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {"sdgjob0001": _job(status="running", jtype="sdg")})
        gated = _run(
            agent._not_ready_violations(
                state, _dispatch_result("validate_sdg_job", {"parent_job_id": "sdgjob0001"}), {}
            )
        )
        assert [v["kind"] for v in gated] == ["not_ready"]
        ungated = _run(
            agent._not_ready_violations(
                state, _dispatch_result("validate_sdg_job", {"num_records": 1000}), {}
            )
        )
        assert ungated == []

    def test_no_double_fire_with_stale_reuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An upstream that is BOTH in-flight AND older than the session would match the
        # stale-reuse (dispatch) check too; not_ready runs first and dedups it.
        state = agent.SessionState(orchestrator_id="orch")
        stale_running = _job(
            status="running", jtype="sdg", created_at=datetime.now(UTC) - timedelta(days=1)
        )
        _patch(monkeypatch, {"sdgjob0001": stale_running}, has_validate=False)
        result = _dispatch_result("validate_training_job", {"parent_job_id": "sdgjob0001"})
        violations = _run(agent._job_claim_violations(state, result, None, {}))
        assert len(violations) == 1
        assert violations[0]["kind"] == "not_ready"
        assert violations[0]["token"] == "sdgjob00"


class TestNoDataAdvisory:
    """The data-availability fork at the delegation boundary: when no dataset has been
    chosen this conversation, steer the subagent to let the user pick existing-vs-fresh
    rather than silently defaulting to SDG. (The upstream-not-ready advisory was removed
    — the hard create-time/dispatch gate owns 'prerequisite not finished'.)"""

    @staticmethod
    def _patch_has_data(monkeypatch: pytest.MonkeyPatch, has_data: bool) -> None:
        async def _has(*_a: Any, **_k: Any) -> bool:
            return has_data

        monkeypatch.setattr(agent, "_conversation_has_dataset", _has)

    def test_fires_when_no_dataset_chosen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_has_data(monkeypatch, False)
        advisory = _run(agent._no_data_advisory(state, "training"))
        assert advisory is not None
        assert "[DATA AVAILABILITY]" in advisory

    def test_none_when_data_already_chosen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_has_data(monkeypatch, True)
        assert _run(agent._no_data_advisory(state, "eval")) is None

    def test_none_for_non_data_dependent_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SDG is not a data-dependent downstream target, so the fork never applies.
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_has_data(monkeypatch, False)
        assert _run(agent._no_data_advisory(state, "sdg")) is None
