"""Tests for the Layer-2 stage-gate in the agent proxy.

The stage-gate enforces the dependency invariant at the proxy: a training/eval step
must not advance while an upstream job it depends on (the SDG dataset, or the model
under eval) is not yet 'succeeded'. Two companion checks:
  - a validate-time 'not_ready' violation (the dispatch tool call references an
    in-flight upstream) that force-corrects Morty to report status and wait, and
  - a delegation-boundary advisory (_upstream_not_ready_advisory) that steers a
    subagent handoff the same way.
Concurrent eval-set prep (an SDG clone while training runs) is NOT gated.
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

    def test_beyond_graph_dispatch_is_gated_on_its_reference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A dispatch tool not in PIPELINE (e.g. validate_serve_job) is still gated,
        # field-driven, on the in-flight prerequisite it references.
        state = agent.SessionState(orchestrator_id="orch")
        sent = _patch(monkeypatch, {"trainjob01": _job(status="running", jtype="training")})
        out = _run(
            agent._apply_job_claim_gate(
                state,
                _dispatch_result("validate_serve_job", {"training_job_id": "trainjob01"}),
                _body(),
            )
        )
        assert len(sent) == 1
        assert "the model's training job" in sent[0]
        assert "'running'" in sent[0]
        assert out is not None
        _ = out

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

    def test_concurrent_sdg_prep_not_gated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Cloning an SDG recipe for an eval set while training still runs: validate_sdg_job
        # is not a gated dispatch, so an in-flight training_job_id in its input is ignored.
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {"trainjob01": _job(status="running", jtype="training")})
        result = _dispatch_result("validate_sdg_job", {"training_job_id": "trainjob01"})
        violations = _run(agent._not_ready_violations(state, result, {}))
        assert violations == []

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


class TestUpstreamNotReadyAdvisory:
    _TID = "a1b2c3d4-1111-2222-3333-444455556666"

    def test_eval_delegation_cites_inflight_training_advises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {self._TID: _job(status="running", jtype="training")})
        advisory = _run(
            agent._upstream_not_ready_advisory(
                state, "eval", f"evaluate the tuned model from job {self._TID}", ""
            )
        )
        assert advisory is not None
        assert "[UPSTREAM NOT READY]" in advisory
        assert "'running'" in advisory
        assert "Do NOT build, confirm, or create" in advisory

    def test_sdg_delegation_not_advised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Delegating to SDG (e.g. clone the recipe for an eval set) is not data-dependent
        # downstream, so concurrent prep while training runs is never gated here.
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {self._TID: _job(status="running", jtype="training")})
        advisory = _run(
            agent._upstream_not_ready_advisory(
                state, "sdg", f"clone recipe from training job {self._TID}", ""
            )
        )
        assert advisory is None

    def test_succeeded_upstream_no_advisory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {self._TID: _job(status="succeeded", jtype="training")})
        advisory = _run(
            agent._upstream_not_ready_advisory(
                state, "eval", f"evaluate model from job {self._TID}", ""
            )
        )
        assert advisory is None

    def test_unrelated_inflight_job_does_not_advise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An in-flight job whose type is NOT a producer of the target (here an eval job
        # mentioned alongside a training handoff) is not a prerequisite — extra work
        # done alongside must not block the node.
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {self._TID: _job(status="running", jtype="eval")})
        advisory = _run(
            agent._upstream_not_ready_advisory(
                state, "training", f"train on data; btw job {self._TID} is still running", ""
            )
        )
        assert advisory is None


class TestDelegationAdvisory:
    """The ordered delegation-boundary gate: one evaluator returns the FIRST unmet
    precondition (data fork → upstream ready), never both stacked into one handoff."""

    _TID = "a1b2c3d4-1111-2222-3333-444455556666"

    @staticmethod
    def _patch_has_data(monkeypatch: pytest.MonkeyPatch, has_data: bool) -> None:
        async def _has(*_a: Any, **_k: Any) -> bool:
            return has_data

        monkeypatch.setattr(agent, "_conversation_has_dataset", _has)

    def test_no_data_wins_and_does_not_also_stack_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No dataset chosen AND an in-flight job is cited: the data fork is the decision
        # to make first; the upstream advisory must NOT also be appended.
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {self._TID: _job(status="running", jtype="training")})
        self._patch_has_data(monkeypatch, False)
        advisory = _run(
            agent._delegation_advisory(state, "training", f"train using job {self._TID}", "")
        )
        assert advisory is not None
        assert "[DATA AVAILABILITY]" in advisory
        assert "[UPSTREAM NOT READY]" not in advisory  # no double-fire

    def test_upstream_check_applies_once_data_is_in_play(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {self._TID: _job(status="running", jtype="training")})
        self._patch_has_data(monkeypatch, True)
        advisory = _run(
            agent._delegation_advisory(
                state, "eval", f"evaluate the model from job {self._TID}", ""
            )
        )
        assert advisory is not None
        assert "[UPSTREAM NOT READY]" in advisory
        assert "[DATA AVAILABILITY]" not in advisory

    def test_no_advisory_when_data_chosen_and_upstream_succeeded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        _patch(monkeypatch, {self._TID: _job(status="succeeded", jtype="training")})
        self._patch_has_data(monkeypatch, True)
        advisory = _run(
            agent._delegation_advisory(
                state, "eval", f"evaluate the model from job {self._TID}", ""
            )
        )
        assert advisory is None
