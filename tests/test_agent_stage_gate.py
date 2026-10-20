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
from datetime import UTC, datetime
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
        self._patch_has_data(monkeypatch, False)
        advisory = _run(
            agent._delegation_advisory(
                state, "training", f"train using job {self._TID}", ""
            )
        )
        assert advisory is not None
        assert "[DATA AVAILABILITY]" in advisory
        # The upstream-ready check no longer lives here (it's a tool-boundary 422); the
        # delegation advisory only surfaces the data-availability fork.
        assert "[UPSTREAM NOT READY]" not in advisory

    def test_no_advisory_when_data_chosen_and_upstream_succeeded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_has_data(monkeypatch, True)
        advisory = _run(
            agent._delegation_advisory(
                state, "eval", f"evaluate the model from job {self._TID}", ""
            )
        )
        assert advisory is None
