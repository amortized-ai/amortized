"""Tests for the anti-fabrication provenance gate in the agent proxy.

The gate blocks identifiers the assistant states to the user that appear nowhere
in the conversation's tool I/O or user/system text, and force-corrects the model.
"""

import asyncio
from typing import Any

import pytest

from amortized.api import agent


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


def _body() -> agent.MessageRequest:
    return agent.MessageRequest(parts=[agent.MessagePart(type="text", text="hi")])


def _text_result(text: str) -> dict[str, Any]:
    return {"info": {"id": "x"}, "parts": [{"type": "text", "text": text}]}


class TestExtractIdCandidates:
    def test_uuid_detected(self) -> None:
        uid = "b9afca6b-1234-4abc-8def-0123456789ab"
        assert uid in agent._extract_id_candidates(f"Job {uid} finished.")

    def test_hex_fragment_with_letter_detected(self) -> None:
        assert "a1b2c3d4" in agent._extract_id_candidates("Submitted job a1b2c3d4.")

    def test_pure_decimal_number_ignored(self) -> None:
        # A record count — not an identifier.
        assert agent._extract_id_candidates("Generated 12345678 records.") == set()

    def test_short_hex_ignored(self) -> None:
        assert agent._extract_id_candidates("value 0.92 and ab12") == set()


class TestProvenanceGate:
    def _patch_corpus(
        self, monkeypatch: pytest.MonkeyPatch, corpus_by_session: dict[str, str]
    ) -> None:
        async def _fetch(sid: str) -> list[dict[str, Any]]:
            blob = corpus_by_session.get(sid, "")
            if not blob:
                return []
            return [
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "mcp_amortized__get_job",
                            "input": {},
                            "state": {"output": blob},
                        }
                    ],
                }
            ]

        monkeypatch.setattr(agent, "_fetch_all_messages", _fetch)

    def test_grounded_id_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        uid = "b9afca6b-1234-4abc-8def-0123456789ab"
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_corpus(monkeypatch, {"orch": f"job {uid} status succeeded"})

        sent: list[str] = []

        async def _send(*a: Any, **k: Any) -> dict[str, Any]:
            sent.append("called")
            return _text_result("corrected")

        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        result = _text_result(f"Your job {uid} is done.")
        out = _run(agent._apply_provenance_gate(state, result, _body()))

        assert out is result  # unchanged
        assert sent == []  # no correction triggered

    def test_ungrounded_id_triggers_correction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_corpus(monkeypatch, {"orch": "no ids here"})

        corrections: list[tuple[str, Any]] = []

        async def _send(sid: str, text: str, **k: Any) -> dict[str, Any]:
            corrections.append((sid, k.get("agent")))
            return _text_result("Here are your options — no fabricated ids.")

        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        result = _text_result("Submitted job a1b2c3d4 with model mixtral, accuracy 0.92.")
        out = _run(agent._apply_provenance_gate(state, result, _body()))

        assert len(corrections) == 1
        assert corrections[0] == ("orch", "morty")
        assert "no fabricated ids" in agent._result_text(out)

    def test_persistent_fabrication_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_corpus(monkeypatch, {"orch": ""})

        calls = 0

        async def _send(*a: Any, **k: Any) -> dict[str, Any]:
            # Model doubles down with the same fabricated id on every retry.
            nonlocal calls
            calls += 1
            return _text_result("It's definitely job a1b2c3d4.")

        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        result = _text_result("Submitted job a1b2c3d4.")
        out = _run(agent._apply_provenance_gate(state, result, _body()))

        # Every retry is attempted before the reply is blocked outright.
        assert calls == agent.MAX_PROVENANCE_RETRIES
        text = agent._result_text(out)
        assert "a1b2c3d4" not in text
        assert "don't have verified details" in text

    def test_correction_succeeds_within_retry_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_corpus(monkeypatch, {"orch": ""})

        calls = 0

        async def _send(*a: Any, **k: Any) -> dict[str, Any]:
            # Fabricates on the first correction, then self-heals on the second.
            nonlocal calls
            calls += 1
            if calls < 2:
                return _text_result("Still job a1b2c3d4.")
            return _text_result("Here are your grounded next steps.")

        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        result = _text_result("Submitted job a1b2c3d4.")
        out = _run(agent._apply_provenance_gate(state, result, _body()))

        assert calls == 2  # stopped as soon as the reply was clean
        assert "grounded next steps" in agent._result_text(out)

    def test_correction_routed_to_active_subagent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        state.subagent_id = "sub-eval"
        state.subagent_target = "eval"
        self._patch_corpus(monkeypatch, {"orch": "", "sub-eval": ""})

        routed: list[tuple[str, Any]] = []

        async def _send(sid: str, text: str, **k: Any) -> dict[str, Any]:
            routed.append((sid, k.get("agent")))
            return _text_result("clean")

        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        result = _text_result("Eval done on job a1b2c3d4.")
        _run(agent._apply_provenance_gate(state, result, _body()))

        assert routed == [("sub-eval", "eval")]

    def test_no_text_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        result: dict[str, Any] = {"info": {}, "parts": []}
        out = _run(agent._apply_provenance_gate(state, result, _body()))
        assert out is result
