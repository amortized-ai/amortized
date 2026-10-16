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

    def test_pure_hex_word_ignored(self) -> None:
        # 8+ hex chars but all letters (no digit) — an English word, not an id.
        assert agent._extract_id_candidates("the deadbeef and facefeed cases") == set()

    def test_hex_needs_both_digit_and_letter(self) -> None:
        # Digits + letters -> an id; this is the discriminator that excludes words.
        assert "a1b2c3d4" in agent._extract_id_candidates("job a1b2c3d4")


class TestProvenanceGate:
    def _patch_corpus(
        self, monkeypatch: pytest.MonkeyPatch, corpus_by_session: dict[str, str]
    ) -> None:
        async def _fetch(sid: str, cache: Any = None) -> list[dict[str, Any]]:
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

    def test_ungrounded_id_gets_caveat_and_no_correction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The gate is now ANNOTATION-ONLY: an ungrounded id gets a silent caveat appended,
        # and the model is NEVER force-corrected (no re-run). Force-correcting restrained a
        # frontier driver, and it is unnecessary — the UI renders cards/ids from tool outputs,
        # so a fabricated id in prose is cosmetic text, not a functional error.
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_corpus(monkeypatch, {"orch": "no ids here"})

        sent: list[Any] = []

        async def _send(*a: Any, **k: Any) -> dict[str, Any]:
            sent.append(a)
            return _text_result("should never be called")

        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        result = _text_result("Submitted job a1b2c3d4 with model mixtral, accuracy 0.92.")
        out = _run(agent._apply_provenance_gate(state, result, _body()))

        assert sent == []  # no correction/re-run sent to the model
        text = agent._result_text(out)
        assert "couldn't verify" in text.lower()
        assert "a1b2c3d4" in text  # the unverified id is named in the caveat

    def test_caveat_appends_once_without_re_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even for a hard fabrication, the gate appends exactly one caveat and does not loop /
        # re-run the model (the provenance gate no longer has any force-correction cycle).
        state = agent.SessionState(orchestrator_id="orch")
        self._patch_corpus(monkeypatch, {"orch": ""})

        sent: list[Any] = []

        async def _send(*a: Any, **k: Any) -> dict[str, Any]:
            sent.append(a)
            return _text_result("unused")

        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        result = _text_result("Submitted job a1b2c3d4.")
        out = _run(agent._apply_provenance_gate(state, result, _body()))

        assert sent == []
        caveats = [p for p in out["parts"] if p.get(agent._CAVEAT_TAG)]
        assert len(caveats) == 1  # single appended caveat, no re-run cycle
        assert "a1b2c3d4" in caveats[0]["text"]

    def test_subagent_ungrounded_id_gets_caveat_no_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Works the same when a subagent is active: caveat appended, no correction sent.
        state = agent.SessionState(orchestrator_id="orch")
        state.subagent_id = "sub-eval"
        state.subagent_target = "eval"
        self._patch_corpus(monkeypatch, {"orch": "", "sub-eval": ""})

        sent: list[Any] = []

        async def _send(*a: Any, **k: Any) -> dict[str, Any]:
            sent.append(a)
            return _text_result("unused")

        monkeypatch.setattr(agent, "_proxy_send_message", _send)

        result = _text_result("Eval done on job a1b2c3d4.")
        out = _run(agent._apply_provenance_gate(state, result, _body()))

        assert sent == []
        assert "a1b2c3d4" in agent._result_text(out)

    def test_no_text_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = agent.SessionState(orchestrator_id="orch")
        result: dict[str, Any] = {"info": {}, "parts": []}
        out = _run(agent._apply_provenance_gate(state, result, _body()))
        assert out is result
