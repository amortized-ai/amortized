"""Tests for the agent proxy subagent routing.

Subagents are leaves: they execute their own job and hand back. A subagent that
needs a prerequisite it can't produce (e.g. the eval agent needs a held-out
dataset) hands back to the orchestrator via request_orchestrator_help; the
orchestrator — the sole delegator — runs the producing subagent (sdg) and then
resumes the caller via the existing completed_subagents + resume path.
"""

from typing import Any, ClassVar

import pytest

from amortized.api import agent


def _tool_part(tool: str, input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool", "tool": f"mcp_amortized__{tool}", "input": input}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _fresh_state() -> agent.SessionState:
    return agent.SessionState(orchestrator_id="orch")


class _Router:
    """Stands in for OpenCode: scripted responses per session."""

    def __init__(self) -> None:
        self._next_id = 0
        # session_id -> list of parts per turn (consumed in order)
        self.script: dict[str, list[list[dict[str, Any]]]] = {}
        # session_id -> parts of the most recent response
        self.last_parts: dict[str, list[dict[str, Any]]] = {}
        # (session_id, text, agent, system) — system is the system-channel steer, if any.
        self.sent: list[tuple[str, str, str | None, str | None]] = []
        # sessions handed out by _proxy_create_session, in order
        self._new_session_queue: list[str] = []

    def add_session(self, script: list[list[dict[str, Any]]]) -> str:
        self._next_id += 1
        sid = f"sess-{self._next_id}"
        self.script[sid] = [list(parts) for parts in script]
        return sid

    def queue_session(self, script: list[list[dict[str, Any]]]) -> str:
        """Pre-script a session that _proxy_create_session will hand out."""
        sid = self.add_session(script)
        self._new_session_queue.append(sid)
        return sid

    async def send(
        self,
        session_id: str,
        text: str,
        agent: str | None = None,
        model: Any = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        self.sent.append((session_id, text, agent, system))
        script = self.script.get(session_id) or []
        parts = script.pop(0) if script else []
        self.last_parts[session_id] = parts
        return {"info": {"id": session_id}, "parts": parts}

    async def fetch_parts(self, session_id: str) -> list[dict[str, Any]]:
        return self.last_parts.get(session_id) or []


@pytest.fixture()
def router(monkeypatch: pytest.MonkeyPatch) -> _Router:
    r = _Router()
    monkeypatch.setattr(agent, "_proxy_send_message", r.send)
    monkeypatch.setattr(agent, "_fetch_all_assistant_parts", r.fetch_parts)
    async def _create() -> str:
        if r._new_session_queue:
            return r._new_session_queue.pop(0)
        return r.add_session([])

    monkeypatch.setattr(agent, "_proxy_create_session", _create)
    return r


def _make_state(
    router: _Router, orchestrator_script: list[list[dict[str, Any]]] | None = None
) -> agent.SessionState:
    orch_id = router.add_session(orchestrator_script or [])
    return agent.SessionState(orchestrator_id=orch_id)


def _run(coro: Any) -> Any:
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


def _body(
    text: str = "hi", event: str | None = None, outcome: str | None = None
) -> agent.MessageRequest:
    return agent.MessageRequest(
        parts=[agent.MessagePart(type="text", text=text)], event=event, outcome=outcome
    )


class TestJobSucceeded:
    def test_structured_outcome_preferred(self) -> None:
        # The structured field decides, regardless of display text.
        assert agent._job_succeeded(_body(outcome="succeeded"), "whatever wording")
        assert not agent._job_succeeded(_body(outcome="failed"), "status: succeeded")

    def test_legacy_substring_fallback_when_no_outcome(self) -> None:
        assert agent._job_succeeded(_body(), "Job x finished with status: succeeded.")
        assert not agent._job_succeeded(_body(), "Job x finished with status: failed.")


class TestSubagentCompletion:
    """A subagent that signals completion hands control back to the orchestrator —
    the sole delegator — which resumes the conversation."""

    def test_subagent_completion_without_stack_resumes_orchestrator(self, router: _Router) -> None:
        state = _make_state(router, orchestrator_script=[[ _text_part("What next?") ]])
        sdg_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "done"})]]
        )
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"

        _run(agent._handle_subagent_message(state, "s1", "confirm", _body()))

        assert state.subagent_id is None  # orchestrator takes over
        orch_msg = next(m for m in router.sent if m[0] == state.orchestrator_id)
        assert orch_msg[2] == "morty"

    def test_internal_tools_stripped_from_response(self, router: _Router) -> None:
        # The completion signal and the orchestrator's onward delegation are internal
        # routing — never surfaced to the user.
        _ = router.queue_session([[_text_part("What kind of dataset?")]])
        state = _make_state(
            router,
            orchestrator_script=[
                [_tool_part("delegate_to_subagent", {"target": "sdg", "context": "c"})]
            ],
        )
        state.subagent_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "eval done"})]]
        )
        state.subagent_target = "eval"

        result = _run(agent._handle_subagent_message(state, "s1", "done", _body()))
        tools = [p for p in result["parts"] if p.get("type") == "tool"]
        assert tools == []


class TestRequestOrchestratorHelp:
    """A subagent is a leaf: it cannot delegate. When it needs a prerequisite it can't
    produce itself it calls request_orchestrator_help, handing back to the orchestrator.
    The orchestrator's one job is to run the producing subagent (sdg); when that producer
    completes, the proxy resumes the caller AUTOMATICALLY (backend-driven, not a model
    decision) — eval → orch → sdg → (auto-resume) → eval."""

    _NEED: ClassVar[str] = "a held-out eval dataset for training job t1"
    _CONTEXT: ClassVar[str] = "evaluating t1; judge gpt-oss; resume at metrics design (Step 3)"

    def test_help_request_stashes_caller_and_seeds_orchestrator(self, router: _Router) -> None:
        # eval (active) hands back for help; the orchestrator acknowledges in text, so
        # this isolates the stash + seed (no onward delegation).
        state = _make_state(router, orchestrator_script=[[_text_part("On it.")]])
        eval_id = router.add_session(
            [
                [
                    _tool_part(
                        "request_orchestrator_help",
                        {"need": self._NEED, "context": self._CONTEXT},
                    )
                ]
            ]
        )
        state.subagent_id = eval_id
        state.subagent_target = "eval"

        _run(agent._handle_subagent_message(state, "s1", "set it up", _body()))

        # The caller is stashed and the orchestrator took over — but because the orchestrator
        # only acknowledged in text (no onward delegate), there's no producer to await, so the
        # auto-resume stays DISARMED. Arming it here off the requester alone is the weak-latch
        # bug: any unrelated later completion would then resurrect eval.
        assert state.completed_subagents["eval"] == eval_id
        assert state.pending_help_target is None
        assert state.pending_help_producer is None
        assert state.subagent_id is None
        assert state.subagent_target is None
        # The orchestrator got the structured ask (as morty), carrying the need + the
        # caller's context, and is told to delegate to the producer. No resume
        # instruction — the proxy does the resume automatically.
        orch_msg = next(m for m in router.sent if m[0] == state.orchestrator_id)
        assert orch_msg[2] == "morty"
        assert "[SUBAGENT NEEDS HELP]" in orch_msg[1]
        assert self._NEED in orch_msg[1]
        assert "resume at metrics design" in orch_msg[1]  # the caller's context rode along
        assert "delegate_to_subagent" in orch_msg[1]

    def test_help_flow_auto_resumes_requester_on_producer_completion(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Isolate routing from the data-availability advisory (the harness can't see the
        # sdg job the resume depends on).
        async def _no_advisory(*_a: Any, **_k: Any) -> None:
            return None

        monkeypatch.setattr(agent, "_delegation_advisory", _no_advisory)

        # The orchestrator only makes the FIRST hop (delegate to sdg). It is NOT scripted
        # to resume — the proxy resumes the requester automatically when sdg completes.
        state = _make_state(
            router,
            orchestrator_script=[
                [
                    _tool_part(
                        "delegate_to_subagent",
                        {"target": "sdg", "context": "held-out set for t1"},
                    )
                ],
            ],
        )
        # eval: first turn asks for help; second turn (auto-resume) continues its workflow.
        eval_id = router.add_session(
            [
                [
                    _tool_part(
                        "request_orchestrator_help",
                        {"need": self._NEED, "context": self._CONTEXT},
                    )
                ],
                [_text_part("Great — now let's design the eval metrics.")],
            ]
        )
        state.subagent_id = eval_id
        state.subagent_target = "eval"
        sdg_id = router.queue_session(
            [
                [_text_part("What format should the eval set be?")],
                [
                    _tool_part(
                        "signal_subagent_completion",
                        {"summary": "SDG job sdg-1 done; 40 records"},
                    )
                ],
            ]
        )

        # Turn A: eval → request_orchestrator_help → orchestrator → delegate(sdg).
        _run(
            agent._handle_subagent_message(
                state, "s1", "evaluate t1 on a fresh set", _body()
            )
        )
        assert state.subagent_target == "sdg"  # sdg is now active
        assert state.subagent_id == sdg_id
        assert state.completed_subagents == {"eval": eval_id}  # caller stashed
        assert state.pending_help_target == "eval"  # armed for auto-resume
        sdg_handoff = next(m for m in router.sent if m[0] == sdg_id)
        assert sdg_handoff[2] == "sdg"
        assert self._NEED in sdg_handoff[1]  # the need rode into the sdg handoff
        # Regression guard: the orchestrator's seed (its instruction sheet — carrying the
        # caller's context and "delegate to sdg…") must NOT leak into the leaf's handoff.
        # The producer is a leaf: it sees only the dataset ask, never the help framing or
        # the caller's resume note.
        assert "[SUBAGENT NEEDS HELP]" not in sdg_handoff[1]
        assert "resume at metrics design" not in sdg_handoff[1]

        # Turn B: sdg completes → proxy auto-resumes eval, with NO orchestrator turn.
        sent_before = len(router.sent)
        _run(agent._handle_subagent_message(state, "s1", "json, please", _body()))
        assert state.subagent_target == "eval"  # control is back on the ORIGINAL eval
        assert state.subagent_id == eval_id
        assert "eval" not in state.completed_subagents  # popped on resume
        assert state.pending_help_target is None  # cleared
        resume_msg = [m for m in router.sent if m[0] == eval_id][-1]
        assert resume_msg[2] == "eval"
        assert "[RESUMED]" in resume_msg[1]
        # The orchestrator was NOT consulted for the resume (backend-driven).
        assert all(m[0] != state.orchestrator_id for m in router.sent[sent_before:])

    def test_completion_wins_over_simultaneous_help(self, router: _Router) -> None:
        # A turn carrying BOTH signal_subagent_completion AND request_orchestrator_help:
        # completion wins (a finishing subagent has nothing left to be helped with), so
        # it hands back to the orchestrator and arms no pending help target.
        state = _make_state(router, orchestrator_script=[[_text_part("What next?")]])
        state.subagent_id = router.add_session(
            [
                [
                    _tool_part("signal_subagent_completion", {"summary": "done"}),
                    _tool_part("request_orchestrator_help", {"need": "n", "context": "c"}),
                ]
            ]
        )
        state.subagent_target = "sdg"

        _run(agent._handle_subagent_message(state, "s1", "x", _body()))

        assert state.pending_help_target is None  # help ignored; no auto-resume armed
        orch_msg = next(m for m in router.sent if m[0] == state.orchestrator_id)
        assert "[SUBAGENT COMPLETED]" in orch_msg[1]  # the completion path was taken

    def test_unrelated_completion_does_not_resume_pending_consumer(self, router: _Router) -> None:
        # Weak-latch guard (review #1): with eval awaiting sdg, a DIFFERENT subagent (training)
        # finishing must NOT resurrect eval — only the awaited producer (sdg) does. Before the
        # fix any completer != requester triggered the resume.
        state = _make_state(router, orchestrator_script=[[_text_part("What next?")]])
        eval_id = router.add_session([[_text_part("SHOULD NOT RUN")]])  # stashed; must stay put
        state.completed_subagents = {"eval": eval_id}
        state.pending_help_target = "eval"
        state.pending_help_producer = "sdg"  # eval is waiting on sdg specifically
        training_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "training done"})]]
        )
        state.subagent_id = training_id
        state.subagent_target = "training"

        _run(agent._handle_subagent_message(state, "s1", "x", _body()))

        assert state.subagent_id != eval_id  # eval NOT resumed
        assert all(m[0] != eval_id for m in router.sent)  # stashed eval never messaged
        assert state.pending_help_target == "eval"  # still awaiting its real producer (sdg)
        orch_msg = next(m for m in router.sent if m[0] == state.orchestrator_id)
        assert "[SUBAGENT COMPLETED]" in orch_msg[1]  # unrelated completion → orchestrator path

    def test_resume_that_itself_completes_releases_control(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review #2: if the resumed consumer finishes right in the resume turn, control must be
        # released — not left assigned to it (with its completion stripped) until a later turn.
        async def _no_advisory(*_a: Any, **_k: Any) -> None:
            return None

        monkeypatch.setattr(agent, "_delegation_advisory", _no_advisory)
        state = _make_state(router)
        eval_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "eval done; scored 0.9"})]]
        )
        state.completed_subagents = {"eval": eval_id}
        state.pending_help_target = "eval"
        state.pending_help_producer = "sdg"
        sdg_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "sdg-1 done; 40 records"})]]
        )
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"

        _run(agent._handle_subagent_message(state, "s1", "x", _body()))

        # sdg completed → eval auto-resumed; eval completed in the resume → control released.
        assert state.subagent_id is None
        assert state.subagent_target is None
        assert state.completed_subagents.get("eval") == eval_id  # re-stashed on its completion
        assert state.pending_help_target is None

    def test_failed_producer_resumes_consumer_with_failure_context(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Review #3: a failed producer job hands back (not stall) and the consumer is resumed
        # with a FAILURE context — never told the data "is ready".
        async def _no_advisory(*_a: Any, **_k: Any) -> None:
            return None

        monkeypatch.setattr(agent, "_delegation_advisory", _no_advisory)
        state = _make_state(router)
        eval_id = router.add_session([[_text_part("I'll tell the user the data isn't available.")]])
        state.completed_subagents = {"eval": eval_id}
        state.pending_help_target = "eval"
        state.pending_help_producer = "sdg"
        sdg_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "job sdg-1 FAILED"})]]
        )
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"

        text = "Job sdg-1 (sdg) finished with status: failed."
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))

        sdg_msg = next(m for m in router.sent if m[0] == sdg_id)
        # failure hand-back delivered on the system channel (4th tuple element)
        assert sdg_msg[3] == agent._SUBAGENT_JOB_FAILED_PROMPT
        resume_msg = [m for m in router.sent if m[0] == eval_id][-1]
        assert "could NOT be produced" in resume_msg[1]
        assert "is ready" not in resume_msg[1]


class TestJobCompleteHandback:
    """When an active subagent's delegated job finishes, the server steers it to
    hand control back instead of letting it loop on post-job inspection calls."""

    def _active_sdg(self, router: _Router) -> agent.SessionState:
        # Subagent responds with plain text (no completion signal) so the handler
        # returns normally and we can inspect what text was forwarded to it.
        sdg_id = router.add_session([[_text_part("ok")]])
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"
        return state

    def test_success_event_delivers_handback_on_system_channel(self, router: _Router) -> None:
        # The hand-back IMPERATIVE now rides the system channel (not appended to the user text),
        # so it reads as a system instruction rather than a fake user turn. The neutral event
        # fact (the job id) stays in the user part so the hand-back can report it.
        state = self._active_sdg(router)
        text = (
            "Job d6be24ce (sdg) finished with status: succeeded."
            " Use present_options to suggest next steps to the user."
        )
        _run(
            agent._handle_subagent_message(
                state, "s1", text, _body(event="job_complete")
            )
        )
        forwarded_text = router.sent[0][1]
        forwarded_system = router.sent[0][3]
        assert forwarded_system == agent._SUBAGENT_JOB_DONE_PROMPT
        assert "signal_subagent_completion" in (forwarded_system or "")
        assert "d6be24ce" in forwarded_text  # job id KEPT in the user part
        assert agent._SUBAGENT_JOB_DONE_PROMPT not in forwarded_text  # NOT on the user channel

    def test_failed_job_delivers_failure_handback_on_system_channel(
        self, router: _Router
    ) -> None:
        # A failed job also gets a defined hand-back (previously it was left untouched, so the
        # producer kept control and the flow could stall — Shabana review #3) — on the system
        # channel, same as success.
        state = self._active_sdg(router)
        text = (
            "Job d6be24ce (sdg) finished with status: failed."
            " Use present_options to suggest next steps to the user."
        )
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        forwarded_text = router.sent[0][1]
        forwarded_system = router.sent[0][3]
        assert forwarded_system == agent._SUBAGENT_JOB_FAILED_PROMPT
        assert "FAILED" in (forwarded_system or "")
        assert "signal_subagent_completion" in (forwarded_system or "")
        assert "d6be24ce" in forwarded_text  # job id KEPT in the user part

    def test_preview_completion_steers_to_full_run_not_handback(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A PREVIEW (10-row sample) completing is NOT the deliverable: steer the producer to run
        # the full create run, not hand back — else the orchestrator advances to training on a
        # preview. The preview steer rides the system channel (like the generic hand-back).
        async def _fetch(token: str) -> dict[str, Any]:
            return {"type": "sdg", "status": "succeeded", "config": {"mode": "preview"}}

        monkeypatch.setattr(agent, "_fetch_job_by_token", _fetch)
        state = self._active_sdg(router)
        text = "Job d6be24ce (sdg) finished with status: succeeded."
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        forwarded_system = router.sent[0][3]
        assert forwarded_system == agent._SUBAGENT_PREVIEW_DONE_PROMPT  # run the full create run
        assert forwarded_system != agent._SUBAGENT_JOB_DONE_PROMPT  # NOT a completion hand-back
        assert "d6be24ce" in router.sent[0][1]  # job id kept in the user part

    def test_create_completion_still_hands_back(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The full create run (no preview mode) completing IS a real completion → hand back.
        async def _fetch(token: str) -> dict[str, Any]:
            return {"type": "sdg", "status": "succeeded", "config": {"mode": "create"}}

        monkeypatch.setattr(agent, "_fetch_job_by_token", _fetch)
        state = self._active_sdg(router)
        text = "Job d6be24ce (sdg) finished with status: succeeded."
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        assert router.sent[0][3] == agent._SUBAGENT_JOB_DONE_PROMPT

    def test_normal_message_not_rewritten(self, router: _Router) -> None:
        # A normal (non job_complete) turn keeps the clean user text verbatim (not rewritten);
        # the active phase block rides the SYSTEM channel rather than being prepended to the user
        # text. (The job-done hand-back steer, also system-channel, fires only on job_complete.)
        state = self._active_sdg(router)
        _run(agent._handle_subagent_message(state, "s1", "confirm", _body()))
        sent = router.sent[0][1]
        assert sent == "confirm"  # user text preserved verbatim, no injection prepended
        system = router.sent[0][3]
        assert system is not None and "phase-1-route" in system  # phase block on system channel


def _signal_phase_results(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in parts if agent._tool_name(p) == "signal_phase"]


class TestPhaseBackfill:
    """When an active subagent's turn carries no signal_phase, the server backfills
    one from subagent_target so the UI progress bar is never stale. The backfill is
    applied by _run_turn to EVERY handler return path (completion hand-back,
    subagent help hand-back, normal), so it is unit-tested here on the pure
    _ensure_phase_signal function it delegates to."""

    def _sub_state(self) -> agent.SessionState:
        state = agent.SessionState(orchestrator_id="orch")
        state.subagent_id = "sub"
        state.subagent_target = "sdg"
        return state

    def test_backfills_phase_when_subagent_forgets(self) -> None:
        result = {"parts": [_text_part("working on it")]}
        out = agent._ensure_phase_signal(self._sub_state(), result)

        signals = _signal_phase_results(out["parts"])
        assert len(signals) == 1
        assert agent._get_tool_input(signals[0]).get("phase") == "sdg"

    def test_does_not_duplicate_when_subagent_signals(self) -> None:
        state = self._sub_state()
        result = {
            "parts": [
                _tool_part("signal_phase", {"phase": "sdg", "step": "gather_requirements"})
            ]
        }
        out = agent._ensure_phase_signal(state, result)

        assert len(_signal_phase_results(out["parts"])) == 1
        assert state.last_signal_step == "gather_requirements"  # remembered for next turn

    def test_backfill_carries_last_signalled_step(self) -> None:
        state = self._sub_state()
        state.last_signal_step = "confirm"
        result = {"parts": [_text_part("still working")]}
        out = agent._ensure_phase_signal(state, result)

        signals = _signal_phase_results(out["parts"])
        assert agent._get_tool_input(signals[0]).get("step") == "confirm"


class TestNoDataAdvisory:
    """Pre-delegation advisory: when THIS conversation has no dataset path, a
    training/eval handoff is annotated so the subagent gets data first instead of
    discovering the gap after it spins up. Scoping lives in _conversation_has_dataset
    (exercised in TestConversationHasDataset); here we test injection + wording."""

    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, *, has_data: bool) -> None:
        async def _has(*_a: Any, **_k: Any) -> bool:
            return has_data

        monkeypatch.setattr(agent, "_conversation_has_dataset", _has)

    def _delegate(self, router: _Router, target: str) -> tuple[str, list[str]]:
        sub_id = router.queue_session([[_text_part("ok")]])
        state = _make_state(router)
        _run(
            agent._maybe_delegate(
                state,
                "s1",
                (target, "orig ctx", False),
                "go",
                {"info": {}, "parts": []},
                _body(),
            )
        )
        sent = [text for sid, text, _ag, _sys in router.sent if sid == sub_id]
        return sub_id, sent

    def test_advisory_prepended_to_training_handoff(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, has_data=False)
        _sub_id, sent = self._delegate(router, "training")
        assert sent
        assert "[DATA AVAILABILITY]" in sent[0]
        assert "train on" in sent[0]
        assert "orig ctx" in sent[0]  # original context preserved

    def test_advisory_for_eval_uses_evaluate_verb(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, has_data=False)
        _sub_id, sent = self._delegate(router, "eval")
        assert "[DATA AVAILABILITY]" in sent[0]
        assert "evaluate on" in sent[0]

    def test_no_advisory_when_conversation_has_dataset(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, has_data=True)
        _sub_id, sent = self._delegate(router, "training")
        assert "[DATA AVAILABILITY]" not in sent[0]

    def test_no_advisory_for_non_data_dependent_target(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # sdg itself produces data — never annotated even with no dataset yet.
        self._patch(monkeypatch, has_data=False)
        _sub_id, sent = self._delegate(router, "sdg")
        assert "[DATA AVAILABILITY]" not in sent[0]

    def test_advisory_suppressed_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Fail-safe: any error determining conversation state → no advisory.
        async def _boom(*_a: Any, **_k: Any) -> bool:
            raise RuntimeError("opencode unreachable")

        monkeypatch.setattr(agent, "_conversation_has_dataset", _boom)
        state = agent.SessionState(orchestrator_id="orch")
        assert _run(agent._no_data_advisory(state, "training")) is None


class TestConversationHasDataset:
    """The conversation-scoped dataset signals that gate the advisory. Only tool
    facts count: fresh SDG (`validate_sdg_job`) or engaging a specific existing
    dataset (`get_dataset`-family). Browsing the catalog, or merely naming a dataset
    in text (including the orchestrator handoff relayed as user text), must NOT count
    — that is availability, not a choice, and the fork must still fire."""

    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        async def _fetch(_sid: str, _cache: Any = None) -> list[dict[str, Any]]:
            return messages or []

        monkeypatch.setattr(agent, "_fetch_all_messages", _fetch)

    @staticmethod
    def _user_msg(text: str) -> dict[str, Any]:
        return {"info": {"role": "user"}, "parts": [_text_part(text)]}

    def _run_check(self) -> bool:
        state = agent.SessionState(orchestrator_id="orch")
        return _run(agent._conversation_has_dataset(state, {}))

    def test_fresh_sdg_this_session_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._setup(
            monkeypatch,
            messages=[{"info": {}, "parts": [_tool_part("validate_sdg_job", {})]}],
        )
        assert self._run_check() is True

    def test_specific_dataset_tool_touched_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(
            monkeypatch,
            messages=[{"info": {}, "parts": [_tool_part("get_dataset_samples", {})]}],
        )
        assert self._run_check() is True

    def test_merely_listing_datasets_is_not_a_choice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Browsing the catalog is not choosing a dataset — the fork must still fire.
        self._setup(
            monkeypatch,
            messages=[{"info": {}, "parts": [_tool_part("list_datasets", {})]}],
        )
        assert self._run_check() is False

    def test_dataset_id_named_in_text_is_not_a_choice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression: a dataset id merely named in text — including the orchestrator
        # handoff (relayed into the subagent session as a user-role message, listing
        # the available datasets as platform state) — is availability, not a choice.
        # Only a dataset *tool* fact counts, so the fork must still fire.
        self._setup(
            monkeypatch,
            messages=[
                self._user_msg(
                    "[CONTEXT] PLATFORM STATE (relevant artifacts): "
                    "available dataset a1b2c3d4 (sdg, succeeded)"
                )
            ],
        )
        assert self._run_check() is False

    def test_bare_training_request_has_no_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(
            monkeypatch,
            messages=[self._user_msg("train a model to assess RFEs")],
        )
        assert self._run_check() is False
