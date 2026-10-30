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
    text: str = "hi",
    event: str | None = None,
    outcome: str | None = None,
    job_type: str | None = None,
    job_mode: str | None = None,
) -> agent.MessageRequest:
    return agent.MessageRequest(
        parts=[agent.MessagePart(type="text", text=text)],
        event=event,
        outcome=outcome,
        job_type=job_type,
        job_mode=job_mode,
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


class TestPrematureCompletionGuard:
    """A subagent that renders an await-user card (show_prompt preview / validate_* confirm /
    approval present_options) OR creates an async job (create_*/retry_job) is NOT done. A
    signal_subagent_completion fired on that same turn is premature — honoring it clears the
    subagent, so the handback re-drive strands the card, or the later job_complete routes to the
    orchestrator instead of the subagent. It must be ignored so the card/job result survives and the
    subagent stays active. A genuine deliverable completion (a job_complete event for the subagent's
    own create-mode job) is exempt: a next-steps card there rides a real completion."""

    def test_premature_completion_with_card_is_suppressed(self, router: _Router) -> None:
        state = _make_state(router)
        state.subagent_id = router.add_session(
            [
                [
                    _tool_part("show_prompt", {"title": "assessor", "prompt_id": "prm_a"}),
                    _tool_part(
                        "present_options", {"step": "review", "options": ["Approve", "Revise"]}
                    ),
                    _tool_part("signal_subagent_completion", {"summary": "done"}),
                ]
            ]
        )
        state.subagent_target = "sdg"

        result = _run(agent._handle_subagent_message(state, "s1", "200", _body()))

        # Subagent keeps control; nothing handed back to the orchestrator.
        assert state.subagent_id is not None
        assert state.subagent_target == "sdg"
        assert not any(m[0] == state.orchestrator_id for m in router.sent)
        # The card survives; only the internal (premature) completion signal is stripped.
        tools = [agent._tool_name(p) for p in result["parts"] if p.get("type") == "tool"]
        assert "show_prompt" in tools
        assert "present_options" in tools
        assert "signal_subagent_completion" not in tools

    def test_premature_completion_with_job_is_suppressed(self, router: _Router) -> None:
        # A job-creation tool (retry_job / create_*) does not render an await-user card, but the
        # subagent must stay alive so the LATER job_complete event routes back to it (issue #548b).
        # A same-turn completion is premature on an ordinary (non-job_complete) turn → suppressed.
        state = _make_state(router)
        state.subagent_id = router.add_session(
            [
                [
                    _tool_part("signal_phase", {"phase": "training", "step": "validate"}),
                    _tool_part("retry_job", {"job_id": "job_a"}),
                    _tool_part("signal_subagent_completion", {"summary": "retried"}),
                ]
            ]
        )
        state.subagent_target = "training"

        result = _run(agent._handle_subagent_message(state, "s1", "retry it", _body()))

        # Subagent stays active (completion stripped); orchestrator not invoked.
        assert state.subagent_id is not None
        assert state.subagent_target == "training"
        assert not any(m[0] == state.orchestrator_id for m in router.sent)
        tools = [agent._tool_name(p) for p in result["parts"] if p.get("type") == "tool"]
        assert "retry_job" in tools
        assert "signal_subagent_completion" not in tools

    def test_deliverable_completion_with_next_steps_card_is_honored(self, router: _Router) -> None:
        state = _make_state(router, orchestrator_script=[[_text_part("What next?")]])
        state.subagent_id = router.add_session(
            [
                [
                    _tool_part("present_options", {"step": "next", "options": ["Train", "Stop"]}),
                    _tool_part("signal_subagent_completion", {"summary": "sdg done"}),
                ]
            ]
        )
        state.subagent_target = "sdg"

        _run(
            agent._handle_subagent_message(
                state,
                "s1",
                "Job j1 (SDG) finished with status: succeeded.",
                _body(event="job_complete", outcome="succeeded", job_type="sdg", job_mode="create"),
            )
        )

        # Real deliverable completion: honored → subagent released, orchestrator resumes.
        assert state.subagent_id is None
        assert any(m[0] == state.orchestrator_id for m in router.sent)


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

    def test_resume_pointer_keeps_consumers_own_step(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 526-4 repro: last_signal_step is a SINGLE shared field. The help detour delegates
        # a fresh sdg producer, which resets it; when eval is auto-resumed it must be pointed
        # back at ITS own last step (model source), not sdg's residue / eval's entry section.
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
        eval_id = router.add_session(
            [
                [
                    _tool_part(
                        "request_orchestrator_help",
                        {"need": self._NEED, "context": self._CONTEXT},
                    )
                ],
                [_text_part("resumed — back to the metrics.")],
            ]
        )
        state.subagent_id = eval_id
        state.subagent_target = "eval"
        state.last_signal_step = "model-source"  # where eval actually is before the detour
        router.queue_session(
            [
                [_text_part("What format?")],
                [
                    _tool_part(
                        "signal_subagent_completion",
                        {"summary": "SDG job sdg-1 done; 40 records"},
                    )
                ],
            ]
        )

        _run(agent._handle_subagent_message(state, "s1", "evaluate t1 on a fresh set", _body()))
        _run(agent._handle_subagent_message(state, "s1", "json, please", _body()))

        assert state.subagent_target == "eval"  # eval is back
        resume_msg = [m for m in router.sent if m[0] == eval_id][-1]
        pointer = resume_msg[3]
        assert pointer is not None
        # eval's model-source section — NOT its entry ("Confirm WHAT to evaluate").
        assert "model source" in pointer.lower()
        assert "confirm what to evaluate" not in pointer.lower()

    def test_resume_pointer_uses_help_turns_own_fresh_signal(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # P2 repro: when the SAME turn that hands back ALSO signals a fresh step, the resume
        # pointer must name THAT step, not the one left in last_signal_step from a prior turn.
        # _ensure_phase_signal (the other writer of last_signal_step) runs later in _run_turn,
        # after this handler clears subagent_target — so the help-turn's signal would otherwise
        # be dropped and the stash would capture the stale prior step (sending eval back a section).
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
        # eval advances to the eval-dataset step AND hands back in the one turn.
        eval_id = router.add_session(
            [
                [
                    _tool_part("signal_phase", {"phase": "eval", "step": "eval-dataset"}),
                    _tool_part(
                        "request_orchestrator_help",
                        {"need": self._NEED, "context": self._CONTEXT},
                    ),
                ],
                # The resumed turn presents its eval-dataset card (its owed edge) — realistic, and
                # not a stall, so the resume stays focused on the pointer (#534 P2c now stall-checks
                # resume turns, so a bare-prose resume here would be nudged).
                [
                    _tool_part("signal_phase", {"phase": "eval", "step": "eval-dataset"}),
                    _tool_part("present_options", {"step": "eval-dataset"}),
                ],
            ]
        )
        state.subagent_id = eval_id
        state.subagent_target = "eval"
        state.last_signal_step = "understand-task"  # the PRIOR (earlier) step, now stale
        router.queue_session(
            [
                [_text_part("What format?")],
                [
                    _tool_part(
                        "signal_subagent_completion",
                        {"summary": "SDG job sdg-1 done; 40 records"},
                    )
                ],
            ]
        )

        _run(agent._handle_subagent_message(state, "s1", "evaluate t1 on a fresh set", _body()))
        _run(agent._handle_subagent_message(state, "s1", "json, please", _body()))

        assert state.subagent_target == "eval"  # eval is back
        resume_msg = [m for m in router.sent if m[0] == eval_id][-1]
        pointer = resume_msg[3]
        assert pointer is not None
        # eval's eval-dataset section (the step it signalled on the hand-back turn) —
        # NOT the stale prior "Confirm WHAT to evaluate" entry section.
        assert "confirm the eval dataset" in pointer.lower()
        assert "confirm what to evaluate" not in pointer.lower()

    def test_bundled_handoff_routes_and_strips_its_prose(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A weak driver (GLM-flash) bundles premature status prose ("standing by while the set
        # generates…") with the real request_orchestrator_help call. The hand-off must still route
        # (the nudge consumer exempts it — see test_not_solo_handoff_is_honored), and the misleading
        # prose must NOT reach the user: the surfaced turn reads like a clean solo hand-off, showing
        # the hand-off's RESULT (the producer starting), not the leaf's narration.
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
        narration = "Standing by while the 100-record held-out set generates."
        eval_id = router.add_session(
            [
                [
                    _text_part(narration),
                    _tool_part(
                        "request_orchestrator_help",
                        {"need": self._NEED, "context": self._CONTEXT},
                    ),
                ],
            ]
        )
        state.subagent_id = eval_id
        state.subagent_target = "eval"
        router.queue_session([[_text_part("What format should the eval set be?")]])

        result = _run(
            agent._handle_subagent_message(state, "s1", "evaluate t1 on a fresh set", _body())
        )

        assert state.subagent_target == "sdg"  # the hand-off routed (producer is now active)
        texts = [p.get("text", "") for p in result["parts"] if p.get("type") == "text"]
        assert narration not in texts  # the leaf's premature narration was stripped
        assert any("What format" in t for t in texts)  # the producer's turn surfaces instead

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
        text = "Job d6be24ce (sdg) finished with status: succeeded."
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
        text = "Job d6be24ce (sdg) finished with status: failed."
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        forwarded_text = router.sent[0][1]
        forwarded_system = router.sent[0][3]
        assert forwarded_system == agent._SUBAGENT_JOB_FAILED_PROMPT
        assert "FAILED" in (forwarded_system or "")
        assert "signal_subagent_completion" in (forwarded_system or "")
        assert "d6be24ce" in forwarded_text  # job id KEPT in the user part

    def test_normal_message_carries_pointer_not_steer(self, router: _Router) -> None:
        # An ordinary (non job_complete) turn: the user text is untouched, and the system channel
        # now carries the resident-workflow POINTER (not a job-done steer).
        state = self._active_sdg(router)
        _run(agent._handle_subagent_message(state, "s1", "confirm", _body()))
        assert router.sent[0][1] == "confirm"  # user text unchanged
        forwarded_system = router.sent[0][3]
        assert forwarded_system is not None
        assert forwarded_system.startswith("[ACTIVE PHASE]")
        assert agent._SUBAGENT_JOB_DONE_PROMPT not in forwarded_system

    def test_deliverable_user_text_carries_no_routing_clause(self, router: _Router) -> None:
        # The hand-back no longer depends on the proxy stripping an orchestrator-worded clause: the
        # frontend sends a neutral fact (fixed at source, studio use-chat.ts), so the user part the
        # subagent sees on its OWN deliverable carries the event fact and nothing that countermands
        # the system-channel hand-back.
        state = self._active_sdg(router)
        text = "Job d6be24ce (sdg) finished with status: succeeded."
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        forwarded_text = router.sent[0][1]
        assert "present_options" not in forwarded_text  # no routing clause on the user channel
        assert "d6be24ce" in forwarded_text  # neutral event fact kept

    def test_preview_completion_is_not_deliverable(self, router: _Router) -> None:
        # The hand-back steer is gated on deliverable: a preview completion (mode != create) is an
        # intermediate step — the subagent keeps control (pointer, not the hand-back steer) and
        # continues its own workflow.
        state = self._active_sdg(router)
        text = "Job d6be24ce (sdg) finished with status: succeeded."
        _run(
            agent._handle_subagent_message(
                state, "s1", text, _body(event="job_complete", job_mode="preview")
            )
        )
        forwarded_system = router.sent[0][3]
        assert forwarded_system is not None and forwarded_system.startswith("[ACTIVE PHASE]")
        assert agent._SUBAGENT_JOB_DONE_PROMPT not in (forwarded_system or "")

    def test_cross_type_completion_is_not_deliverable(self, router: _Router) -> None:
        # A job of a DIFFERENT type than the active subagent (e.g. a split the training subagent
        # kicked off) is not its deliverable — it keeps control and continues its workflow.
        sub_id = router.add_session([[_text_part("ok")]])
        state = _make_state(router)
        state.subagent_id = sub_id
        state.subagent_target = "training"
        text = "Job s1 (SPLIT) finished with status: succeeded."
        _run(
            agent._handle_subagent_message(
                state, "s1", text, _body(event="job_complete", job_type="split")
            )
        )
        forwarded_system = router.sent[0][3]
        assert forwarded_system is not None and forwarded_system.startswith("[ACTIVE PHASE]")
        assert agent._SUBAGENT_JOB_DONE_PROMPT not in (forwarded_system or "")

    def test_structured_job_fields_decide_deliverable(self, router: _Router) -> None:
        # The structured job_type/job_mode fields drive the deliverable check (no reliance on the
        # display-text "(TYPE)" token): a training create run for the training subagent hands back.
        sub_id = router.add_session([[_text_part("ok")]])
        state = _make_state(router)
        state.subagent_id = sub_id
        state.subagent_target = "training"
        _run(
            agent._handle_subagent_message(
                state,
                "s1",
                "Job t1 finished with status: succeeded.",
                _body(event="job_complete", job_type="training", job_mode="create"),
            )
        )
        assert router.sent[0][3] == agent._SUBAGENT_JOB_DONE_PROMPT


class TestSubagentPointer:
    """A leaf subagent is re-anchored to its RESIDENT workflow section via a short pointer on
    the system channel (api/phases.subagent_pointer), delivered on the delegation handoff and on
    every ordinary turn — but never on a job_complete turn (the hand-back steer owns it there)."""

    def _delegate_and_get_system(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch, target: str, resume: bool
    ) -> str | None:
        sub_id = router.queue_session([[_text_part("ok")]])
        state = _make_state(router)
        if resume:
            state.completed_subagents[target] = sub_id
            state.last_signal_step = "confirm"  # a prior step is remembered across a resume
        _run(
            agent._maybe_delegate(
                state, "s1", (target, "ctx", resume), "go", {"info": {}, "parts": []}, _body()
            )
        )
        return next(sysm for sid, _t, _ag, sysm in router.sent if sid == sub_id)

    def test_fresh_delegation_handoff_carries_entry_pointer(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sysm = self._delegate_and_get_system(router, monkeypatch, "eval", resume=False)
        assert sysm is not None
        assert sysm.startswith("[ACTIVE PHASE]")
        assert "Step 0 — Confirm WHAT to evaluate" in sysm  # entry section named

    def test_fresh_delegation_resets_last_signal_step(self, router: _Router) -> None:
        router.queue_session([[_text_part("ok")]])
        state = _make_state(router)
        state.last_signal_step = "review"  # stale from a prior subagent
        _run(
            agent._maybe_delegate(
                state, "s1", ("sdg", "ctx", False), "go", {"info": {}, "parts": []}, _body()
            )
        )
        assert state.last_signal_step == ""  # cleared so the pointer/backfill start at entry

    def test_resume_handoff_carries_pointer(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sysm = self._delegate_and_get_system(router, monkeypatch, "eval", resume=True)
        assert sysm is not None and sysm.startswith("[ACTIVE PHASE]")

    def test_job_complete_turn_keeps_handback_steer_not_pointer(self, router: _Router) -> None:
        sdg_id = router.add_session([[_text_part("ok")]])
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"
        text = "Job a1 (sdg) finished with status: succeeded."
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        assert router.sent[0][3] == agent._SUBAGENT_JOB_DONE_PROMPT  # steer, not the pointer


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


class TestDelegationStallCheck:
    """#534 P2c: a freshly-delegated or resumed subagent turn is routed through the SAME stall nudge
    the ordinary handler runs. _maybe_delegate used to return these responses directly, so a turn
    that signalled its owed edge, narrated the hand-off, and stopped was never adjudicated —
    stranding the pipeline with no correction. The shared check now runs on every _maybe_delegate
    exit (fresh, resume, help-producer)."""

    @staticmethod
    def _signal(step: str) -> dict[str, Any]:
        return _tool_part("signal_phase", {"phase": "eval", "step": step})

    # A stalled hand-off: signals the owed-solo step but narrates instead of calling the tool.
    _STALL_THEN_CORRECT: ClassVar[list[list[dict[str, Any]]]] = [
        [_tool_part("signal_phase", {"phase": "eval", "step": "held-out-request-help"}),
         _text_part("I'll get the held-out set generated.")],
        [_tool_part("signal_phase", {"phase": "eval", "step": "held-out-request-help"}),
         _tool_part("request_orchestrator_help", {"need": "n", "context": "c"})],
    ]

    def test_fresh_delegation_turn_is_stall_nudged(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _make_state(router)
        eval_id = router.queue_session([list(t) for t in self._STALL_THEN_CORRECT])
        _run(
            agent._maybe_delegate(
                state, "s1", ("eval", "evaluate t1", False), "evaluate t1",
                {"info": {}, "parts": []}, _body(),
            )
        )
        # The stalled first turn was re-driven on the system channel naming the owed tool …
        redrive = [
            sys for sid, text, _ag, sys in router.sent if sid == eval_id and text == "" and sys
        ]
        assert redrive and "request_orchestrator_help" in redrive[0]
        # … and the re-driven turn produced the real hand-off (visible via GET, not in
        # the POST parts — the POST response contract carries only step markers; leaking
        # full GET parts causes the frontend's UUID fallback to pick stale IDs).
        fetched = _run(agent._fetch_all_assistant_parts(eval_id))
        assert any(agent._tool_name(p) == "request_orchestrator_help" for p in fetched)

    def test_resumed_turn_is_stall_nudged(
        self, router: _Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _make_state(router)
        eval_id = router.add_session([list(t) for t in self._STALL_THEN_CORRECT])
        state.completed_subagents["eval"] = eval_id  # stashed, awaiting resume
        _run(
            agent._maybe_delegate(
                state, "s1", ("eval", "resumed", True), "go on",
                {"info": {}, "parts": []}, _body(),
            )
        )
        redrive = [
            sys for sid, text, _ag, sys in router.sent if sid == eval_id and text == "" and sys
        ]
        assert redrive and "request_orchestrator_help" in redrive[0]
        fetched = _run(agent._fetch_all_assistant_parts(eval_id))
        assert any(agent._tool_name(p) == "request_orchestrator_help" for p in fetched)
