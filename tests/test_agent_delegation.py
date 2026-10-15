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
        self.sent: list[tuple[str, str, str | None]] = []
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
    ) -> dict[str, Any]:
        self.sent.append((session_id, text, agent))
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
        assert agent._SUBAGENT_JOB_FAILED_PROMPT in sdg_msg[1]  # defined failure hand-back
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

    def test_success_event_rewrites_to_handback(self, router: _Router) -> None:
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
        forwarded = router.sent[0][1]
        assert agent._SUBAGENT_JOB_DONE_PROMPT in forwarded  # hand-back steer appended
        assert "d6be24ce" in forwarded  # [5] the job id is KEPT, not replaced away
        assert "signal_subagent_completion" in forwarded

    def test_failed_job_rewrites_to_failure_handback(self, router: _Router) -> None:
        # A failed job also gets a defined hand-back (previously it was left untouched, so the
        # producer kept control and the flow could stall — Shabana review #3).
        state = self._active_sdg(router)
        text = (
            "Job d6be24ce (sdg) finished with status: failed."
            " Use present_options to suggest next steps to the user."
        )
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        forwarded = router.sent[0][1]
        assert agent._SUBAGENT_JOB_FAILED_PROMPT in forwarded  # failure hand-back appended
        assert "d6be24ce" in forwarded  # the job id is KEPT, not replaced away
        assert "FAILED" in forwarded
        assert "signal_subagent_completion" in forwarded

    def test_normal_message_not_rewritten(self, router: _Router) -> None:
        state = self._active_sdg(router)
        _run(agent._handle_subagent_message(state, "s1", "confirm", _body()))
        assert router.sent[0][1] == "confirm"


class TestEnforceSingleInteraction:
    """A turn either ASKS one question or CONFIRMS one job, never both. Whichever
    interaction leads the turn wins; the other kind (and repeats) is stripped so
    the UI never shows a question batched with a confirm card."""

    def test_ask_turn_keeps_only_first_question_but_preserves_passive_parts(self) -> None:
        # Extra questions and a premature confirm card are stripped, but passive
        # content (text, show_* cards) is kept so the user still sees context.
        result = {
            "parts": [
                _text_part("here are your options"),
                _tool_part("present_options", {"question": "which teacher?"}),
                _tool_part("present_options", {"question": "how many records?"}),
                _tool_part("show_prompt", {}),
                _tool_part("present_options", {"question": "prompt ok?"}),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
            ]
        }
        out = agent._enforce_single_interaction(result, _fresh_state())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools.count("present_options") == 1  # exactly one question survives
        assert "validate_sdg_job" not in tools  # premature confirm card dropped
        assert "show_prompt" in tools  # passive display card preserved
        assert any(p.get("type") == "text" for p in out["parts"])  # text preserved

    def test_ask_turn_preserves_text_written_after_the_question(self) -> None:
        # Regression: a driver wrote the proposed metric-set table as text AFTER
        # the approval question; it must not be dropped, or the user approves
        # option buttons with no metrics shown.
        result = {
            "parts": [
                _tool_part("present_options", {"question": "Approve this metric set?"}),
                _text_part("| Criterion | What it checks |\n|---|---|\n| accuracy | ... |"),
                _tool_part("present_options", {"question": "Approve this metric set?"}),
            ]
        }
        out = agent._enforce_single_interaction(result, _fresh_state())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools == ["present_options"]  # the duplicate question is stripped
        texts = [p for p in out["parts"] if p.get("type") == "text"]
        assert texts and "Criterion" in texts[0]["text"]  # the metric table survives

    @staticmethod
    def _training_ready() -> agent.SessionState:
        # A state whose training prerequisites are met, so these tests isolate the
        # single-interaction invariant (not the training precondition gate).
        state = _fresh_state()
        state.shown_vram = True
        state.confirmed_data = True
        state.offered_choice.add("training")
        return state

    def test_confirm_turn_drops_options_stacked_under_the_card(self) -> None:
        # validate_* first, then a redundant "confirm?" present_options -> the
        # confirmation card wins; the stacked question is dropped, passive cards stay.
        result = {
            "parts": [
                _tool_part("validate_training_job", {"algorithm": "sft"}),
                _tool_part("show_vram_estimate", {}),
                _tool_part("signal_phase", {"phase": "training", "step": "confirm"}),
                _tool_part("present_options", {"question": "Submit this job?"}),
            ]
        }
        out = agent._enforce_single_interaction(result, self._training_ready())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert "present_options" not in tools  # no options under the confirm card
        assert tools == ["validate_training_job", "show_vram_estimate", "signal_phase"]

    def test_errored_card_does_not_shadow_the_successful_retry(self) -> None:
        # The model's validate_* errored, it self-corrected and re-validated in the SAME
        # turn. The gate must keep the COMPLETED card (so the UI renders it) and suppress
        # the dead errored one — not keep the error and strip the live card (which left the
        # confirm card showing as a collapsed error chip).
        err = {
            "type": "tool",
            "tool": "mcp_amortized__validate_sdg_job",
            "callID": "err-1",
            "state": {"status": "error", "error": "Input validation error"},
        }
        ok = {
            "type": "tool",
            "tool": "mcp_amortized__validate_sdg_job",
            "callID": "ok-1",
            "state": {"status": "completed", "input": {"mode": "preview"}},
        }
        state = _fresh_state()
        out = agent._enforce_single_interaction({"parts": [err, ok]}, state)
        kept = [p.get("callID") for p in out["parts"] if p.get("type") == "tool"]
        assert kept == ["ok-1"]  # the live card survives; the dead errored one is dropped
        # get_session_messages filters suppressed_call_ids, so the live card must NOT be
        # suppressed (client renders it) while the errored chip IS hidden.
        assert "ok-1" not in state.suppressed_call_ids
        assert "err-1" in state.suppressed_call_ids

    def test_keeps_the_last_valid_confirm_card(self) -> None:
        # Two successful validates in one turn (the model tweaked the config and
        # re-validated): keep the LAST one, matching the client's #459 last-valid-wins
        # backward-walk. The gate suppresses the rest, so it must agree with the client
        # or it would hide the card #459 would have picked.
        def _ok(call_id: str) -> dict[str, Any]:
            return {
                "type": "tool",
                "tool": "mcp_amortized__validate_sdg_job",
                "callID": call_id,
                "state": {"status": "completed", "input": {"mode": "preview"}},
            }

        state = _fresh_state()
        out = agent._enforce_single_interaction({"parts": [_ok("ok-1"), _ok("ok-2")]}, state)
        kept = [p.get("callID") for p in out["parts"] if p.get("type") == "tool"]
        assert kept == ["ok-2"]  # last valid wins
        assert "ok-1" in state.suppressed_call_ids
        assert "ok-2" not in state.suppressed_call_ids

    def test_confirm_turn_drops_repeat_confirm_card(self) -> None:
        result = {
            "parts": [
                _text_part("confirm below"),
                _tool_part("validate_training_job", {"algorithm": "osft"}),
                _tool_part("present_options", {"question": "ok?"}),
                _tool_part("validate_training_job", {"algorithm": "osft"}),
            ]
        }
        out = agent._enforce_single_interaction(result, self._training_ready())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools == ["validate_training_job"]

    def test_show_prompt_gate_drops_confirm_card_stacked_under_the_prompt(self) -> None:
        # The exact observed batch: show the assessor prompt, then stack the SDG
        # confirmation (repeated) under it in the same turn. The prompt review wins;
        # every confirm card is dropped so the prompt stands alone for review.
        result = {
            "parts": [
                _tool_part("signal_phase", {"phase": "sdg", "step": "confirm"}),
                _tool_part("show_prompt", {"title": "Assessor system prompt"}),
                _tool_part("get_model_pricing", {}),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
            ]
        }
        state = _fresh_state()
        out = agent._enforce_single_interaction(result, state)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert "validate_sdg_job" not in tools  # job confirm deferred to next turn
        assert tools == ["signal_phase", "show_prompt", "get_model_pricing"]
        # [3] the prompt-gate hold parks a model-directed review steer (not user text).
        assert "review" in (state.pending_interaction_steer or "").lower()
        assert not any(p.get("type") == "text" for p in out["parts"])

    def test_show_prompt_with_review_question_keeps_the_question(self) -> None:
        # The CORRECT flow: show the prompt, then ask approve/edit. This is an ask
        # turn — the present_options leads and the prompt rides along as passive
        # context. The gate must NOT fire here.
        result = {
            "parts": [
                _tool_part("show_prompt", {"title": "Assessor system prompt"}),
                _tool_part("present_options", {"question": "Approve this prompt?"}),
            ]
        }
        out = agent._enforce_single_interaction(result, _fresh_state())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools == ["show_prompt", "present_options"]

    def test_review_gate_does_not_livelock_on_a_reviewed_prompt(self) -> None:
        # First turn gates (prompt unseen); after the user has seen it, a driver that
        # re-shows the SAME prompt before validate must be able to confirm — the
        # confirm proceeds and the now-redundant review card is dropped instead.
        state = _fresh_state()
        prompt = {"title": "Assessor system prompt", "prompt": "You are a reviewer."}
        turn1 = {
            "parts": [
                _tool_part("show_prompt", dict(prompt)),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
            ]
        }
        out1 = agent._enforce_single_interaction(turn1, state)
        t1 = [agent._tool_name(p) for p in out1["parts"] if p.get("type") == "tool"]
        assert t1 == ["show_prompt"]  # first time: confirm deferred, prompt reviewed

        turn2 = {
            "parts": [
                _tool_part("show_prompt", dict(prompt)),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
            ]
        }
        out2 = agent._enforce_single_interaction(turn2, state)
        t2 = [agent._tool_name(p) for p in out2["parts"] if p.get("type") == "tool"]
        assert t2 == ["validate_sdg_job"]  # reviewed: confirm proceeds, re-show dropped

    def test_edited_prompt_re_gates(self) -> None:
        # An edited prompt is a new artifact (new signature) and must be reviewed
        # again, even though an earlier version was already shown.
        state = _fresh_state()
        state.reviewed_card_sigs.add(
            agent._review_card_signature(_tool_part("show_prompt", {"prompt": "v1"}))
        )
        turn = {
            "parts": [
                _tool_part("show_prompt", {"prompt": "v2 edited"}),
                _tool_part("validate_sdg_job", {"mode": "preview"}),
            ]
        }
        out = agent._enforce_single_interaction(turn, state)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools == ["show_prompt"]  # new content re-gates the confirm

    def test_noop_without_any_interaction(self) -> None:
        result = {"parts": [_text_part("working"), _tool_part("list_models", {})]}
        out = agent._enforce_single_interaction(result, _fresh_state())
        assert out["parts"] == result["parts"]
        assert out is result  # unchanged object when nothing to cut

    def test_noop_when_present_options_is_last_and_alone(self) -> None:
        result = {
            "parts": [
                _text_part("pick one"),
                _tool_part("present_options", {"question": "which?"}),
            ]
        }
        out = agent._enforce_single_interaction(result, _fresh_state())
        assert len(out["parts"]) == 2


def _tools(out: dict[str, Any]) -> list[str]:
    return [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]


def _nudge(state: Any) -> str:
    # The interaction gate parks its model-directed steer on the session (never in the
    # user-facing result): a HARD re-run steer for a held committing card, or a SOFT
    # next-turn steer for a prompt-gate hold.
    return state.pending_rerun_steer or state.pending_interaction_steer or ""


def test_held_card_steer_is_delivered_to_the_model_next_turn_then_cleared() -> None:
    # [3] The gate parks a MODEL-directed steer on the session; the next message to that
    # session leads with it (so the model does the missing step itself), then it clears —
    # the steer is never shown to the user.
    state = _fresh_state()
    state.pending_interaction_steer = "Show the SDG preview before the full run."
    led = agent._consume_interaction_steer(state, "yes, go")
    assert led.startswith("Show the SDG preview before the full run.")
    assert "yes, go" in led
    assert state.pending_interaction_steer is None
    assert agent._consume_interaction_steer(state, "next") == "next"  # nothing parked


class TestInteractionSteerDelivery:
    """The gate parks a model-directed steer on the session; the next handler turn leads
    the model's message with it — except a job_complete event (which replaces user_text),
    where the steer stays parked for the next real turn (R2-A)."""

    def test_subagent_turn_leads_with_the_parked_steer(self, router: _Router) -> None:
        sub = router.add_session([[_text_part("ok")]])
        state = _make_state(router)
        state.subagent_id = sub
        state.subagent_target = "sdg"
        state.pending_interaction_steer = "Show the preview first."
        _run(agent._handle_subagent_message(state, "s1", "go ahead", _body(text="go ahead")))
        sent = next(t for (sid, t, _a) in router.sent if sid == sub)
        assert sent.startswith("Show the preview first.")
        assert "go ahead" in sent
        assert state.pending_interaction_steer is None  # consumed

    def test_job_complete_keeps_the_steer_parked(self, router: _Router) -> None:
        sub = router.add_session([[_text_part("done")]])
        state = _make_state(router)
        state.subagent_id = sub
        state.subagent_target = "sdg"
        state.pending_interaction_steer = "Show the preview first."
        _run(
            agent._handle_subagent_message(
                state, "s1", "ignored",
                _body(text="ignored", event="job_complete", outcome="succeeded"),
            )
        )
        sent = next(t for (sid, t, _a) in router.sent if sid == sub)
        assert agent._SUBAGENT_JOB_DONE_PROMPT in sent  # job-done prompt forwarded
        assert "Show the preview first." not in sent  # steer NOT consumed on job_complete
        assert state.pending_interaction_steer == "Show the preview first."  # stays parked


class TestCommittingCardReRun:
    """A held committing card (unmet precondition) triggers an in-turn model re-run — the
    hard rail — so the premature card never reaches the client (it falls behind the steer
    message). Validated on the training data precondition; SDG previewing is a soft prompt
    default, not a gate, so it is deliberately NOT re-run."""

    def test_unmet_precondition_reruns_the_model(self, router: _Router) -> None:
        # Turn 1: the subagent jumps to a training confirm with no data/model/VRAM shown ->
        # the precondition gate holds it. The re-run then has the model do the setup.
        sub = router.add_session(
            [
                [_tool_part("validate_training_job", {"algorithm": "osft"})],
                [_tool_part("get_dataset", {"run_id": "r1"})],  # re-run: do setup instead
            ]
        )
        state = _make_state(router)
        state.subagent_id = sub
        state.subagent_target = "training"
        tid = "turn-1"
        state.turns[tid] = agent.TurnState()
        _run(agent._run_turn(state, "s1", tid, "train it", _body(text="train it")))
        sub_sends = [t for (sid, t, _a) in router.sent if sid == sub]
        assert sub_sends[0] == "train it"  # the original turn
        assert len(sub_sends) == 2  # the gate re-ran the model once (the hard rail)
        assert sub_sends[1] != "train it"  # the re-run posted the data-precondition steer
        assert state.pending_rerun_steer is None  # consumed, not carried over

    def test_no_rerun_when_preconditions_met(self, router: _Router) -> None:
        # All training preconditions satisfied -> the confirm is NOT held and NOT re-run.
        sub = router.add_session(
            [[_tool_part("validate_training_job", {"algorithm": "osft"})]]
        )
        state = _make_state(router)
        state.subagent_id = sub
        state.subagent_target = "training"
        state.confirmed_data = True
        state.offered_choice.add("training")
        state.shown_vram = True
        tid = "turn-1"
        state.turns[tid] = agent.TurnState()
        _run(agent._run_turn(state, "s1", tid, "train it", _body(text="train it")))
        sub_sends = [t for (sid, t, _a) in router.sent if sid == sub]
        assert sub_sends == ["train it"]  # no re-run
        assert state.pending_rerun_steer is None

    def test_rerun_exhaustion_defers_the_steer(self, router: _Router) -> None:
        # The driver re-emits the premature card on the re-run too, so the hard rail
        # exhausts (MAX_INTERACTION_RERUNS). The steer must NOT be silently dropped —
        # it is deferred as a soft steer so the model is corrected on its next turn
        # (otherwise the card is held yet the model never learns why — the dead end).
        sub = router.add_session(
            [
                [_tool_part("validate_training_job", {"algorithm": "osft"})],  # turn 1: held
                [_tool_part("validate_training_job", {"algorithm": "osft"})],  # re-run: re-emitted
            ]
        )
        state = _make_state(router)
        state.subagent_id = sub
        state.subagent_target = "training"
        tid = "turn-1"
        state.turns[tid] = agent.TurnState()
        _run(agent._run_turn(state, "s1", tid, "train it", _body(text="train it")))
        sub_sends = [t for (sid, t, _a) in router.sent if sid == sub]
        assert len(sub_sends) == 2  # original + exactly one re-run
        assert state.pending_rerun_steer is None  # hard steer cleared
        # The fix: deferred, not lost — the model receives the correction next turn.
        assert state.pending_interaction_steer
        assert "data" in state.pending_interaction_steer.lower()


class TestTrainingConfirmPreconditions:
    """A validate_training_job card may render only after the user has (in order)
    confirmed the data, been offered a model/method choice, and seen the VRAM estimate —
    the training arm of the confirm-card precondition evaluator. Signals accumulate as
    turns render them and reset once the job card renders."""

    def _confirm(self) -> dict[str, Any]:
        return {"parts": [_tool_part("validate_training_job", {"algorithm": "osft"})]}

    def test_held_until_data_confirmed_first(self) -> None:
        state = _fresh_state()
        out = agent._enforce_single_interaction(self._confirm(), state)
        assert "validate_training_job" not in _tools(out)
        assert "get_dataset" in _nudge(state).lower()  # first unmet precondition

    def test_get_dataset_satisfies_the_data_precondition(self) -> None:
        state = _fresh_state()
        state.subagent_target = "training"
        # A prior setup turn inspects the dataset (passive tool, no card/question).
        agent._enforce_single_interaction(
            {"parts": [_tool_part("get_dataset", {"run_id": "r1"})]}, state
        )
        assert state.confirmed_data is True
        # Data done; the next unmet precondition is the model choice.
        out = agent._enforce_single_interaction(self._confirm(), state)
        assert "validate_training_job" not in _tools(out)
        assert "model" in _nudge(state).lower()

    def test_model_choice_recorded_only_in_training_context(self) -> None:
        state = _fresh_state()
        state.subagent_target = "training"
        agent._enforce_single_interaction(
            {"parts": [_tool_part("present_options", {"question": "Which base model?"})]},
            state,
        )
        assert "training" in state.offered_choice

    def test_data_and_vram_recorded_only_in_training_context(self) -> None:
        # get_dataset / estimate_training_resources fired inside an SDG or eval subagent
        # must NOT pre-satisfy a later training confirm — the signals gate training alone.
        state = _fresh_state()
        state.subagent_target = "sdg"
        agent._enforce_single_interaction(
            {
                "parts": [
                    _tool_part("get_dataset", {"run_id": "r1"}),
                    _tool_part("estimate_training_resources", {"model": "m"}),
                ]
            },
            state,
        )
        assert state.confirmed_data is False
        assert state.shown_vram is False

    def test_renders_once_all_preconditions_met(self) -> None:
        state = _fresh_state()
        state.subagent_target = "training"
        # data (prior turn) + model choice (prior turn) + VRAM (this turn).
        agent._enforce_single_interaction(
            {"parts": [_tool_part("get_dataset", {"run_id": "r1"})]}, state
        )
        agent._enforce_single_interaction(
            {"parts": [_tool_part("present_options", {"question": "Which model?"})]}, state
        )
        result = {
            "parts": [
                _tool_part("estimate_training_resources", {"model": "m"}),
                _tool_part("validate_training_job", {"algorithm": "osft"}),
            ]
        }
        out = agent._enforce_single_interaction(result, state)
        assert "validate_training_job" in _tools(out)

    def test_second_job_re_gates_after_confirm(self) -> None:
        state = _fresh_state()
        state.shown_vram = True
        state.confirmed_data = True
        state.offered_choice.add("training")
        out1 = agent._enforce_single_interaction(self._confirm(), state)
        assert "validate_training_job" in _tools(out1)  # first job passes
        # The signals were cleared, so a second training job must re-establish them.
        assert state.shown_vram is False
        assert "training" not in state.offered_choice
        out2 = agent._enforce_single_interaction(self._confirm(), state)
        assert "validate_training_job" not in _tools(out2)


class TestEvalConfirmPreconditions:
    """A validate_eval_job card may render only after the user has been offered the
    evaluation setup (the judge model has no default) — the eval arm of the evaluator."""

    def _confirm(self) -> dict[str, Any]:
        return {"parts": [_tool_part("validate_eval_job", {"training_job_id": "t1"})]}

    def test_held_until_a_choice_was_offered(self) -> None:
        state = _fresh_state()
        out = agent._enforce_single_interaction(self._confirm(), state)
        assert "validate_eval_job" not in _tools(out)
        assert "judge" in _nudge(state).lower()

    def test_renders_after_eval_choice_offered(self) -> None:
        state = _fresh_state()
        state.subagent_target = "eval"
        agent._enforce_single_interaction(
            {"parts": [_tool_part("present_options", {"question": "Which judge model?"})]},
            state,
        )
        out = agent._enforce_single_interaction(self._confirm(), state)
        assert "validate_eval_job" in _tools(out)

    def test_training_choice_does_not_satisfy_eval(self) -> None:
        state = _fresh_state()
        state.offered_choice.add("training")  # a training choice is not an eval choice
        out = agent._enforce_single_interaction(self._confirm(), state)
        assert "validate_eval_job" not in _tools(out)


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
        sent = [text for sid, text, _ag in router.sent if sid == sub_id]
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
