"""Tests for the agent proxy subagent delegation routing.

Covers subagent → subagent delegation (e.g. eval → sdg) and the
stack-based return of control when the delegated subagent completes.
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


class TestSubagentDelegation:
    def test_subagent_delegation_switches_sessions(self, router: _Router) -> None:
        # Active eval subagent delegates to sdg
        sdg_id = router.queue_session([[_text_part("What kind of dataset?")]])
        eval_id = router.add_session(
            [[_tool_part("delegate_to_subagent", {"target": "sdg", "context": "c"})]]
        )
        state = _make_state(router)
        state.subagent_id = eval_id
        state.subagent_target = "eval"

        result = _run(
            agent._handle_subagent_message(state, "s1", "build the dataset", _body())
        )

        # The new sdg session became active; eval was stashed on the stack
        assert state.subagent_target == "sdg"
        assert state.subagent_id == sdg_id
        assert len(state.subagent_stack) == 1
        stashed_target, stashed_id = state.subagent_stack[0]
        assert stashed_target == "eval"
        assert stashed_id == eval_id  # the original eval session
        # Response contains the user-visible parts from the sdg agent
        texts = [p.get("text") for p in result["parts"] if p.get("type") == "text"]
        assert "What kind of dataset?" in texts

    def test_subagent_completion_returns_to_parent_not_orchestrator(self, router: _Router) -> None:
        # sdg (child) completes; eval (parent, on the stack) resumes
        eval_id = router.add_session([[_text_part("Dataset ready — continuing eval setup")]])
        sdg_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "SDG job 123 done"})]]
        )
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"
        state.subagent_stack = [("eval", eval_id)]

        result = _run(
            agent._handle_subagent_message(state, "s1", "confirm", _body())
        )

        # Control returned to the eval agent
        assert state.subagent_id == eval_id
        assert state.subagent_target == "eval"
        assert state.subagent_stack == []
        # The orchestrator was never contacted
        assert all(s != state.orchestrator_id for s, _, _ in router.sent)
        # Parent received the completion summary
        parent_msg = next(m for m in router.sent if m[0] == eval_id)
        assert "[SUBAGENT COMPLETED]" in parent_msg[1]
        assert "SDG job 123 done" in parent_msg[1]
        # User-visible parts include both the sdg tail and the eval response
        texts = [p.get("text") for p in result["parts"] if p.get("type") == "text"]
        assert "Dataset ready — continuing eval setup" in texts

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

    def test_parent_may_delegate_again_after_child_completes(self, router: _Router) -> None:
        # eval resumes after sdg completes and immediately delegates to training
        training_id = router.queue_session([[_text_part("Which model?")]])
        eval_id = router.add_session(
            [[_tool_part("delegate_to_subagent", {"target": "training", "context": "c2"})]]
        )
        sdg_id = router.add_session(
            [[_tool_part("signal_subagent_completion", {"summary": "sdg done"})]]
        )
        state = _make_state(router)
        state.subagent_id = sdg_id
        state.subagent_target = "sdg"
        state.subagent_stack = [("eval", eval_id)]

        _run(agent._handle_subagent_message(state, "s1", "confirm", _body()))

        assert state.subagent_target == "training"
        assert state.subagent_id == training_id
        assert state.subagent_stack == [("eval", eval_id)]

    def test_internal_tools_stripped_from_response(self, router: _Router) -> None:
        _ = router.queue_session([[_text_part("What kind of dataset?")]])
        state = _make_state(router)
        state.subagent_id = router.add_session(
            [[_tool_part("delegate_to_subagent", {"target": "sdg", "context": "c"})]]
        )
        state.subagent_target = "eval"

        result = _run(
            agent._handle_subagent_message(state, "s1", "build it", _body())
        )
        tools = [
            p for p in result["parts"] if p.get("type") == "tool"
        ]
        assert tools == []


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
        assert forwarded == agent._SUBAGENT_JOB_DONE_PROMPT
        assert "signal_subagent_completion" in forwarded

    def test_failed_job_not_rewritten(self, router: _Router) -> None:
        state = self._active_sdg(router)
        text = (
            "Job d6be24ce (sdg) finished with status: failed."
            " Use present_options to suggest next steps to the user."
        )
        _run(agent._handle_subagent_message(state, "s1", text, _body(event="job_complete")))
        assert router.sent[0][1] == text  # untouched — subagent handles the failure

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
        out = agent._enforce_single_interaction(result, _fresh_state())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert "validate_sdg_job" not in tools  # job confirm deferred to next turn
        assert tools == ["signal_phase", "show_prompt", "get_model_pricing"]

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


class TestPreviewBeforeCreateGate:
    """A full-job SDG confirm (validate_sdg_job mode='create') may render only after a
    preview of the SAME recipe has been shown standalone for approval. This stops a
    weak driver from skipping the mandated ~10-sample preview and jumping straight to
    the full run, and re-gates when the recipe is edited after a preview."""

    _RECIPE: ClassVar[dict[str, Any]] = {"columns": [{"name": "q"}], "topic": "openshift"}

    def _create(self, **extra: Any) -> dict[str, Any]:
        return _tool_part("validate_sdg_job", {**self._RECIPE, "mode": "create", **extra})

    def _preview(self, **extra: Any) -> dict[str, Any]:
        return _tool_part(
            "validate_sdg_job", {**self._RECIPE, "mode": "preview", **extra}
        )

    def test_create_without_prior_preview_is_held_and_nudged(self) -> None:
        result = {"parts": [_text_part("confirm the full run"), self._create()]}
        out = agent._enforce_single_interaction(result, _fresh_state())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert "validate_sdg_job" not in tools  # full-job confirm held
        texts = [p["text"] for p in out["parts"] if p.get("type") == "text"]
        assert any("preview" in t.lower() for t in texts)  # steered to preview first

    def test_create_with_omitted_mode_defaults_to_create_and_is_gated(self) -> None:
        # SDGJobRequest.mode defaults to 'create', so an omitted mode is a full run.
        result = {"parts": [_tool_part("validate_sdg_job", dict(self._RECIPE))]}
        out = agent._enforce_single_interaction(result, _fresh_state())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert "validate_sdg_job" not in tools

    def test_preview_then_create_same_turn_keeps_preview(self) -> None:
        # The driver stacks the full run under the preview. The preview leads; the
        # create is dropped and must come on a later turn after approval.
        state = _fresh_state()
        result = {"parts": [self._preview(), self._create()]}
        out = agent._enforce_single_interaction(result, state)
        modes = [
            agent._validate_mode(p)
            for p in out["parts"]
            if agent._tool_name(p) == "validate_sdg_job"
        ]
        assert modes == ["preview"]  # only the preview rendered
        # The preview is now approved, so a later create proceeds.
        out2 = agent._enforce_single_interaction({"parts": [self._create()]}, state)
        assert any(
            agent._tool_name(p) == "validate_sdg_job" for p in out2["parts"]
        )

    def test_create_after_standalone_preview_proceeds(self) -> None:
        state = _fresh_state()
        agent._enforce_single_interaction({"parts": [self._preview()]}, state)
        out = agent._enforce_single_interaction({"parts": [self._create()]}, state)
        assert any(agent._tool_name(p) == "validate_sdg_job" for p in out["parts"])

    def test_record_count_change_does_not_re_gate(self) -> None:
        # num_records counts samples, not recipe — a preview of 10 approves the full run.
        state = _fresh_state()
        agent._enforce_single_interaction(
            {"parts": [self._preview(num_records=10)]}, state
        )
        out = agent._enforce_single_interaction(
            {"parts": [self._create(num_records=500)]}, state
        )
        assert any(agent._tool_name(p) == "validate_sdg_job" for p in out["parts"])

    def test_edited_recipe_re_gates_the_create(self) -> None:
        state = _fresh_state()
        agent._enforce_single_interaction({"parts": [self._preview()]}, state)
        edited = _tool_part(
            "validate_sdg_job",
            {"columns": [{"name": "q"}], "topic": "kubernetes", "mode": "create"},
        )
        out = agent._enforce_single_interaction({"parts": [edited]}, state)
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert "validate_sdg_job" not in tools  # new recipe needs a fresh preview

    def test_prompt_gate_precedes_the_preview_gate(self) -> None:
        # With a new prompt AND an ungated create, the prompt review wins first.
        result = {
            "parts": [
                _tool_part("show_prompt", {"prompt": "You are an assessor."}),
                self._create(),
            ]
        }
        out = agent._enforce_single_interaction(result, _fresh_state())
        tools = [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]
        assert tools == ["show_prompt"]  # prompt stands alone; create held

    def test_training_confirm_is_not_preview_gated(self) -> None:
        # The preview gate is SDG-only: a training confirm is never held with the SDG
        # preview nudge (it has its own precondition gates, covered separately).
        result = {"parts": [_tool_part("validate_training_job", {"algorithm": "sft"})]}
        out = agent._enforce_single_interaction(result, _fresh_state())
        texts = [p.get("text", "") for p in out["parts"] if p.get("type") == "text"]
        assert not any("preview" in t.lower() for t in texts)


def _tools(out: dict[str, Any]) -> list[str]:
    return [agent._tool_name(p) for p in out["parts"] if p.get("type") == "tool"]


def _nudge(out: dict[str, Any]) -> str:
    return " ".join(p.get("text", "") for p in out["parts"] if p.get("type") == "text")


class TestTrainingConfirmPreconditions:
    """A validate_training_job card may render only after the user has (in order)
    confirmed the data, been offered a model/method choice, and seen the VRAM estimate —
    the training arm of the confirm-card precondition evaluator. Signals accumulate as
    turns render them and reset once the job card renders."""

    def _confirm(self) -> dict[str, Any]:
        return {"parts": [_tool_part("validate_training_job", {"algorithm": "osft"})]}

    def test_held_until_data_confirmed_first(self) -> None:
        out = agent._enforce_single_interaction(self._confirm(), _fresh_state())
        assert "validate_training_job" not in _tools(out)
        assert "get_dataset" in _nudge(out).lower()  # first unmet precondition

    def test_get_dataset_satisfies_the_data_precondition(self) -> None:
        state = _fresh_state()
        # A prior setup turn inspects the dataset (passive tool, no card/question).
        agent._enforce_single_interaction(
            {"parts": [_tool_part("get_dataset", {"run_id": "r1"})]}, state
        )
        assert state.confirmed_data is True
        # Data done; the next unmet precondition is the model choice.
        out = agent._enforce_single_interaction(self._confirm(), state)
        assert "validate_training_job" not in _tools(out)
        assert "model" in _nudge(out).lower()

    def test_model_choice_recorded_only_in_training_context(self) -> None:
        state = _fresh_state()
        state.subagent_target = "training"
        agent._enforce_single_interaction(
            {"parts": [_tool_part("present_options", {"question": "Which base model?"})]},
            state,
        )
        assert "training" in state.offered_choice

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
        out = agent._enforce_single_interaction(self._confirm(), _fresh_state())
        assert "validate_eval_job" not in _tools(out)
        assert "judge" in _nudge(out).lower()

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
    subagent→subagent delegation, normal), so it is unit-tested here on the pure
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
