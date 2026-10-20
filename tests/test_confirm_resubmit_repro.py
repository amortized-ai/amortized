"""Regression for the resubmit hold (session 806dac44): a same-config training resubmit was
delegated FRESH (resume=False), so the old _reset_confirm_signals wiped model/vram/data
signals the user had already satisfied this session; the next validate_training_job was then
HELD on an unmet precondition and never rendered until the user prodded ("where is confirm?").

The fix makes re-gating config-driven: each gate stamps a fingerprint of the fields it covers
when its card renders, and re-gates on the next card only if that fingerprint changed. So a
same-config resubmit renders with no hold, while a changed base model / data / VRAM param
re-gates exactly the affected prerequisite.
"""

from typing import Any

from amortized.api import agent

# A training config; the resubmit drops the broken max_tokens_per_gpu but keeps model/data
# and every VRAM-relevant field, so every gate's fingerprint is unchanged.
_CONFIG: dict[str, Any] = {
    "algorithm": "osft",
    "model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "parent_job_id": "sdg-9f36627d",
    "unfreeze_rank_ratio": 0.2,
    "per_device_train_batch_size": 4,
    "max_length": 2048,
}


def _card(config: dict[str, Any], call_id: str = "card") -> dict[str, Any]:
    return {
        "type": "tool",
        "tool": "mcp_amortized__validate_training_job",
        "input": config,
        "callID": call_id,
    }


def _confirmed_training_state() -> agent.SessionState:
    # The user already chose the model, saw VRAM, and confirmed the data earlier this session.
    state = agent.SessionState(orchestrator_id="orch")
    state.subagent_target = "training"
    state.shown_vram = True
    state.confirmed_data = True
    state.offered_choice.add("training")
    return state


def _kept_tools(result: dict[str, Any]) -> list[str]:
    return [agent._tool_name(p) for p in result["parts"] if p.get("type") == "tool"]


def _render(state: agent.SessionState, config: dict[str, Any], call_id: str) -> dict[str, Any]:
    return agent._enforce_single_interaction({"parts": [_card(config, call_id)]}, state)


def test_first_confirm_renders_when_prereqs_met() -> None:
    """Baseline: with signals earned and no prior fingerprint, the first card leads and is kept."""
    state = _confirmed_training_state()
    out = _render(state, _CONFIG, "card1")
    assert "validate_training_job" in _kept_tools(out)
    assert state.pending_rerun_steer is None
    # The render stamped a fingerprint for every training gate (keyed by the bare tool name).
    assert ("validate_training_job", "model") in state.gate_fp


def test_same_config_resubmit_renders_with_no_hold() -> None:
    """THE FIX: after the card renders, an identical-config resubmit matches every fingerprint
    and renders again — no reset, no hold, no dependence on a user nudge."""
    state = _confirmed_training_state()
    _render(state, _CONFIG, "card1")
    out = _render(state, dict(_CONFIG), "card2")
    assert "validate_training_job" in _kept_tools(out)
    assert state.pending_rerun_steer is None


def test_dropping_an_untracked_field_still_renders() -> None:
    """The deployed case verbatim: the resubmit drops max_tokens_per_gpu (no gate covers it),
    so every fingerprint is unchanged and the card renders."""
    state = _confirmed_training_state()
    _render(state, {**_CONFIG, "max_tokens_per_gpu": 8192}, "card1")
    out = _render(state, _CONFIG, "card2")  # broken field dropped
    assert "validate_training_job" in _kept_tools(out)


def test_changed_base_model_re_gates_model_and_vram() -> None:
    """A genuinely new job (different base model off the same dataset) re-gates: the model and
    VRAM fingerprints changed, so the card is held while data stays confirmed."""
    state = _confirmed_training_state()
    _render(state, _CONFIG, "card1")
    out = _render(state, {**_CONFIG, "model_name_or_path": "meta-llama/Llama-3.1-8B"}, "card2")
    assert "validate_training_job" not in _kept_tools(out)  # held
    assert state.pending_rerun_steer is not None
    assert "model" in state.pending_rerun_steer.lower()  # the model gate fired first


def test_changed_vram_param_re_gates_only_vram() -> None:
    """Bumping a VRAM-relevant hyperparameter (batch size) re-gates VRAM alone — model and
    data fingerprints are unchanged, so the first unmet gate is vram."""
    state = _confirmed_training_state()
    _render(state, _CONFIG, "card1")
    out = _render(state, {**_CONFIG, "per_device_train_batch_size": 16}, "card2")
    assert "validate_training_job" not in _kept_tools(out)
    assert "vram" in state.pending_rerun_steer.lower()


def test_changed_parent_re_gates_data() -> None:
    """A different dataset re-gates the data prerequisite."""
    state = _confirmed_training_state()
    _render(state, _CONFIG, "card1")
    out = _render(state, {**_CONFIG, "parent_job_id": "sdg-other"}, "card2")
    assert "validate_training_job" not in _kept_tools(out)
    assert "data" in state.pending_rerun_steer.lower()
