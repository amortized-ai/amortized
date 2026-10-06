"""Tests for the present_options intra-call dedup guard."""

import asyncio
from typing import Any

from amortized.api.ui import (
    ModelPricingItem,
    OptionItem,
    PresentOptionsRequest,
    ShowModelPricingRequest,
    ShowPromptRequest,
    SignalPhaseRequest,
    _dedup_options,
    present_options,
    show_model_pricing,
    show_prompt,
    signal_phase,
)


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


def _opt(title: str, value: str) -> OptionItem:
    return OptionItem(title=title, description="", value=value)


def test_duplicate_value_removed() -> None:
    opts = [_opt("Yes", "Yes, train it"), _opt("Yep", "yes, train it ")]
    out = _dedup_options(opts)
    assert len(out) == 1
    assert out[0].title == "Yes"


def test_duplicate_title_removed() -> None:
    opts = [_opt("Train", "Train on all data"), _opt("train", "Train on a subset")]
    assert len(_dedup_options(opts)) == 1


def test_distinct_options_preserved_in_order() -> None:
    opts = [_opt("A", "do a"), _opt("B", "do b"), _opt("C", "do c")]
    out = _dedup_options(opts)
    assert [o.title for o in out] == ["A", "B", "C"]


def test_endpoint_dedups() -> None:
    body = PresentOptionsRequest(
        step="s",
        question="q?",
        options=[_opt("Yes", "do it"), _opt("Yes", "do it")],
    )
    resp = _run(present_options(body))
    assert len(resp.options) == 1


def test_present_options_carries_halt_directive() -> None:
    # The tool result must tell the model to end its turn, so a weaker driver
    # can't batch more questions + a validate_* behind one present_options.
    body = PresentOptionsRequest(step="s", question="q?", options=[_opt("Yes", "do it")])
    resp = _run(present_options(body))
    assert "STOP" in resp.agent_instruction
    assert "NEXT turn" in resp.agent_instruction
    dumped = resp.model_dump()
    assert dumped["agent_instruction"] == resp.agent_instruction  # serialized to the model


def test_show_prompt_echoes_full_text() -> None:
    prompt = "You are an RFE assessor.\nScore 5 criteria, 7/10 PASS."
    resp = _run(show_prompt(ShowPromptRequest(title="Assessor prompt", prompt=prompt)))
    assert resp.rendered is True
    assert resp.prompt == prompt
    assert resp.title == "Assessor prompt"


def test_signal_phase_accepts_eval() -> None:
    resp = _run(signal_phase(SignalPhaseRequest(phase="eval", step="review")))
    assert resp.phase == "eval"
    assert resp.step == "review"


def test_show_model_pricing_coerces_formatted_numbers() -> None:
    # An agent that passes costs as formatted strings should still render the card
    # rather than 422 (observed with GLM dropping the pricing card on first try).
    body = ShowModelPricingRequest(
        models=[
            ModelPricingItem(
                model_id="openai/gpt-oss-120b",
                name="gpt-oss",
                prompt_cost_per_1m="$0.037",  # type: ignore[arg-type]
                completion_cost_per_1m="0.17/1M tokens",  # type: ignore[arg-type]
                context_length="131072",  # type: ignore[arg-type]
            )
        ]
    )
    resp = _run(show_model_pricing(body))
    assert resp.rendered is True
    item = resp.models[0]
    assert item.prompt_cost_per_1m == 0.037
    assert item.completion_cost_per_1m == 0.17
    assert item.context_length == 131072
