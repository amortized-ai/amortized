"""Tests for the present_options intra-call dedup guard."""

import asyncio
from typing import Any

from amortized.api.ui import (
    OptionItem,
    PresentOptionsRequest,
    ShowPromptRequest,
    SignalPhaseRequest,
    _dedup_options,
    present_options,
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
