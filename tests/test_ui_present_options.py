"""Tests for the present_options intra-call dedup guard."""

import asyncio
from typing import Any

from amortized.api.ui import (
    OptionItem,
    PresentOptionsRequest,
    _dedup_options,
    present_options,
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
