"""Tests for monitor secret-masking / size-bounding (`monitor_redact.redact`)."""

from __future__ import annotations

from amortized.api.monitor_redact import (
    _MAX_DEPTH,
    _MAX_ITEMS,
    _MAX_STR,
    _REDACTED,
    redact,
)


def test_masks_secret_keys() -> None:
    out = redact(
        {
            "api_key": "sk-123",
            "apiKey": "sk-456",
            "base_url": "https://host",
            "password": "p",
            "authorization": "Bearer x",
            "access_key": "AKIA",
            "private_key": "-----BEGIN",
            "model": "gpt-oss",
        }
    )
    assert out["api_key"] == _REDACTED
    assert out["apiKey"] == _REDACTED
    assert out["password"] == _REDACTED
    assert out["authorization"] == _REDACTED
    assert out["access_key"] == _REDACTED
    assert out["private_key"] == _REDACTED
    # non-secret fields are preserved
    assert out["base_url"] == "https://host"
    assert out["model"] == "gpt-oss"


def test_does_not_overmatch_author() -> None:
    # bare "auth" must not swallow legitimate fields
    out = redact({"author": "alice", "authority": "high"})
    assert out == {"author": "alice", "authority": "high"}


def test_masks_nested_secret() -> None:
    out = redact({"judge": {"model": "gpt-oss", "api_key": "secret"}})
    assert out["judge"]["model"] == "gpt-oss"
    assert out["judge"]["api_key"] == _REDACTED


def test_truncates_long_strings() -> None:
    long = "x" * (_MAX_STR + 500)
    out = redact({"blob": long})
    assert out["blob"].startswith("x" * 10)
    assert out["blob"].endswith("chars]")
    assert len(out["blob"]) < len(long)


def test_depth_bound() -> None:
    # build nesting deeper than the cap
    obj: dict = {}
    cur = obj
    for _ in range(_MAX_DEPTH + 3):
        cur["next"] = {}
        cur = cur["next"]
    out = redact(obj)
    # walk down until we hit the truncation sentinel
    node = out
    depth = 0
    while isinstance(node, dict) and "next" in node:
        node = node["next"]
        depth += 1
    assert node == "[…]"
    assert depth <= _MAX_DEPTH


def test_list_breadth_bound() -> None:
    out = redact({"items": list(range(_MAX_ITEMS + 50))})
    items = out["items"]
    assert len(items) == _MAX_ITEMS + 1  # capped items + overflow marker
    assert "items]" in items[-1]


def test_never_raises_on_weird_objects() -> None:
    class Boom:
        def __str__(self) -> str:
            raise RuntimeError("no str")

    # must not propagate
    assert redact({"x": Boom()}) == {"x": _REDACTED}


def test_passthrough_scalars() -> None:
    assert redact(5) == 5
    assert redact(True) is True
    assert redact(None) is None
    assert redact("short") == "short"
