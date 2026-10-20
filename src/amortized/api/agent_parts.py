"""Tool-part primitives shared by the agent proxy and its submodules.

These parse the opencode message `parts` shape and are the one low-level dependency the
`agent_sdg_prompt` submodule shares. They live in this leaf module so that submodule
need not import the heavyweight `agent` module (which would be a cycle).
"""

from __future__ import annotations

from typing import Any


def _tool_name(part: dict[str, Any]) -> str:
    raw = part.get("tool") or part.get("toolName") or ""
    if "__" in raw:
        return raw.split("__")[-1]
    if raw.startswith("amortized_"):
        return raw[len("amortized_") :]
    return raw


def _get_tool_input(part: dict[str, Any]) -> dict[str, Any]:
    inp = part.get("input")
    if isinstance(inp, dict):
        return inp
    state = part.get("state")
    if isinstance(state, dict):
        state_input = state.get("input")
        if isinstance(state_input, dict):
            return state_input
    return {}


def _part_errored(part: dict[str, Any]) -> bool:
    """True when a tool-call part is in opencode's ``error`` state.

    An errored call renders no interactive card/question, so the interaction gate must
    not treat it as a turn's leading interaction: when the model fixes a bad tool call
    and re-calls in the same turn (e.g. a validate_* that errored, then succeeded), the
    dead errored part must not shadow the live one — otherwise the gate keeps the error
    and strips the successful card the user is meant to confirm.
    """
    state = part.get("state")
    return isinstance(state, dict) and state.get("status") == "error"
