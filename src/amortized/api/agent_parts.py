"""Tool-part primitives shared by the agent proxy and its gate submodules.

These parse the opencode message `parts` shape and are the one low-level dependency the
gate submodules (agent_confirm, agent_sdg_prompt) share. They live in this leaf module so
those submodules need not import the heavyweight `agent` module (which would be a cycle).
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
