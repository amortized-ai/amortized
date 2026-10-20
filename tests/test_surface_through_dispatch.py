"""_surface_through_dispatch truncates a turn at the first terminal dispatch ack.

opencode runs the whole tool loop, so after a dispatch ack (delegate_to_subagent /
signal_subagent_completion / request_orchestrator_help returns {"status":"ok"}) a weak
driver keeps generating and narrates work that hasn't happened ("I've kicked off…",
"it's underway"). Replaying the real eval/orchestrator turns showed this follows the
help ack 1/1 and the sdg delegate ack 3/8. The proxy must surface only up to that tool
— content before it is real, the tail is speculative — so the false claim never reaches
the user. Detection reads the RAW parts, so the dispatch is still acted on.
"""

from typing import Any

from amortized.api import agent


def _text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def _tool(name: str) -> dict[str, Any]:
    # Real opencode parts name tools "amortized_<name>"; _tool_name normalizes.
    return {"type": "tool", "tool": f"amortized_{name}"}


def _texts(parts: list[dict[str, Any]]) -> list[str]:
    return [p["text"] for p in parts if p.get("type") == "text"]


def test_drops_narration_after_help() -> None:
    """The deployed eval failure verbatim: a leading step, the help ack, then the false
    'I've kicked off…' tail — only the leading content survives."""
    parts = [
        _text("Setting up the eval."),
        _tool("signal_phase"),  # not a dispatch ack — kept
        _tool("request_orchestrator_help"),
        _text("I've kicked off the process… I'll confirm once ready."),
    ]
    out = agent._surface_through_dispatch(parts)
    assert _texts(out) == ["Setting up the eval."]
    assert all(agent._tool_name(p) != "request_orchestrator_help" for p in out)


def test_drops_underway_after_delegate() -> None:
    """The orchestrator fulfill_help case: delegate to sdg, then 'it's underway' — the
    delegate is detected on raw parts elsewhere; here only the tail is dropped."""
    parts = [
        _tool("delegate_to_subagent"),
        _text("The fresh eval dataset build is underway."),
    ]
    assert agent._surface_through_dispatch(parts) == []


def test_keeps_signal_phase_and_cards_before_dispatch() -> None:
    """signal_phase is internal-but-not-terminal: everything up to the first real dispatch
    ack is preserved, including a validate_* card the gate may need to hold."""
    parts = [
        _tool("signal_phase"),
        _tool("validate_training_job"),
        _text("Ready to submit?"),
    ]
    out = agent._surface_through_dispatch(parts)
    assert [agent._tool_name(p) for p in out if p.get("type") == "tool"] == [
        "signal_phase",
        "validate_training_job",
    ]
    assert _texts(out) == ["Ready to submit?"]


def test_no_dispatch_is_identity() -> None:
    """A normal turn with no terminal tool is passed through unchanged."""
    parts = [_text("Here are your jobs."), _tool("list_jobs")]
    assert agent._surface_through_dispatch(parts) == parts


def test_completion_ack_also_truncates() -> None:
    parts = [_tool("signal_subagent_completion"), _text("All done, moving on!")]
    assert agent._surface_through_dispatch(parts) == []
