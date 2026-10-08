"""Confirm-card precondition policy: the single declarative table (and the pure engine
that consults it) deciding whether a validate_* confirmation card may render this turn, or
must hold while a prerequisite setup step fires first.

This is the SYNC half of the stage-gating — it turns on and reads session-local signals
(preview approved, data confirmed, model/judge offered, VRAM shown). Readiness
preconditions that need the DB (an upstream job being 'succeeded') live in the async
stage-gate `_not_ready_violations` in agent.py. `_enforce_single_interaction` (agent.py)
is the consumer: it keeps whichever interaction leads a turn and holds a card whose first
unmet precondition this table names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from amortized.api.agent_parts import _get_tool_input, _tool_name

if TYPE_CHECKING:
    from amortized.api.agent import SessionState

_CONFIRM_CARD_TOOLS = {
    "validate_training_job",
    "validate_eval_job",
    "validate_sdg_job",
}

# Tools whose result renders a card the user must REVIEW/approve before the job it
# gates is confirmed (e.g. an assessor/system prompt that ships in the training
# data). Unlike purely informational show_* cards (VRAM, pricing) — which belong on
# the confirm card — a review card must get its own turn. Add future review cards
# here; the gate in _enforce_single_interaction treats the category uniformly.
_REVIEW_CARD_TOOLS = {"show_prompt"}


def _review_card_signature(part: dict[str, Any]) -> str:
    """Stable content signature of a review card, so an edited prompt re-gates but
    an unchanged re-show of an already-reviewed prompt does not."""
    inp = _get_tool_input(part)
    text = str(inp.get("prompt") or inp.get("title") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_mode(part: dict[str, Any]) -> str:
    """The `mode` of a validate_sdg_job call. SDGJobRequest.mode defaults to 'create',
    so an omitted mode is a full-job confirm, not a preview."""
    return str(_get_tool_input(part).get("mode") or "create").strip().lower()


# Fields that count a sample, not the recipe. A preview approves the RECIPE (columns,
# models, processors, seed, topic); changing only how many records to make must not
# force a re-preview. parent_job_id is lineage, likewise not recipe-defining. Mirrors
# the clone-for-eval invariant ("mirror the recipe verbatim; only num_records changes").
_SDG_NON_RECIPE_FIELDS = {"mode", "num_records", "parent_job_id"}


def _sdg_config_signature(part: dict[str, Any]) -> str:
    """Stable signature of an SDG job's RECIPE, identical for a preview and the full-job
    confirm of the same recipe (they differ only in `mode`/record count). An edited
    recipe yields a new signature, so the preview-before-create gate re-fires."""
    recipe = {
        k: v for k, v in _get_tool_input(part).items() if k not in _SDG_NON_RECIPE_FIELDS
    }
    canonical = json.dumps(recipe, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Tools that establish a confirm card's prerequisite ACTIONS, tracked as session
# signals (see SessionState). The VRAM estimate and dataset inspection are ordinary
# (passive) tool calls that must have happened before a training confirm.
_VRAM_TOOL = "estimate_training_resources"
_DATA_CONFIRM_TOOLS = {"get_dataset", "get_dataset_samples", "split_dataset"}


# Steering appended when a confirm card is HELD because a prerequisite is unmet. One per
# precondition; the first unmet one's text is what the model sees.
_PREVIEW_FIRST_NUDGE = (
    "A full SDG run must be previewed first. Call validate_sdg_job with mode "
    '"preview", let the user approve the ~10-sample preview, then confirm the full '
    "run. If you already previewed but changed the recipe, re-run the preview so the "
    "user approves the new configuration before the full job is created."
)
_TRAINING_DATA_NUDGE = (
    "Confirm the training data before the job card. Call get_dataset on the training "
    "run and show the user its record count — even if the data came in via "
    "parent_job_id (auto-resolving data is not the same as the user confirming it)."
)
_TRAINING_MODEL_NUDGE = (
    "Let the user choose the base model/method before the job card. Present the options "
    "with present_options (with VRAM estimates) and wait for their reply — never "
    "auto-select a base model and jump straight to the confirmation."
)
_TRAINING_VRAM_NUDGE = (
    "Show the VRAM estimate before the job card. Call estimate_training_resources for "
    "the final configuration so the user sees the resource cost before confirming."
)
_EVAL_CHOICE_NUDGE = (
    "Let the user decide the evaluation setup before the job card. The judge model has "
    "no default — ask which model to judge with (present_options, a soft suggestion is "
    "fine) and wait for their reply before confirming the eval job."
)


def _record_vram(s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str) -> None:
    if any(_tool_name(p) == _VRAM_TOOL for p in parts):
        s.shown_vram = True


def _record_data(s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str) -> None:
    if any(_tool_name(p) in _DATA_CONFIRM_TOOLS for p in parts):
        s.confirmed_data = True


def _record_choice(choice: str) -> _SignalRecorder:
    """A recorder that marks `choice` offered when a question leads the turn inside that
    choice's own subagent (ask_turn + matching target)."""

    def record(s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str) -> None:
        if ask and target == choice:
            s.offered_choice.add(choice)

    return record


def _reset_choice(choice: str) -> Callable[[SessionState], None]:
    return lambda s: s.offered_choice.discard(choice)


# A gate's signal recorder: given the turn's parts, whether a question leads it, and the
# active subagent target, turn on the session signal this gate reads.
_SignalRecorder = Callable[["SessionState", list[dict[str, Any]], bool, str], None]


@dataclass(frozen=True)
class _ConfirmGate:
    """One ordered precondition a confirm card must clear before it may render. Each gate
    owns the whole lifecycle of its prerequisite: the predicate that reads the session
    signal (`satisfied`), the steer emitted when unmet (`nudge`), how the signal is turned
    ON from a turn's activity (`record`), and how it is cleared once the card renders so a
    second job of the type re-gates (`reset`). `record`/`reset` are None for a gate whose
    signal is maintained elsewhere (the SDG preview sig is recorded when the preview card
    itself renders)."""

    name: str
    satisfied: Callable[[SessionState, dict[str, Any]], bool]
    nudge: str
    record: _SignalRecorder | None = None
    reset: Callable[[SessionState], None] | None = None


# The confirm-card precondition registry — the single ordered table the gate consults to
# answer "may this validate_* card render this turn, or must a prerequisite fire first?".
# Per confirm tool, the preconditions in dependency order; the FIRST unmet one holds the
# card and steers the model. Declarative: a new gate (an eval metric review, a training
# method card) is one more entry, not new control flow. (Readiness preconditions that
# need the DB — an upstream job being 'succeeded' — stay in the async stage-gate
# `_not_ready_violations`, deduped there against the dispatch-provenance check; this
# table is the SYNC half that turns on session-local signals.)
_CONFIRM_PRECONDITIONS: dict[str, tuple[_ConfirmGate, ...]] = {
    "validate_sdg_job": (
        _ConfirmGate(
            "preview",
            lambda s, c: _validate_mode(c) != "create"
            or _sdg_config_signature(c) in s.approved_preview_sigs,
            _PREVIEW_FIRST_NUDGE,
        ),
    ),
    "validate_training_job": (
        _ConfirmGate(
            "data", lambda s, c: s.confirmed_data, _TRAINING_DATA_NUDGE,
            record=_record_data, reset=lambda s: setattr(s, "confirmed_data", False),
        ),
        _ConfirmGate(
            "model", lambda s, c: "training" in s.offered_choice, _TRAINING_MODEL_NUDGE,
            record=_record_choice("training"), reset=_reset_choice("training"),
        ),
        _ConfirmGate(
            "vram", lambda s, c: s.shown_vram, _TRAINING_VRAM_NUDGE,
            record=_record_vram, reset=lambda s: setattr(s, "shown_vram", False),
        ),
    ),
    "validate_eval_job": (
        _ConfirmGate(
            "choice", lambda s, c: "eval" in s.offered_choice, _EVAL_CHOICE_NUDGE,
            record=_record_choice("eval"), reset=_reset_choice("eval"),
        ),
    ),
}


def _first_unmet_precondition(
    state: SessionState, card: dict[str, Any]
) -> _ConfirmGate | None:
    """The first precondition the leading confirm card has not satisfied, or None."""
    for gate in _CONFIRM_PRECONDITIONS.get(_tool_name(card), ()):
        if not gate.satisfied(state, card):
            return gate
    return None


def _record_confirm_signals(
    state: SessionState, parts: list[dict[str, Any]], ask_turn: bool
) -> None:
    """Accumulate the prerequisite-action signals confirm cards read, by letting every
    gate that owns a recorder observe this turn (the VRAM estimate and dataset inspection
    passive tool calls, and — when a question leads inside a training/eval subagent — that
    the user was offered a choice). Each gate's recorder sets only its own signal."""
    target = state.subagent_target or ""
    for gates in _CONFIRM_PRECONDITIONS.values():
        for gate in gates:
            if gate.record is not None:
                gate.record(state, parts, ask_turn, target)


_JOB_TYPE_FOR_CONFIRM = {
    "validate_sdg_job": "sdg",
    "validate_training_job": "training",
    "validate_eval_job": "eval",
}
_CONFIRM_TOOL_FOR_JOB_TYPE = {jtype: tool for tool, jtype in _JOB_TYPE_FOR_CONFIRM.items()}


def _reset_confirm_signals(state: SessionState, job_type: str) -> None:
    """Clear a job type's prerequisite signals once its confirm card has rendered, so a
    SECOND job of that type in the same conversation must re-establish them — each gate
    resets its own signal."""
    tool = _CONFIRM_TOOL_FOR_JOB_TYPE.get(job_type, "")
    for gate in _CONFIRM_PRECONDITIONS.get(tool, ()):
        if gate.reset is not None:
            gate.reset(state)
