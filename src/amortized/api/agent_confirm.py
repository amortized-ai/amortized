"""Confirm-card precondition policy: the single declarative table (and the pure engine
that consults it) deciding whether a validate_* confirmation card may render this turn, or
must hold while a prerequisite setup step fires first.

This is the SYNC half of the stage-gating — it turns on and reads session-local signals
(preview approved, data confirmed, model/judge offered, VRAM shown, eval metrics approved).
Readiness
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

# The validate_*_job tools that render a job-launch confirmation card, and the job type
# each one commits. This map is the single source of truth for that correspondence across
# the proxy (agent.py imports it); the tool set and the inverse derive from it, so a new
# job type is declared in exactly one place and the three can never drift apart.
_JOB_TYPE_FOR_CONFIRM = {
    "validate_sdg_job": "sdg",
    "validate_training_job": "training",
    "validate_eval_job": "eval",
}
_CONFIRM_TOOL_FOR_JOB_TYPE = {jtype: tool for tool, jtype in _JOB_TYPE_FOR_CONFIRM.items()}
_CONFIRM_CARD_TOOLS = frozenset(_JOB_TYPE_FOR_CONFIRM)

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


# Tools that establish a confirm card's prerequisite ACTIONS, tracked as session
# signals (see SessionState). The VRAM estimate and dataset inspection are ordinary
# (passive) tool calls that must have happened before a training confirm.
_VRAM_TOOL = "estimate_training_resources"
_DATA_CONFIRM_TOOLS = {"get_dataset", "get_dataset_samples", "split_dataset"}


# Steering appended when a confirm card is HELD because a prerequisite is unmet. One per
# precondition; the first unmet one's text is what the model sees. (SDG previewing is a
# prompt default, not a hard gate — see the sdg workflow — so there is no preview nudge.)
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
_EVAL_DATA_NUDGE = (
    "Confirm the eval dataset before the job card. Call get_dataset on the eval run and "
    "show the user its record count — the eval set's own size — so they confirm what will "
    "be scored (auto-resolving a dataset is not the same as the user confirming it)."
)
_EVAL_METRICS_NUDGE = (
    "Present the eval metrics for approval before the job card. Show the rubric criteria as "
    "a table and ask with present_options ('Looks good, continue' / 'Suggest changes'); "
    "wait for the user to approve them before confirming the eval job."
)


# The two eval-only present_options are the judge offer and the rubric approval; only their
# CONTENT tells them apart. A rubric approval is recognized by the metric vocabulary in its
# question/step, or by the approval options step-3-metrics mandates — so the judge offer
# ("Which judge model?") does not accidentally satisfy the metrics gate.
_METRICS_REVIEW_TERMS = ("metric", "rubric", "criteri")
_METRICS_APPROVAL_LABELS = ("suggest changes", "looks good", "reuse these metrics")


def _is_metrics_review(part: dict[str, Any]) -> bool:
    if _tool_name(part) != "present_options":
        return False
    inp = _get_tool_input(part)
    haystack = f"{inp.get('question', '')} {inp.get('step', '')}".lower()
    if any(term in haystack for term in _METRICS_REVIEW_TERMS):
        return True
    for opt in inp.get("options") or []:
        if isinstance(opt, dict):
            label = f"{opt.get('title', '')} {opt.get('value', '')}".lower()
            if any(approval in label for approval in _METRICS_APPROVAL_LABELS):
                return True
    return False


def _record_vram(s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str) -> None:
    # Both signals gate ONLY validate_training_job, so record them only while the
    # training subagent is active — a VRAM estimate or get_dataset emitted inside an
    # SDG/eval subagent must not pre-satisfy a later training confirm (matches the
    # target-scoping _record_choice already applies to the model-choice gate).
    if target == "training" and any(_tool_name(p) == _VRAM_TOOL for p in parts):
        s.shown_vram = True


def _record_data(s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str) -> None:
    if target == "training" and any(_tool_name(p) in _DATA_CONFIRM_TOOLS for p in parts):
        s.confirmed_data = True


def _record_eval_data(s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str) -> None:
    # The eval dataset's own confirmation — scoped to the eval subagent so a training
    # get_dataset can't pre-satisfy it (mirrors the target-scoping on _record_data/_record_vram).
    if target == "eval" and any(_tool_name(p) in _DATA_CONFIRM_TOOLS for p in parts):
        s.eval_data_confirmed = True


def _record_metrics(s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str) -> None:
    if target == "eval" and any(_is_metrics_review(p) for p in parts):
        s.metrics_reviewed = True


def _record_eval_choice(
    s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str
) -> None:
    # The judge/setup offer — a NON-metrics eval present_options. The metrics-approval
    # present_options has its own gate, so it must not also satisfy the judge gate (else a
    # model could approve metrics and skip the judge offer entirely).
    if ask and target == "eval" and any(
        _tool_name(p) == "present_options" and not _is_metrics_review(p) for p in parts
    ):
        s.offered_choice.add("eval")


def _record_choice(choice: str) -> _SignalRecorder:
    """A recorder that marks `choice` offered when a question leads the turn inside that
    choice's own subagent (ask_turn + matching target)."""

    def record(s: SessionState, parts: list[dict[str, Any]], ask: bool, target: str) -> None:
        if ask and target == choice:
            s.offered_choice.add(choice)

    return record


# A gate's signal recorder: given the turn's parts, whether a question leads it, and the
# active subagent target, turn on the session signal this gate reads.
_SignalRecorder = Callable[["SessionState", list[dict[str, Any]], bool, str], None]

# A gate's config fingerprint: a stable hash of ONLY the validate_* input fields this gate's
# prerequisite is about (the dataset identity for `data`, the base model for `model`, the
# VRAM-affecting hyperparameters for `vram`). Compared against the fingerprint stamped when
# the card last rendered: equal → the prerequisite still holds (an unchanged resubmit or a
# refinement of an unrelated field renders with no hold); different → that one field changed,
# so re-gate just that prerequisite (a new base model re-asks model + VRAM, not the data).
_Fingerprint = Callable[[dict[str, Any]], str]


def _config_fp(*keys: str) -> _Fingerprint:
    """Fingerprint exactly `keys` from a validate_* card's input. An explicit key list (not
    the whole dict) so an irrelevant extra field — both configs are extra='allow' — never
    spuriously re-gates; missing keys hash as null, so dropping a field (e.g. the broken
    max_tokens_per_gpu) that no gate tracks leaves every fingerprint unchanged."""

    def fp(card_input: dict[str, Any]) -> str:
        payload = {k: card_input.get(k) for k in keys}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    return fp


# Per-gate fingerprint field lists, keyed to the real validate_* request models
# (TrainingJobRequest/EvalJobRequest in models.py). The tool-call input is the FLAT request
# (the MCP tool auto-derives from the endpoint body), so these are top-level keys.
_FP_TRAIN_DATA = _config_fp("parent_job_id", "data_run_id", "data_path")
_FP_TRAIN_MODEL = _config_fp("model_name_or_path", "training_job_id", "algorithm")
_FP_TRAIN_VRAM = _config_fp(
    "model_name_or_path", "training_job_id", "algorithm",
    "per_device_train_batch_size", "max_length", "gradient_accumulation_steps",
    "gradient_checkpointing", "bf16", "load_in_4bit", "use_peft",
    "lora_r", "lora_alpha", "lora_dropout", "unfreeze_rank_ratio",
)
_FP_EVAL_DATA = _config_fp("parent_job_id", "eval_data_run_id")
_FP_EVAL_METRICS = _config_fp("rubric", "eval_mode")
_FP_EVAL_CHOICE = _config_fp(
    "judge", "endpoint", "training_job_id", "model_name_or_path", "served_model_name"
)


@dataclass(frozen=True)
class _ConfirmGate:
    """One ordered precondition a confirm card must clear before it may render. Each gate
    owns the whole lifecycle of its prerequisite: the predicate that reads the session
    signal (`satisfied`), the steer emitted when unmet (`nudge`), how the signal is turned
    ON from a turn's activity (`record`), and the fingerprint of the config fields this
    prerequisite covers (`fingerprint`) — stamped when the card renders and compared on the
    next card so a CHANGED field re-gates while an unchanged resubmit does not. `record`/
    `fingerprint` are None for a gate whose signal is maintained elsewhere."""

    name: str
    satisfied: Callable[[SessionState, dict[str, Any]], bool]
    nudge: str
    record: _SignalRecorder | None = None
    fingerprint: _Fingerprint | None = None


# The confirm-card precondition registry — the single ordered table the gate consults to
# answer "may this validate_* card render this turn, or must a prerequisite fire first?".
# Per confirm tool, the preconditions in dependency order; the FIRST unmet one holds the
# card and steers the model. Declarative: a new gate (an eval metric review, a training
# method card) is one more entry, not new control flow. (Readiness preconditions that
# need the DB — an upstream job being 'succeeded' — stay in the async stage-gate
# `_not_ready_violations`, deduped there against the dispatch-provenance check; this
# table is the SYNC half that turns on session-local signals.)
_CONFIRM_PRECONDITIONS: dict[str, tuple[_ConfirmGate, ...]] = {
    # No SDG precondition: previewing before a full run is a strong PROMPT default (see the
    # sdg workflow), not a hard gate — a capable model, or an explicit user "skip the
    # preview", may create directly. Hard gates stay only for user-independent prereqs.
    "validate_training_job": (
        _ConfirmGate(
            "data", lambda s, c: s.confirmed_data, _TRAINING_DATA_NUDGE,
            record=_record_data, fingerprint=_FP_TRAIN_DATA,
        ),
        _ConfirmGate(
            "model", lambda s, c: "training" in s.offered_choice, _TRAINING_MODEL_NUDGE,
            record=_record_choice("training"), fingerprint=_FP_TRAIN_MODEL,
        ),
        _ConfirmGate(
            "vram", lambda s, c: s.shown_vram, _TRAINING_VRAM_NUDGE,
            record=_record_vram, fingerprint=_FP_TRAIN_VRAM,
        ),
    ),
    "validate_eval_job": (
        _ConfirmGate(
            "data", lambda s, c: s.eval_data_confirmed, _EVAL_DATA_NUDGE,
            record=_record_eval_data, fingerprint=_FP_EVAL_DATA,
        ),
        _ConfirmGate(
            "metrics", lambda s, c: s.metrics_reviewed, _EVAL_METRICS_NUDGE,
            record=_record_metrics, fingerprint=_FP_EVAL_METRICS,
        ),
        _ConfirmGate(
            "choice", lambda s, c: "eval" in s.offered_choice, _EVAL_CHOICE_NUDGE,
            record=_record_eval_choice, fingerprint=_FP_EVAL_CHOICE,
        ),
    ),
}


def _first_unmet_precondition(
    state: SessionState, card: dict[str, Any]
) -> _ConfirmGate | None:
    """The first precondition the leading confirm card has not satisfied, or None. A gate is
    unmet if its signal was never earned OR — once earned — the config field it covers has
    CHANGED since the card last rendered (fingerprint mismatch), so a new base model re-asks
    model + VRAM while an unchanged resubmit renders. The first-ever card has no stored
    fingerprint to compare against, so it gates on the signal alone."""
    tool = _tool_name(card)
    card_input = _get_tool_input(card)
    for gate in _CONFIRM_PRECONDITIONS.get(tool, ()):
        if not gate.satisfied(state, card):
            return gate
        if gate.fingerprint is not None:
            stored = state.gate_fp.get((tool, gate.name))
            if stored is not None and stored != gate.fingerprint(card_input):
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


def _stamp_confirm_fingerprints(state: SessionState, card: dict[str, Any]) -> None:
    """Record the config fingerprint of each gate's covered fields when a validate_* card
    RENDERS (see _enforce_single_interaction), so the NEXT card for the same job re-gates only
    the prerequisites whose fields changed. Replaces the old per-delegation signal reset: a
    same-config resubmit now matches every fingerprint and renders without a hold (the
    deployed-run bug where a 'fresh' resubmit wiped still-valid signals and held the card)."""
    tool = _tool_name(card)
    card_input = _get_tool_input(card)
    for gate in _CONFIRM_PRECONDITIONS.get(tool, ()):
        if gate.fingerprint is not None:
            state.gate_fp[(tool, gate.name)] = gate.fingerprint(card_input)
