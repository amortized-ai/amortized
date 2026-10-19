"""Agent session proxy with subagent routing.

Proxies chat messages to OpenCode and intercepts delegation/completion
signals to route conversations between the orchestrator and ephemeral
subagent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import ssl
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from amortized.api import monitor_log
from amortized.api.agent_claims import (
    _job_claim_caveat,
    _job_claim_correction,
)
from amortized.api.agent_confirm import (
    _CONFIRM_CARD_TOOLS,
    _CONFIRM_TOOL_FOR_JOB_TYPE,
    _JOB_TYPE_FOR_CONFIRM,
    _REVIEW_CARD_TOOLS,
    _first_unmet_precondition,
    _record_confirm_signals,
    _reset_confirm_signals,
    _review_card_signature,
)
from amortized.api.agent_parts import _get_tool_input, _part_errored, _tool_name
from amortized.api.agent_sdg_prompt import (
    _bind_prompts_to_config as _bind_prompts_to_config,  # re-export for tests
)
from amortized.api.agent_sdg_prompt import (
    _bind_sdg_prompt_to_approved,
)
from amortized.api.agent_sdg_prompt import (
    _shown_prompts as _shown_prompts,  # re-export for tests
)
from amortized.api.phases import (
    phase_block,
    skill_step_count,
    subagent_phase,
    system_reminder,
    with_block,
    with_skill_step_block,
)
from amortized.config import settings
from amortized.core import lineage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])

OPENCODE_TIMEOUT = 300.0
SESSION_TTL_HOURS = 4

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


# Retention for finished turns awaiting a client poll. A finished result is kept
# until the client has polled it (consumed) — never evicted merely because newer
# turns arrived — with a TTL so an abandoned client's result eventually clears, and
# a hard ceiling to bound memory against a client that never polls.
MAX_TURNS_TRACKED = 8
UNPOLLED_TURN_TTL = timedelta(minutes=10)
MAX_TURNS_HARD = 64
# Max pending (active, not-yet-finished) turns per session. Turns are serialized by
# state.lock, so beyond this a client is only queuing work unboundedly — reject with
# 429 rather than letting state.turns / _background_tasks grow without limit.
MAX_ACTIVE_TURNS = 16


@dataclass
class TurnState:
    """A single message turn, run in the background so the HTTP POST returns fast.

    The blocking work (orchestrator + optional subagent turn) can exceed proxy
    timeouts; running it detached and polling the result via GET keeps every HTTP
    request short. ``result`` mirrors the shape the POST used to return.
    """

    active: bool = True
    result: dict[str, Any] | None = None
    error: str | None = None
    error_status: int | None = None
    error_code: str | None = None
    consumed: bool = False
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    task: asyncio.Task[None] | None = None


@dataclass
class SessionState:
    orchestrator_id: str
    subagent_id: str | None = None
    subagent_target: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    completed_subagents: dict[str, str] = field(default_factory=dict)
    # The consumer (eval/training) that handed back via request_orchestrator_help and awaits
    # a dataset: pending_help_target is the consumer to RESUME, pending_help_producer is the
    # producer (sdg) it is WAITING ON. The proxy auto-resumes the consumer only when THAT
    # producer completes — matching the awaited producer (not merely "some other subagent
    # finished") stops an unrelated completion during a detour from spuriously resurrecting
    # the consumer. Both are armed once the orchestrator actually delegates to the producer,
    # and cleared together on resume/teardown.
    pending_help_target: str | None = None
    pending_help_producer: str | None = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    # When this conversation's proxy state was created. A job whose DB created_at
    # predates this is from an EARLIER conversation, so reporting it as "just
    # generated" is undisclosed reuse — see _apply_job_claim_gate.
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    turns: dict[str, TurnState] = field(default_factory=dict)
    turn_order: list[str] = field(default_factory=list)
    # opencode assistant-message ids already attributed to a monitor turn record,
    # so each message is counted once even across delegating turns / sessions.
    seen_message_ids: set[str] = field(default_factory=set)
    # Last workflow step a subagent actually signalled, so when it forgets to call
    # signal_phase the server can backfill the UI progress bar at the right step
    # instead of snapping it back to the start of the phase.
    last_signal_step: str = ""
    # The skill each subagent target has loaded (its frontmatter `name`, captured from the
    # opencode `skill` tool call), keyed by target so one subagent's skill never bleeds into
    # another's turns. Drives per-step skill injection for stepped skills.
    active_skill_by_target: dict[str, str] = field(default_factory=dict)
    # Per-target cursor into the active stepped skill's step slices: the index of the step
    # whose detail to inject next. Advances one step per subagent turn while a stepped skill
    # is active; reset when a fresh session is delegated to the target.
    skill_step_idx: dict[str, int] = field(default_factory=dict)
    # Content signatures of review cards (e.g. an assessor prompt) already rendered
    # to the user standalone. The review gate fires only for a prompt NOT in this
    # set; once shown, a later confirm that re-includes the same prompt is allowed
    # (the redundant review card is dropped), so a driver that re-shows the prompt
    # before every validate_* can't livelock. See _enforce_single_interaction.
    reviewed_card_sigs: set[str] = field(default_factory=set)
    # Model-directed steers the interaction gate produces when it HOLDS a card — never
    # shown to the user (the old code leaked them as user-facing text). Two delivery modes:
    #  - SOFT (a job confirm stacked under an unreviewed prompt): delivered to the model on
    #    its NEXT turn (prepended in _handle_*_message). The prompt still renders this turn
    #    so the user can review it; the steer just stops the model re-stacking the confirm.
    pending_interaction_steer: str | None = None
    #  - HARD (an unmet committing-card precondition — a full-run / train / eval confirm the
    #    model jumped to): _run_turn posts it and RE-RUNS the model this same turn, so the
    #    premature committing card never reaches the client (the stale turn falls behind the
    #    steer message, like the provenance gate). See _enforce_single_interaction + _run_turn.
    pending_rerun_steer: str | None = None
    # Prerequisite ACTIONS a confirm card requires, accumulated as turns render them, so
    # the (sync) confirm-precondition evaluator can hold a validate_* card the model
    # jumped to without the mandated setup. Reset for a job type once its confirm card
    # renders, so a SECOND job of that type in the same conversation re-gates. See
    # _enforce_single_interaction / _CONFIRM_PRECONDITIONS.
    shown_vram: bool = False  # estimate_training_resources rendered (training)
    confirmed_data: bool = False  # a dataset tool (get_dataset/…) rendered (training data)
    offered_choice: set[str] = field(default_factory=set)  # targets that asked present_options
    eval_data_confirmed: bool = False  # a dataset tool rendered inside the eval subagent
    metrics_reviewed: bool = False  # eval rubric/metric set presented for approval (eval)
    # callIDs of interactive cards (stacked present_options, a repeat confirm, a redundant
    # review) the interaction gate stripped from a turn result. get_session_messages filters
    # these so a card removed from the result is also absent from the raw opencode history the
    # UI re-fetches — otherwise the client re-derives the stripped card and the gate is a
    # no-op for rendering. The committing-card hold uses the _run_turn re-run rail instead;
    # this covers the non-committing strips it doesn't. Cumulative (callIDs are unique; bounded
    # by the session's tool-call count). See _enforce_single_interaction.
    suppressed_call_ids: set[str] = field(default_factory=set)


_sessions: dict[str, SessionState] = {}

# Strong references to in-flight turn tasks so they are not garbage-collected before
# completion (asyncio holds only weak references to tasks); each removes itself on done.
_background_tasks: set[asyncio.Task[None]] = set()

# ---------------------------------------------------------------------------
# Shared HTTP client
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None
_cleanup_task: asyncio.Task[None] | None = None


def _upstream_client_kwargs() -> dict[str, Any]:
    """TLS/mTLS options for the agent upstream.

    The default in-cluster opencode Service is plain HTTP and needs none of these.
    An OpenShell-sandboxed opencode is only reachable via the gateway, which requires
    a client certificate (mTLS) plus CA verification of the gateway's server cert.

    When a client cert is configured we build a single SSLContext holding both the CA
    trust and the client cert. httpx does not reliably present a client cert when
    ``cert`` and a ``verify`` CA path are passed separately (the server then aborts the
    TLS 1.3 handshake with CERTIFICATE_REQUIRED).
    """
    cert = settings.agent_upstream_client_cert
    key = settings.agent_upstream_client_key
    if not (cert and key):
        if settings.agent_upstream_insecure_tls:
            return {"verify": False}
        if settings.agent_upstream_ca_bundle:
            return {"verify": settings.agent_upstream_ca_bundle}
        return {}
    if settings.agent_upstream_insecure_tls:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    elif settings.agent_upstream_ca_bundle:
        ctx = ssl.create_default_context(cafile=settings.agent_upstream_ca_bundle)
    else:
        ctx = ssl.create_default_context()
    ctx.load_cert_chain(cert, key)
    return {"verify": ctx}


async def startup() -> None:
    global _http_client, _cleanup_task
    _http_client = httpx.AsyncClient(timeout=OPENCODE_TIMEOUT, **_upstream_client_kwargs())
    _cleanup_task = asyncio.create_task(_session_cleanup_loop())


async def shutdown() -> None:
    global _http_client, _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _cleanup_task
        _cleanup_task = None
    for task in list(_background_tasks):
        task.cancel()
    _background_tasks.clear()
    if _http_client:
        await _http_client.aclose()
        _http_client = None


def _client() -> httpx.AsyncClient:
    assert _http_client is not None, "agent proxy not started"
    return _http_client


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------


async def _session_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(600)
        cutoff = datetime.now(UTC) - timedelta(hours=SESSION_TTL_HOURS)
        expired = [sid for sid, s in _sessions.items() if s.last_activity < cutoff]
        for sid in expired:
            del _sessions[sid]
        if expired:
            logger.info("Evicted %d expired agent sessions", len(expired))


# ---------------------------------------------------------------------------
# Response scanning — detect delegation/completion from tool parts
# ---------------------------------------------------------------------------

INTERNAL_TOOLS = {
    "delegate_to_subagent",
    "signal_subagent_completion",
    "request_orchestrator_help",
}

# Injected in place of the generic job-completion nudge when an ACTIVE SUBAGENT's
# delegated job finishes successfully — a terminal instruction to hand control
# back now, so a weaker model doesn't keep calling read-only tools (get_job,
# get_dataset_samples, …) and never signal completion.
_SUBAGENT_JOB_DONE_PROMPT = (
    "[SYSTEM EVENT] Your delegated job finished successfully. Your task for this "
    "delegation is complete. Your ONLY next action is to call "
    "signal_subagent_completion with a short summary that includes the job ID. "
    "Do NOT call any other tools (no get_job, list_jobs, get_dataset, "
    "get_dataset_samples, or present_options) and do NOT inspect or re-verify the "
    "dataset — hand control back now."
)

# Counterpart for a FAILED job. Without a defined hand-back, a weak producer keeps control
# after its job fails (retrying / inspecting) and the flow stalls. Steer it to hand control
# back immediately with a failure summary, so the proxy can resume the waiting consumer with
# the bad news instead of hanging.
_SUBAGENT_JOB_FAILED_PROMPT = (
    "[SYSTEM EVENT] Your delegated job FAILED. Your task for this delegation is over. Your "
    "ONLY next action is to call signal_subagent_completion with a short summary that states "
    "the job FAILED and includes the job ID. Do NOT retry and do NOT call any other tools — "
    "hand control back now."
)

# Seeded into the orchestrator when a subagent hands back via request_orchestrator_help:
# a subagent (e.g. eval) needs a dataset it can't produce itself (e.g. a held-out set).
# The orchestrator is the sole delegator, so its one job is to run the producing
# subagent (sdg) — handing it ONLY a plain dataset build-order, never the caller's
# {context} (that note is for the orchestrator; the producer is a leaf and must not
# learn the rest of the pipeline exists). The resume is NOT its job — when the producer
# completes, the proxy resumes the requester automatically (see pending_help_target).
# {target}/{need}/{context} are filled from the caller's target and the tool's arguments.
_SUBAGENT_HELP_PROMPT = (
    "[SUBAGENT NEEDS HELP]\n"
    "The {target} workflow needs a dataset it cannot produce itself.\n\n"
    "dataset needed: {need}\n\n"
    "background for YOU (do NOT pass this on): {context}\n\n"
    "Your ONLY action this turn: a single delegate_to_subagent call to the producer "
    '(target "sdg", resume: false). Its context must be ONLY a plain build order for '
    "the dataset above, worded as if a user asked for that dataset directly — do NOT "
    "mention the {target} step, resuming, waiting, or anything else about the pipeline "
    "(the producer is a leaf and must not know the rest exists). The platform resumes "
    "the {target} agent automatically once the job finishes."
)


def _job_succeeded(body: MessageRequest, user_text: str) -> bool:
    """Whether a job_complete event reports success. Prefers the structured
    `outcome` field; falls back to the legacy display-text substring only when the
    client didn't send one (version skew during a rollout)."""
    if body.outcome is not None:
        return body.outcome.strip().lower() == "succeeded"
    return "status: succeeded" in user_text.lower()


_VALID_TARGETS = {"sdg", "training", "eval"}


def _detect_delegation(parts: list[dict[str, Any]]) -> tuple[str, str, bool] | None:
    for part in parts:
        if part.get("type") != "tool":
            continue
        if _tool_name(part) == "delegate_to_subagent":
            inp = _get_tool_input(part)
            target = inp.get("target", "")
            if target not in _VALID_TARGETS:
                logger.warning("Ignoring delegation to unknown target: %r", target)
                return None
            return (target, inp.get("context", ""), inp.get("resume", False))
    return None


def _detect_completion(parts: list[dict[str, Any]]) -> str | None:
    for part in parts:
        if part.get("type") != "tool":
            continue
        if _tool_name(part) == "signal_subagent_completion":
            inp = _get_tool_input(part)
            return str(inp.get("summary", "")) or "Task completed."
    return None


def _detect_help_request(parts: list[dict[str, Any]]) -> tuple[str, str] | None:
    """A subagent's request_orchestrator_help call → (need, context), else None.

    A subagent is a leaf: it cannot delegate. When it needs a prerequisite it can't
    produce itself (e.g. the eval agent needs a held-out dataset), it hands back to
    the orchestrator with this structured request instead."""
    for part in parts:
        if part.get("type") != "tool":
            continue
        if _tool_name(part) == "request_orchestrator_help":
            inp = _get_tool_input(part)
            return (str(inp.get("need", "")), str(inp.get("context", "")))
    return None


def _strip_internal_tools(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in parts if _tool_name(p) not in INTERNAL_TOOLS]


def _enforce_single_interaction(result: dict[str, Any], state: SessionState) -> dict[str, Any]:
    """A turn either ASKS the user one question or CONFIRMS one job — never both.

    present_options asks a question and MUST end the turn; a validate_* renders a
    confirmation card that already has its own Confirm/Cancel. A weaker driver
    batches them, in either order, into one turn:

    - question(s) first, then a premature validate_* — the user never gets to
      pick (the sample-size / system-prompt choices get steamrolled by a confirm
      card). Fix: the first question wins; the extra questions and the confirm
      card are stripped.
    - validate_* first, then a present_options — the confirmation card renders
      with redundant option buttons stacked underneath it (observed: a "full
      SFT" confirm card with "Confirm & train / Adjust settings" options below).
      Fix: the card wins; the stacked question is stripped.
    - a review card (show_prompt — an assessor/system prompt that ships in the
      training data) then a validate_* — the driver shows the prompt and in the
      SAME turn stacks the job confirmation under it, so the user confirms before
      reviewing the prompt. Fix: the review wins; the confirm card is stripped,
      leaving the prompt on its own turn. The gate fires ONLY for a prompt the
      user has not already seen (tracked by content signature in session state);
      once reviewed, a later confirm that re-includes the same prompt proceeds and
      the now-redundant review card is dropped instead — so a driver that re-shows
      the prompt before every validate_* can't livelock. An edited prompt (new
      signature) re-gates.
    - a confirm card (validate_*) whose ordered PREREQUISITES aren't met yet — the
      driver jumped to the job card without the mandated setup. Fix: the card is held
      and the model is steered to do the missing step first. The prerequisites per job
      type are declared in _CONFIRM_PRECONDITIONS and the first unmet one fires:
        · SDG full run (mode="create") — a preview of the SAME recipe must have been
          shown standalone first (tracked by recipe signature; an edited recipe
          re-gates). A preview shown standalone records its signature; the matching
          full-job confirm then proceeds on a later turn.
        · Training — the user must have confirmed the data (get_dataset shown), been
          offered a base-model/method choice (present_options in the training step),
          and seen the VRAM estimate, before the job card.
        · Eval — the eval dataset must have been confirmed (get_dataset shown), the rubric/
          metric set presented for approval, and the evaluation setup offered (the judge has
          no default), before the job card.
      These prerequisite ACTIONS are tracked as session signals accumulated as turns
      render them (_record_confirm_signals) and cleared when a new delegation for that
      target begins (_reset_confirm_signals in _maybe_delegate) so a NEW job of the type
      re-gates while an in-session refinement of the same card does not. (Readiness
      prerequisites that need the DB — an upstream job being 'succeeded' — live in the
      async stage-gate _not_ready_violations, deduped there against dispatch-provenance;
      this function is the sync half that turns on session-local signals.)

    Whichever interaction LEADS the turn is kept; every OTHER interactive element
    (extra questions, repeat/stacked confirm cards) is stripped. Passive content
    — explanatory text, show_* display cards, signal_phase — is ALWAYS kept, even
    after the kept interaction, so content the question is about (e.g. a proposed
    metric-set table the driver wrote as trailing text) is never dropped. The
    monitor log is recorded from the raw opencode parts, not this result, so the
    batching stays visible/measurable there."""
    parts = result.get("parts") or []
    first_q = first_card = last_card = first_review = None
    for i, part in enumerate(parts):
        # An errored tool call renders nothing — skip it so a failed validate_* can't lead
        # the turn and shadow the model's successful retry in the same turn (which would keep
        # the dead card and strip the live one the user must confirm).
        if _part_errored(part):
            continue
        name = _tool_name(part)
        if first_q is None and name == "present_options":
            first_q = i
        if name in _CONFIRM_CARD_TOOLS:
            # first_card fixes the ask-vs-confirm ORDER; last_card is the card we KEEP — the
            # last valid one, so a corrected/re-validated config supersedes an earlier one.
            # This matches the client's #459 "last valid wins" backward-walk; they must agree
            # because the gate suppresses the rest, so the client can't recover a dropped card.
            if first_card is None:
                first_card = i
            last_card = i
        if first_review is None and name in _REVIEW_CARD_TOOLS:
            first_review = i

    ask_turn = first_q is not None and (first_card is None or first_q < first_card)

    # Accumulate the prerequisite-action signals a later confirm card reads (VRAM
    # estimate / dataset inspection are passive and may land on a setup turn with no
    # question or card at all), before the early return below.
    _record_confirm_signals(state, parts, ask_turn)

    if first_q is None and first_card is None:
        # A review card shown on its own (no question, no confirm card) IS reviewed this
        # turn — record it so a later turn that stacks it under a confirm doesn't re-gate.
        if first_review is not None:
            state.reviewed_card_sigs.add(_review_card_signature(parts[first_review]))
        return result

    # Review gate: a review card must be reviewed on its own turn before the job it
    # gates is confirmed — but only the FIRST time the user sees that prompt, so a
    # driver re-showing it before every confirm can't livelock.
    review_sig = _review_card_signature(parts[first_review]) if first_review is not None else None
    new_review = review_sig is not None and review_sig not in state.reviewed_card_sigs
    prompt_gate = not ask_turn and first_card is not None and new_review

    # Confirm-card precondition gate: a validate_* card may lead a turn only after its
    # ordered prerequisites (preview approved / data confirmed / model offered / VRAM
    # shown / eval setup chosen) are met. The first unmet one holds the card and steers
    # the model. The prompt review gate takes precedence, so this runs only when
    # prompt_gate is not already firing.
    unmet = (
        _first_unmet_precondition(state, parts[last_card])
        if (not ask_turn and not prompt_gate and last_card is not None)
        else None
    )

    if prompt_gate or unmet is not None:
        # No interaction leads: keep passive parts (incl. the review card); drop the
        # confirm card so the prompt/preview stands alone. The job is confirmed next turn.
        keep_index: int | None = None
        drop_review = False
    else:
        # Whichever interaction leads the turn is the one the model is really making;
        # keep exactly that one and strip every OTHER interactive element. For a confirm
        # turn that's the LAST valid card (last_card) — a corrected/re-validated config
        # supersedes an earlier one. If a confirm proceeds with an already-reviewed prompt
        # stacked on it, drop the now-redundant review card (its text is already inside it).
        keep_index = first_q if ask_turn else last_card
        drop_review = not ask_turn and first_card is not None and review_sig is not None

    kept_indices = {
        i
        for i, part in enumerate(parts)
        if i == keep_index
        or (
            _tool_name(part) != "present_options"
            and _tool_name(part) not in _CONFIRM_CARD_TOOLS
            and not (drop_review and _tool_name(part) in _REVIEW_CARD_TOOLS)
        )
    }
    kept = [part for i, part in enumerate(parts) if i in kept_indices]
    # Record the dropped cards' callIDs so get_session_messages also hides them: the client
    # re-derives this turn's cards from its own GET and would otherwise re-render a card the
    # gate removed from the result. (The committing-card hold is additionally re-run-railed in
    # _run_turn; recording its callID here is harmless — that message sits behind the steer.)
    for i, part in enumerate(parts):
        if i not in kept_indices:
            call_id = part.get("callID")
            if call_id:
                state.suppressed_call_ids.add(str(call_id))
    # Mark the prompt reviewed once it is actually rendered to the user this turn,
    # so the next confirm that re-includes it proceeds instead of re-gating.
    if review_sig is not None and not drop_review:
        state.reviewed_card_sigs.add(review_sig)
    # NB: the confirm signals are NOT reset here when the card renders. Resetting on render
    # treated a mere REFINEMENT of the same not-yet-submitted card (e.g. the user tweaks
    # epochs, so the model re-issues validate_training_job) as a brand-new job and re-gated
    # it — bouncing the model back to re-ask the model/data it already settled. The reset
    # now happens at the delegation boundary (_maybe_delegate) instead, so a genuinely new
    # job (a fresh delegation) re-gates while in-session refinements and the eval base-model
    # reuse cycle keep their signals.

    # A held card steers the MODEL to do the missing step first — delivered to its
    # session on the next turn (see _handle_*_message), NOT appended to the user-facing
    # result. The old code put the steer in the result's parts, so the USER saw text
    # addressed to the model and the model (whose history showed a successful validate_*)
    # never learned the card was held.
    if unmet is not None:
        # HARD rail: _run_turn re-runs the model this same turn so the premature
        # committing card never reaches the client.
        state.pending_rerun_steer = unmet.nudge
    elif prompt_gate:
        # SOFT rail: the prompt renders this turn (the user can review it); just steer the
        # model, on its next turn, to stop stacking the confirm under it.
        state.pending_interaction_steer = (
            "You stacked a job confirmation under a prompt the user has not reviewed "
            "yet. Show the prompt on its own for review first; present the job "
            "confirmation only on a later turn, after the user approves the prompt."
        )
    if len(kept) == len(parts):
        return result
    return {**result, "parts": kept}


# Active subagent → the workflow phase its turns belong to (UI progress bar). The phase
# name equals the subagent target, so derive it from _VALID_TARGETS rather than re-listing
# the types: a target that drives a turn always has a matching phase, and the two sets
# cannot fall out of sync (a missing phase would strand the UI bar on a stale stage).
_PHASE_FOR_TARGET = {t: t for t in _VALID_TARGETS}

# The phase during which a stepped skill's per-step detail is surfaced: requirement
# gathering. Per-step injection is confined to this phase so leftover step slices can't
# bleed into the later config-building/validate turns (whose detail stays in the SKILL.md
# body). Matches the `signal_phase.step` slug the sdg/training gather phase reports.
_GATHER_PHASE = "phase-2-gather"

# How many leading steps of a stepped skill ship INSIDE the served SKILL.md (appended by
# gen_native_skills), hence are already present at load and must NOT be re-injected. Keep in
# sync with the Makefile append (currently the first step only).
_RESIDENT_SKILL_STEPS = 1


# Synthetic tool name for the phase signal _ensure_phase_signal backfills when the model
# forgot to call signal_phase. Deliberately distinct from opencode's real
# `amortized_signal_phase` so _run_turn can keep this backfill (the client's own GET lacks
# it) while dropping the other fetched tool parts it must not ship to the client (issue #505).
_BACKFILL_PHASE_TOOL = "mcp_amortized__signal_phase"


def _with_active_skill_step(state: SessionState, target: str, phase: str | None, text: str) -> str:
    """Prepend the active stepped skill's current step-detail block to a gather turn, then
    advance the per-target cursor so the next turn gets the next step.

    No-op outside the gather phase, when no skill is active, or when the active skill is not
    stepped (opt-in by `steps/` presence) — so callers wrap every gather turn without first
    checking skill state. Called once per subagent turn (the message handler / resume
    handoff), NOT in _ensure_phase_signal which can run twice on an interaction rerun."""
    if phase != _GATHER_PHASE:
        return text
    skill = state.active_skill_by_target.get(target)
    count = skill_step_count(skill)
    if not count:
        return text
    # Step 1 is appended into the served SKILL.md at build time (gen_native_skills), so it is
    # already present when the skill loads. The cursor therefore starts PAST it — the first
    # injected turn gets step 2 — and advances one step per gather turn through the rest.
    idx = state.skill_step_idx.get(target, _RESIDENT_SKILL_STEPS)
    text = with_skill_step_block(skill, idx, text)
    if idx < count:
        state.skill_step_idx[target] = idx + 1
    return text


def _ensure_phase_signal(state: SessionState, result: dict[str, Any]) -> dict[str, Any]:
    """Guarantee the UI progress bar reflects the active subagent's phase.

    The bar is driven by signal_phase tool results. A weaker driver sometimes
    forgets to call it, leaving the bar on a stale phase (the loop that stranded
    the eval stage on the training bar). When a subagent is active and its turn
    carries no signal_phase, backfill one from the server-known subagent_target,
    carrying the last step the model actually signalled. UI only — this is NOT
    recorded to the monitor log, which must still reflect what the model did."""
    phase = _PHASE_FOR_TARGET.get(state.subagent_target or "")
    if not phase:
        return result
    parts = result.get("parts") or []
    signalled = False
    for part in parts:
        name = _tool_name(part)
        if name == "signal_phase":
            step = _get_tool_input(part).get("step")
            if step:
                state.last_signal_step = str(step)
            signalled = True
        elif name == "skill":
            # opencode-native skill load: record which skill this target has active so the
            # proxy can inject that skill's per-step detail. Scanned in the same pass (not an
            # early return) because the skill call often trails a signal_phase in the turn.
            # Idempotent — safe if this runs twice in a turn (the interaction rerun loop).
            skill_name = _get_tool_input(part).get("name")
            if skill_name and state.subagent_target:
                state.active_skill_by_target[state.subagent_target] = str(skill_name)
    if signalled:
        return result  # model signalled this turn; nothing to backfill
    payload = {"phase": phase, "step": state.last_signal_step}
    result["parts"] = [
        *parts,
        {
            "type": "tool",
            "tool": _BACKFILL_PHASE_TOOL,
            "input": payload,
            "output": json.dumps(payload),
        },
    ]
    return result


async def _fetch_all_assistant_parts(opencode_session_id: str) -> list[dict[str, Any]]:
    """GET session messages and return parts from all assistant messages.

    The POST response only has step-start/step-finish — tool call details
    only appear in the GET messages endpoint. OpenCode may generate multiple
    assistant messages per user message (multi-step tool loops), so we
    collect parts from all of them.
    """
    try:
        resp = await _client().get(
            f"{_opencode_url()}/session/{opencode_session_id}/message",
            timeout=10.0,
        )
        content_type = resp.headers.get("content-type", "")
        if resp.status_code != 200 or "application/json" not in content_type:
            return []
        messages = resp.json()
        if not isinstance(messages, list):
            return []
        all_parts: list[dict[str, Any]] = []
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("info", {}).get("role") == "user":
                last_user_idx = i
        for msg in messages[last_user_idx + 1 :]:
            if msg.get("info", {}).get("role") == "assistant":
                all_parts.extend(msg.get("parts", []))
        return all_parts
    except Exception:
        logger.warning("Failed to fetch session messages for %s", opencode_session_id)
        return []


# ---------------------------------------------------------------------------
# Monitor metrics capture (best-effort; never breaks a turn)
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> int:
    """Best-effort int, folding dict token buckets (e.g. cache {read, write})."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        return sum(_coerce_int(v) for v in value.values())
    return 0


async def _fetch_all_messages(
    opencode_session_id: str,
    cache: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """GET a session's full message history (info + parts), or [] on any failure.

    When a per-turn ``cache`` is supplied, the result is memoized by session id so
    the same turn's provenance gate and metrics recorder don't each re-fetch every
    session over HTTP. The provenance gate pops a session from the cache after it
    posts a correction, so the re-grounding pass still sees the fresh messages."""
    if cache is not None and opencode_session_id in cache:
        return cache[opencode_session_id]
    try:
        resp = await _client().get(
            f"{_opencode_url()}/session/{opencode_session_id}/message",
            timeout=10.0,
        )
        content_type = resp.headers.get("content-type", "")
        if resp.status_code != 200 or "application/json" not in content_type:
            result: list[dict[str, Any]] = []
        else:
            messages = resp.json()
            result = messages if isinstance(messages, list) else []
    except Exception:
        logger.warning("monitor: failed to fetch messages for %s", opencode_session_id)
        result = []
    if cache is not None:
        cache[opencode_session_id] = result
    return result


# ---------------------------------------------------------------------------
# Anti-fabrication provenance gate
# ---------------------------------------------------------------------------
#
# An identifier the assistant states to the user must have appeared somewhere the
# model legitimately saw it — a tool result, a system event, a user message, or a
# delegation handoff. One that appears nowhere in the whole conversation's tool
# I/O or user/system text is fabricated (the GLM-5.3-flash run invented a job id
# `a1b2c3d4`, a model `mixtral`, and a `0.92` score — the id was the clean tell).
#
# Deliberately conservative: only ID-shaped hex/UUID tokens are checked (not
# free-form metrics or model names, which cannot be flagged without false
# positives), a token is grounded by *substring* match (so an abbreviated prefix
# of a real UUID still passes), and bare hex runs must mix digits and letters (so
# pure-decimal counts and pure-hex words are never treated as identifiers). On a
# violation we force-correct once: make the model re-answer grounded; if it still
# can't, we relay its reply with an appended caveat naming the unverified id(s)
# rather than discard a possibly-correct answer.

# UUID, or any hex run of >=8 chars (job/run ids are UUIDs; models often surface a
# short prefix). Anchored so pure-prose words can't match (hex letters are a-f).
_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{8,}")


def _extract_id_candidates(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _ID_RE.findall(text.lower()):
        if "-" in tok:
            out.add(tok)  # dash-structured UUID — unambiguously an identifier
            continue
        # A bare hex run is only id-like if it mixes digits AND hex letters. This
        # excludes both pure-decimal counts ("12345678" records) and pure-hex
        # English words ("deadbeef", "facefeed") that would otherwise be
        # false-flagged as fabricated ids and cost the model a correction round.
        if any(c in "0123456789" for c in tok) and any(c in "abcdef" for c in tok):
            out.add(tok)
    return out


def _result_text(result: dict[str, Any]) -> str:
    """The user-facing assistant prose for a turn (what Studio renders)."""
    return "\n".join(
        str(p.get("text") or "")
        for p in result.get("parts", [])
        if p.get("type") == "text" and (p.get("text") or "").strip()
    )


async def _grounded_corpus(
    state: SessionState, cache: dict[str, list[dict[str, Any]]] | None = None
) -> str:
    """Everything the model legitimately saw this conversation, lowercased.

    Union over every role session of: user/system message text (user input,
    `[SYSTEM EVENT]`s, and the delegation handoff relayed as user text) plus every
    tool call's input and output. An id the assistant states that is a substring
    of this corpus is grounded; one that is not is fabricated.
    """
    sessions = {state.orchestrator_id}
    if state.subagent_id:
        sessions.add(state.subagent_id)
    sessions.update(state.completed_subagents.values())

    chunks: list[str] = []
    for sid in sessions:
        for msg in await _fetch_all_messages(sid, cache):
            role = (msg.get("info") or {}).get("role")
            for part in msg.get("parts") or []:
                ptype = part.get("type")
                if role == "user" and ptype == "text":
                    txt = str(part.get("text") or "")
                    # Skip the gate's own grounding correction: it NAMES the fabricated
                    # id(s), which would otherwise re-enter the corpus and let the model's
                    # repeat of the same id pass the recheck ([4]).
                    if txt.startswith("[GROUNDING VIOLATION"):
                        continue
                    chunks.append(txt)
                elif ptype == "tool":
                    chunks.append(str(_get_tool_input(part)))
                    part_state = part.get("state")
                    if isinstance(part_state, dict):
                        output = part_state.get("output")
                        if isinstance(output, str):
                            chunks.append(output)
    return "\n".join(chunks).lower()


# How many times to force-correct a reply that states a fabricated identifier
# before giving up and appending a caveat. Each attempt re-grounds against a
# freshly-fetched corpus (a correction may call a tool that surfaces the id for
# real), so a legitimately-recoverable turn self-heals on the first pass. Capped at
# 1: further retries rarely recover and multiply model round-trips per turn.
MAX_PROVENANCE_RETRIES = 1

# How many times _run_turn re-runs the model to clear a held committing card (the hard
# rail): one correction attempt, then let the turn through so a non-complying driver
# cannot loop.
MAX_INTERACTION_RERUNS = 1

# Marks a gate's visible caveat (provenance / job-claim warning) so _run_turn's client-contract
# restore keeps it in the turn result and the Studio client can surface it. See _caveat_part.
_CAVEAT_TAG = "amortized_caveat"


def _caveat_part(text: str) -> dict[str, Any]:
    """A gate's visible caveat as a tagged text part. The #505 restore in _run_turn drops all
    but the handler markers + phase backfill, so this tag is what keeps the caveat in the
    client-facing result; the Studio client reads the tag and appends the caveat text to its
    GET-derived content (the client renders session.text from GET, not turn.result)."""
    return {"type": "text", "text": text, _CAVEAT_TAG: True}


async def _ungrounded_ids(
    state: SessionState,
    result: dict[str, Any],
    cache: dict[str, list[dict[str, Any]]] | None = None,
) -> list[str]:
    """Identifier tokens in the reply that appear nowhere the model could have seen."""
    candidates = _extract_id_candidates(_result_text(result))
    if not candidates:
        return []
    corpus = await _grounded_corpus(state, cache)
    return sorted(c for c in candidates if c not in corpus)


async def _apply_provenance_gate(
    state: SessionState,
    result: dict[str, Any],
    body: MessageRequest,
    cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Block + force-correct fabricated identifiers before they reach the user."""
    try:
        violations = await _ungrounded_ids(state, result, cache)
        if not violations:
            return result

        active_id = state.subagent_id or state.orchestrator_id
        agent = state.subagent_target if state.subagent_id else "morty"
        current = result
        for attempt in range(1, MAX_PROVENANCE_RETRIES + 1):
            logger.warning(
                "Provenance gate: ungrounded identifier(s) %s session=%s agent=%s "
                "(correction %d/%d)",
                violations,
                active_id,
                agent,
                attempt,
                MAX_PROVENANCE_RETRIES,
            )
            correction = (
                "[GROUNDING VIOLATION — internal system check, not from the user]\n"
                "Your previous reply stated identifier(s) that appear in NO tool result, "
                f"system event, or user message in this conversation: {', '.join(violations)}.\n"
                "Do NOT invent job IDs, run IDs, dataset IDs, model names, or metrics. "
                "Re-send your reply to the user using ONLY values that came from a tool "
                "result in this session. If you do not have a real value, do not state "
                "one — call the appropriate tool to fetch it first, or tell the user you "
                "don't have it yet."
            )
            post = await _proxy_send_message(active_id, correction, agent=agent, model=body.model)
            # The correction adds new messages to the active session; drop its cached
            # copy so re-grounding fetches them (other sessions stay cached).
            if cache is not None:
                cache.pop(active_id, None)
            # Re-ground the model's ACTUAL corrected reply: the POST response carries only
            # step markers, so re-checking it would find no text and always "pass" — the
            # real reply text is in the GET parts. (A correction may also call a tool that
            # now surfaces the id legitimately.)
            refetched = await _fetch_all_assistant_parts(active_id)
            # Fall back to the POST parts if the fetch is empty (a transient GET failure,
            # or a test without a live opencode) so the recheck still sees the reply.
            current = {**post, "parts": refetched or post.get("parts") or []}
            violations = await _ungrounded_ids(state, current, cache)
            if not violations:
                return current

        # Exhausted the correction and the model still states an unverified id.
        # Rather than discard a possibly-correct reply (a tightened-but-imperfect
        # id detector can still false-positive), keep it and append a visible
        # caveat naming the unverified id(s) so the user is warned without losing
        # content.
        logger.warning(
            "Provenance gate: still ungrounded after %d correction(s) %s session=%s",
            MAX_PROVENANCE_RETRIES,
            violations,
            active_id,
        )
        parts = list(current.get("parts") or [])
        parts.append(
            _caveat_part(
                "\n\n⚠️ I couldn't verify the identifier(s) "
                f"{', '.join(violations)} against platform data in this "
                "session — treat them as unconfirmed until I fetch the real "
                "values."
            )
        )
        return {**current, "parts": parts}
    except Exception:
        logger.warning("Provenance gate failed (passing through)", exc_info=True)
        return result


# ---------------------------------------------------------------------------
# Job-claim verification gate
# ---------------------------------------------------------------------------
#
# The provenance gate above checks that an id the model states EXISTS in the
# conversation. This gate goes one level deeper: when the model reports a job to
# the user, it checks that the job's real DB record MATCHES the claim. Three general
# checks (not keyed to one symptom):
#   A. state truth    — "finished/ready/generated" => DB status must be succeeded.
#   B. flow attribution — "just ran/generated" for a job created in an EARLIER
#      conversation, with no matching validate_*_job this session and no reuse
#      disclosure => undisclosed stale reuse (caught the 200-requested /
#      500-record-stale-job substitution).
#   C. dispatch provenance — a training/eval dispatch THIS turn (a validate_*_job
#      tool call) whose parent_job_id is a dataset from an EARLIER conversation, not
#      run or disclosed as reused this session => undisclosed stale substitution,
#      caught at the point of action. Keyed on the tool call's structured
#      parent_job_id argument, not on prose, so phrasing cannot evade it.
#   E. premature submission — a validate_*_job card is rendered THIS turn (the job is
#      created only when the user clicks Confirm) yet the prose claims it is already
#      submitted/queued/running/generating => false, caught before it misleads the user
#      (the GLM-5.3-flash run said "the dataset generation is underway … job queued" off
#      a validate call, with a fabricated job id). Structural: validate_* call + claim.
# Same remediation shape as the provenance gate: force-correct once, else append a
# visible caveat rather than discard. Reuse is NOT blocked — only forced to be
# disclosed — so legitimate reuse (e.g. the eval flow) keeps working.

# These lexicons only decide WHETHER a claim is being made; the verdict always
# comes from the DB record, never from the text.
_DONE_CLAIM_RE = re.compile(
    r"\b(finished|complete|completed|ready|done|generated|produced|succeeded|trained)\b",
    re.I,
)
_FRESH_RUN_RE = re.compile(
    r"(now running|is running|are running|kicked off|just (ran|generated|finished|trained)"
    r"|run finished|generation (job|is|finished|complete)|is (generating|being generated)"
    r"|\bgenerated\b|\bproducing\b|spinning up)",
    re.I,
)
_REUSE_DISCLOSED_RE = re.compile(
    r"(reus|existing|already (generated|have|exists|ran)|previous|earlier"
    r"|from (a |an )?(prior|earlier|previous)|from before)",
    re.I,
)
# Job type ↔ its validate_*_job tool. The correspondence is owned once by
# _JOB_TYPE_FOR_CONFIRM in agent_confirm (the confirm cards ARE these tools); alias it here
# under validate-centric names so the fact lives in a single place and cannot drift.
_VALIDATE_TOOL_BY_TYPE = _CONFIRM_TOOL_FOR_JOB_TYPE
_TYPE_BY_VALIDATE_TOOL = _JOB_TYPE_FOR_CONFIRM

# Check E ("premature submit") — a validate_*_job tool call only RENDERS a confirmation
# card; the job is created solely when the user clicks Confirm on it (the frontend POSTs
# create_*). So on a turn that proposes such a card, prose asserting the job is already
# submitted / queued / running / generating is false — the job does not exist yet. This
# lexicon only decides WHETHER such a claim is made; the verdict is structural (a
# validate_* call is present in the same turn). "validate", "I'll build/set up", and plain
# future tense are deliberately excluded so legitimate "here's the card to confirm" prose
# never trips it.
_SUBMITTED_CLAIM_RE = re.compile(
    r"(underway|queued|submitted|dispatched|kicked off|spinning up"
    r"|now (generating|running)|has (started|begun)"
    r"|is (now )?(running|generating|live|being generated)"
    r"|generation (has|is) (started|begun|underway|running)"
    r"|job (is|was|has been) (created|submitted|queued|running|started|live))",
    re.I,
)

# Check C — downstream dispatch tools Morty calls to commit a training/eval run onto a
# parent dataset, for the stale-reuse provenance check. The call carries parent_job_id
# as a structured argument, verified from the parent's DB record with no prose parsing.
# SDG is a dataset *producer*, not a consumer of one, so it is excluded. (Check D, the
# field-driven stage-gate, keys on the full _TYPE_BY_VALIDATE_TOOL — every committing
# dispatch — so an incidental job id on a read-only tool is never treated as a dispatch.)
_DISPATCH_PARENT_TYPE = {t: jt for t, jt in _TYPE_BY_VALIDATE_TOOL.items() if jt != "sdg"}


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _sentences_with(text: str, token: str) -> str:
    """The sentence-ish chunks of `text` that mention `token`, so a claim about one
    job isn't attributed to another mentioned in the same reply."""
    tl = token.lower()
    hits = [c for c in re.split(r"[\n.!?;]+", text) if tl in c.lower()]
    return " ".join(hits)


async def _fetch_job_by_token(
    token: str, job_cache: dict[str, dict[str, Any] | None] | None = None
) -> dict[str, Any] | None:
    """The DB job a stated id-token resolves to (by id prefix), or None. Best-effort
    — any DB issue returns None so the gate simply doesn't fire."""
    if job_cache is not None and token in job_cache:
        return job_cache[token]
    job: dict[str, Any] | None = None
    try:
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository

        async with get_pool().acquire() as conn:
            job = await Repository(conn).find_job_by_id_prefix(token)
    except Exception:
        job = None
    if job_cache is not None:
        job_cache[token] = job
    return job


def _session_ids(state: SessionState) -> set[str]:
    """Every opencode session making up this conversation — the orchestrator plus
    the active and completed subagents."""
    sessions = {state.orchestrator_id}
    if state.subagent_id:
        sessions.add(state.subagent_id)
    sessions.update(state.completed_subagents.values())
    return sessions


async def _session_has_validate(
    state: SessionState,
    job_type: str,
    cache: dict[str, list[dict[str, Any]]] | None,
) -> bool:
    """Whether a validate_*_job was called anywhere in the conversation so far — the
    structural signal that a fresh job really was set up this session. Scoped to the
    type's validate tool when known; an empty/unknown type matches ANY validate."""
    tool = _VALIDATE_TOOL_BY_TYPE.get(job_type)
    wanted = {tool} if tool else set(_VALIDATE_TOOL_BY_TYPE.values())
    for sid in _session_ids(state):
        for msg in await _fetch_all_messages(sid, cache):
            for part in msg.get("parts") or []:
                if part.get("type") == "tool" and _tool_name(part) in wanted:
                    return True
    return False


# Job types whose subagent can't do anything without a dataset — so a delegation to
# one when THIS conversation has no data path is premature.
_DATA_DEPENDENT_TARGETS = {"training": "train on", "eval": "evaluate on"}

# Tools that mean the user has engaged a SPECIFIC existing dataset (inspecting its
# samples or splitting it) — a de-facto choice of existing data, so the fork is moot.
# `list_datasets` is deliberately excluded: merely browsing the catalog is not a
# choice, so the existing-vs-generate fork should still be offered.
_DATASET_TOOLS = {"get_dataset", "get_dataset_samples", "split_dataset"}


async def _conversation_has_dataset(
    state: SessionState,
    cache: dict[str, list[dict[str, Any]]] | None,
) -> bool:
    """Whether THIS conversation already has, or has engaged, a dataset — so a
    training/eval delegation is NOT premature. Both signals are structural tool
    facts (conversation-scoped, never a lifetime "has this user ever" query):

    1. An SDG job was set up here (`validate_sdg_job`) — fresh data in flight.
    2. A dataset tool was touched here (`get_dataset`/`get_dataset_samples`/
       `split_dataset`) — the user engaged a SPECIFIC existing dataset.

    Deliberately NOT inferred from message text. The orchestrator's handoff lists
    the available datasets as platform state, and that handoff is relayed into the
    subagent session as a user-role message — so scanning text for dataset ids
    conflates "available/listed" with "chosen" and suppresses the fork when the
    user has not actually decided. A real choice always surfaces as a tool fact:
    resolving even a fuzzy name ("the whimsical lark data") routes through
    get_dataset, which trips signal 2 regardless of how it was phrased.
    """
    if await _session_has_validate(state, "sdg", cache):
        return True
    for sid in _session_ids(state):
        for msg in await _fetch_all_messages(sid, cache):
            for part in msg.get("parts") or []:
                if part.get("type") == "tool" and _tool_name(part) in _DATASET_TOOLS:
                    return True
    return False


async def _no_data_advisory(state: SessionState, target: str) -> str | None:
    """Advisory to prepend to a training/eval handoff when THIS conversation has no
    dataset selected yet — directing the subagent to let the user CHOOSE between an
    existing dataset and a fresh one, rather than silently defaulting to generation.

    The orchestrator routes to training/eval off the user's verb ("train"), but
    those jobs can't run without data, and the user must get to decide how that data
    is provided. Rather than let the subagent spin up and jump straight to SDG, the
    proxy injects the fork at the delegation boundary — the earliest point the
    backend can act, since the orchestrator's choice is model-internal.

    Fail-safe: fires only when none of the conversation-scoped dataset signals hold
    (see _conversation_has_dataset) — i.e. the user has NOT already chosen (no SDG in
    flight, no existing dataset pointed to). Any uncertainty (a signal present, or a
    fetch error) returns None, leaving the handoff unchanged — so once a choice is
    made the fork isn't re-asked, and upload/split/train-on-existing flows are
    untouched. It never blocks the delegation; it states the fork to present.
    """
    if target not in _DATA_DEPENDENT_TARGETS:
        return None
    try:
        if await _conversation_has_dataset(state, {}):
            return None
    except Exception:
        return None
    verb = _DATA_DEPENDENT_TARGETS[target]
    return (
        "[DATA AVAILABILITY]\n"
        "No dataset has been selected in this conversation yet. Before setting up the"
        f" {target} job, let the user choose how to provide the data — present two"
        " options with present_options: (1) use an existing dataset (help them pick"
        " one), or (2) generate a fresh dataset with SDG. Do NOT assume: a"
        f" {target} job needs data to {verb}, so don't create or confirm it, and"
        " don't start generating, until the user has chosen and the dataset exists."
    )


async def _job_claim_violations(
    state: SessionState,
    result: dict[str, Any],
    cache: dict[str, list[dict[str, Any]]] | None,
    job_cache: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Job claims in the reply that the job's real DB record contradicts."""
    text = _result_text(result)
    if not text.strip():
        return []
    violations: list[dict[str, Any]] = []
    tokens = _extract_id_candidates(text)
    saw_real_types: set[str] = set()
    for token in sorted(tokens):
        job = await _fetch_job_by_token(token, job_cache)
        if not job:
            continue  # not a real job (or ambiguous) — provenance gate covers fabrication
        saw_real_types.add(str(job.get("type") or ""))
        window = _sentences_with(text, token)
        if not window:
            continue
        status = str(job.get("status") or "")
        jtype = str(job.get("type") or "")
        cfg = job.get("config")
        num = cfg.get("num_records") if isinstance(cfg, dict) else None
        if _DONE_CLAIM_RE.search(window) and status != "succeeded":
            violations.append(
                {"token": token, "kind": "state", "status": status, "type": jtype, "num": num}
            )
            continue  # a wrong-status job is reported once; don't also flag reuse
        if (
            _FRESH_RUN_RE.search(window)
            and not _REUSE_DISCLOSED_RE.search(window)
            and not await _session_has_validate(state, jtype, cache)
        ):
            created = _as_utc(job.get("created_at"))
            session_start = _as_utc(state.created_at)
            if created and session_start and created < session_start:
                violations.append(
                    {"token": token, "kind": "reuse", "status": status, "type": jtype, "num": num}
                )
    flagged = {v["token"] for v in violations if v.get("token")}
    # Check D — the stage-gate: a training/eval dispatch this turn whose upstream job
    # isn't 'succeeded' yet. Structural (tool-arg + DB status), fires regardless of
    # prose. Run before Check C so "wait for it to finish" wins over "disclose reuse"
    # when an upstream is both in-flight and old. Dedup by upstream prefix.
    for v in await _not_ready_violations(state, result, job_cache):
        if v["token"] not in flagged:
            violations.append(v)
            flagged.add(v["token"])
    # Check C — a training/eval dispatch this turn onto a stale, undisclosed parent
    # dataset. Keyed on the dispatch tool call's parent_job_id, independent of prose,
    # so it fires even when the reply names no id. Dedup against A/B/D by parent prefix.
    for v in await _dispatch_provenance_violations(state, result, text, cache, job_cache):
        if v["token"] not in flagged:
            violations.append(v)
            flagged.add(v["token"])
    # Check E — premature submission. This turn renders a validate_*_job confirm card
    # (job not created until the user clicks Confirm), yet the prose claims the job is
    # already submitted/queued/running/generating. Structural: the turn's own validate_*
    # call + a submission claim in prose; no id needed. Suppressed PER JOB TYPE when the
    # reply names a REAL job of THAT type — then the claim is attributable to it and
    # Checks A-D own it (and a preview of the same type, legitimately in flight before the
    # full-run card, won't false-trip). Scoped to the card's own type so naming an
    # unrelated real job (e.g. a prior training run) can't launder a false "the SDG run is
    # underway" claim about a just-rendered SDG card.
    violations.extend(_premature_submit_violations(result, text, saw_real_types))
    return violations


def _premature_submit_violations(
    result: dict[str, Any], text: str, saw_real_types: set[str]
) -> list[dict[str, Any]]:
    """Check E — see the call site. Returns one violation per validate_* job type whose
    confirm card is rendered this turn while the prose claims the job already runs, unless
    a REAL job of that same type is named in the reply (then Checks A-D own the claim)."""
    if not _SUBMITTED_CLAIM_RE.search(text):
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for part in result.get("parts") or []:
        if part.get("type") != "tool":
            continue
        jtype = _TYPE_BY_VALIDATE_TOOL.get(_tool_name(part))
        if jtype and jtype not in seen and jtype not in saw_real_types:
            seen.add(jtype)
            out.append({"token": "", "kind": "premature_submit", "type": jtype})
    return out


async def _dispatch_provenance_violations(
    state: SessionState,
    result: dict[str, Any],
    text: str,
    cache: dict[str, list[dict[str, Any]]] | None,
    job_cache: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """A training/eval dispatch in this turn whose parent dataset is a succeeded job
    from an EARLIER conversation, with no SDG validated this session and no reuse
    disclosure — undisclosed stale substitution, caught at the point of action. The
    verdict is wholly structural: the dispatch tool call's parent_job_id argument, the
    parent's created_at vs the session start, and the presence of a validate_sdg_job
    in the corpus. Only reuse disclosure is read from prose (a communication act with
    no structural proxy), and matching it merely suppresses the flag — fail-safe."""
    violations: list[dict[str, Any]] = []
    session_start = _as_utc(state.created_at)
    if session_start is None:
        return violations
    reuse_disclosed = bool(_REUSE_DISCLOSED_RE.search(text))
    for part in result.get("parts") or []:
        if part.get("type") != "tool":
            continue
        downstream = _DISPATCH_PARENT_TYPE.get(_tool_name(part))
        if downstream is None:
            continue
        parent_id = str(_get_tool_input(part).get("parent_job_id") or "")
        if not parent_id:
            continue  # data_path / data_run_id dispatch — not an SDG chain
        if reuse_disclosed:
            continue  # reuse acknowledged in the reply — allowed
        parent = await _fetch_job_by_token(parent_id, job_cache)
        if not parent:
            continue
        created = _as_utc(parent.get("created_at"))
        if created is None or created >= session_start:
            continue  # produced this conversation — a fresh, legitimate chain
        if await _session_has_validate(state, "sdg", cache):
            continue  # a fresh SDG really was set up this session
        cfg = parent.get("config")
        num = cfg.get("num_records") if isinstance(cfg, dict) else None
        violations.append(
            {
                "token": parent_id[:8],
                "kind": "dispatch",
                "status": str(parent.get("status") or ""),
                "type": downstream,
                "num": num,
            }
        )
    return violations


async def _not_ready_violations(
    state: SessionState,
    result: dict[str, Any],
    job_cache: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Check D — the stage-gate. A committing dispatch in this turn (_TYPE_BY_VALIDATE_TOOL)
    whose upstream job (the SDG dataset, or the model under eval) is not yet 'succeeded'
    — i.e. the workflow is being advanced before its prerequisite finished. Field-driven
    and purely structural: for each dispatch tool, every upstream-job reference its input
    carries (lineage.GATED_FIELDS) and that job's DB status; no prose. Each dispatch is
    gated on exactly the prerequisites it names, so extra work elsewhere is never a
    concern. Blocks forward movement so Morty reports the true status and waits, instead
    of confirming/creating a job that can't run yet. Independent of job age (unlike the
    stale-reuse 'dispatch' check) — it fires on any non-succeeded upstream."""
    violations: list[dict[str, Any]] = []
    for part in result.get("parts") or []:
        if part.get("type") != "tool":
            continue
        dtype = _TYPE_BY_VALIDATE_TOOL.get(_tool_name(part))
        if dtype is None:
            continue  # not a committing dispatch tool
        inp = _get_tool_input(part)
        for _field, up_id, label in lineage.present_gated_fields(inp):
            up = await _fetch_job_by_token(up_id, job_cache)
            if not up:
                continue  # unresolvable — not a stage-gate concern (fabrication gate owns it)
            status = str(up.get("status") or "")
            if status == lineage.READY_STATUS:
                continue
            violations.append(
                {
                    "token": up_id[:8],
                    "kind": "not_ready",
                    "status": status,
                    "type": dtype,
                    "label": label,
                }
            )
    return violations


async def _apply_job_claim_gate(
    state: SessionState,
    result: dict[str, Any],
    body: MessageRequest,
    cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Verify + force-correct false job-completion / stale-reuse claims."""
    try:
        job_cache: dict[str, dict[str, Any] | None] = {}
        violations = await _job_claim_violations(state, result, cache, job_cache)
        if not violations:
            return result

        active_id = state.subagent_id or state.orchestrator_id
        agent = state.subagent_target if state.subagent_id else "morty"
        current = result
        for attempt in range(1, MAX_PROVENANCE_RETRIES + 1):
            logger.warning(
                "Job-claim gate: contradicted claim(s) %s session=%s agent=%s (correction %d/%d)",
                violations,
                active_id,
                agent,
                attempt,
                MAX_PROVENANCE_RETRIES,
            )
            post = await _proxy_send_message(
                active_id, _job_claim_correction(violations), agent=agent, model=body.model
            )
            if cache is not None:
                cache.pop(active_id, None)
            # Re-check the model's ACTUAL corrected reply (POST = step markers only; the
            # real text is in the GET parts), else the recheck sees no claim and "passes".
            refetched = await _fetch_all_assistant_parts(active_id)
            # Fall back to the POST parts if the fetch is empty (a transient GET failure,
            # or a test without a live opencode) so the recheck still sees the reply.
            current = {**post, "parts": refetched or post.get("parts") or []}
            violations = await _job_claim_violations(state, current, cache, job_cache)
            if not violations:
                return current

        logger.warning(
            "Job-claim gate: still contradicted after %d correction(s) %s session=%s",
            MAX_PROVENANCE_RETRIES,
            violations,
            active_id,
        )
        parts = list(current.get("parts") or [])
        parts.append(_caveat_part(_job_claim_caveat(violations)))
        return {**current, "parts": parts}
    except Exception:
        logger.warning("Job-claim gate failed (passing through)", exc_info=True)
        return result


async def _record_turn_metrics(
    state: SessionState,
    turn: TurnState,
    session_id: str,
    turn_id: str,
) -> None:
    """Append one monitor ``turn`` record. Attributes every opencode assistant
    message to exactly one proxy turn (dedup by id across all of the
    conversation's sessions), which fixes both the delegation undercount (a
    single active-session read misses subagents) and the within-turn undercount
    (multi-step tool loops span several assistant messages).
    """
    try:
        role_by_session: dict[str, str] = {state.orchestrator_id: "orchestrator"}
        if state.subagent_id and state.subagent_target:
            role_by_session[state.subagent_id] = state.subagent_target
        for target, sid in state.completed_subagents.items():
            role_by_session.setdefault(sid, target)

        new_msgs: list[tuple[int, str, dict[str, Any]]] = []
        for sid, role in role_by_session.items():
            for msg in await _fetch_all_messages(sid):
                info = msg.get("info") or {}
                if info.get("role") != "assistant":
                    continue
                mid = info.get("id")
                if not mid or mid in state.seen_message_ids:
                    continue
                state.seen_message_ids.add(mid)
                created = _coerce_int((info.get("time") or {}).get("created"))
                new_msgs.append((created, role, msg))
        new_msgs.sort(key=lambda item: item[0])

        tokens = {"input": 0, "output": 0, "reasoning": 0, "cache": 0}
        tokens_by_role: dict[str, int] = {}
        cost = 0.0
        model_id: str | None = None
        provider_id: str | None = None
        tool_calls: list[dict[str, Any]] = []
        # Assistant prose, in chronological message order. Lets an offline LLM
        # judge check grounding/fabrication (claim-vs-tool-output) from the log
        # alone, instead of a human reading the live transcript.
        texts: list[dict[str, Any]] = []

        for _created, role, msg in new_msgs:
            info = msg.get("info") or {}
            msg_tokens = info.get("tokens") or {}
            msg_total = 0
            for key in ("input", "output", "reasoning", "cache"):
                val = _coerce_int(msg_tokens.get(key))
                tokens[key] += val
                msg_total += val
            tokens_by_role[role] = tokens_by_role.get(role, 0) + msg_total
            msg_cost = info.get("cost")
            if isinstance(msg_cost, (int, float)) and not isinstance(msg_cost, bool):
                cost += float(msg_cost)
            if info.get("modelID"):
                model_id = info.get("modelID")
            if info.get("providerID"):
                provider_id = info.get("providerID")
            msg_parts = msg.get("parts") or []
            # Per-message signals for `clean_delegation`: whether this assistant
            # message emitted any text and how many tool parts it carried. Phase 2
            # requires the delegating message to be ONLY the delegate call.
            msg_text = "\n".join(
                p["text"].strip()
                for p in msg_parts
                if p.get("type") == "text" and (p.get("text") or "").strip()
            )
            msg_has_text = bool(msg_text)
            msg_tool_count = sum(1 for p in msg_parts if p.get("type") == "tool")
            if msg_has_text:
                texts.append({"role": role, "text": msg_text})
            for part in msg_parts:
                if part.get("type") != "tool":
                    continue
                name = _tool_name(part)
                entry: dict[str, Any] = {"tool": name, "role": role}
                part_state = part.get("state")
                if isinstance(part_state, dict):
                    if part_state.get("status"):
                        entry["status"] = part_state.get("status")
                    output = part_state.get("output")
                    if isinstance(output, str) and output:
                        entry["output"] = output
                if name == "delegate_to_subagent":
                    entry["target"] = _get_tool_input(part).get("target")
                    # solo = the delegating message was the delegate call alone.
                    entry["solo"] = (not msg_has_text) and msg_tool_count == 1
                elif name.startswith("validate_"):
                    # Capture pipeline-wiring inputs so chaining (SDG->training->eval
                    # via parent_job_id) is checkable offline. Config args aren't
                    # otherwise logged; only record the fields when present/non-empty.
                    inp = _get_tool_input(part)
                    # `training_job_id` on an eval config = the subject under eval is
                    # a tuned model (not a base/gateway model) — lets the monitor gate
                    # "eval set must mirror the training pipeline" checks on that case.
                    for key in (
                        "parent_job_id",
                        "data_run_id",
                        "eval_data_run_id",
                        "training_job_id",
                    ):
                        cfg_val = inp.get(key)
                        if cfg_val:
                            entry[key] = cfg_val
                    # `mode` (preview/create) makes the SDG preview-before-create
                    # ordering checkable offline.
                    mode = inp.get("mode")
                    if mode:
                        entry["mode"] = mode
                    # Eval requires an explicit judge; record its model only
                    # (never base_url/api_key) so "judge set" is checkable.
                    judge = inp.get("judge")
                    if isinstance(judge, dict) and judge.get("model"):
                        entry["judge"] = judge["model"]
                elif name == "signal_phase":
                    # Capture phase/step so the UI progress signal is checkable
                    # offline (e.g. the eval stage must signal phase=eval, not
                    # a stale training phase).
                    inp = _get_tool_input(part)
                    phase = inp.get("phase")
                    if phase:
                        entry["phase"] = phase
                    step = inp.get("step")
                    if step:
                        entry["step"] = step
                elif name == "present_options":
                    # Capture the question and option titles so "a real choice
                    # was offered" (e.g. the training base-model list) is
                    # checkable offline — the option content is not otherwise
                    # logged, so without this the monitor cannot tell whether
                    # Morty presented options or auto-picked.
                    inp = _get_tool_input(part)
                    question = inp.get("question")
                    if question:
                        entry["question"] = question
                    options = inp.get("options")
                    if isinstance(options, list):
                        titles = [
                            opt["title"]
                            for opt in options
                            if isinstance(opt, dict) and opt.get("title")
                        ]
                        if titles:
                            entry["options"] = titles
                tool_calls.append(entry)

        started = turn.started_at
        finished = turn.finished_at
        duration_ms = (
            int((finished - started).total_seconds() * 1000)
            if started is not None and finished is not None
            else None
        )
        monitor_log.append_record(
            session_id,
            {
                "kind": "turn",
                "session_id": session_id,
                "conversation_root": state.orchestrator_id,
                "turn_id": turn_id,
                "role": state.subagent_target if state.subagent_id else "orchestrator",
                "model": model_id,
                "provider": provider_id,
                "tokens": {**tokens, "total": sum(tokens.values())},
                "tokens_by_role": tokens_by_role,
                "cost": cost,
                "tool_calls": tool_calls,
                "texts": texts,
                "started_at": started.isoformat() if started else None,
                "finished_at": finished.isoformat() if finished else None,
                "duration_ms": duration_ms,
            },
        )
    except Exception:
        logger.warning("monitor: failed to record turn metrics turn=%s", turn_id, exc_info=True)


# ---------------------------------------------------------------------------
# Upstream helpers
# ---------------------------------------------------------------------------


def _opencode_url() -> str:
    return settings.agent_upstream_url.rstrip("/")


async def _proxy_get(
    target_id: str,
    path: str,
    empty: dict[str, Any],
    session_id: str,
) -> Any:
    try:
        resp = await _client().get(f"{_opencode_url()}/session/{target_id}/{path}", timeout=10.0)
        if resp.status_code == 404:
            return empty
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "application/json" not in content_type:
            return empty
        if not resp.content or not resp.content.strip():
            return empty
        return resp.json()
    except (httpx.HTTPStatusError, ValueError):
        logger.exception("Upstream error on GET %s: session=%s", path, session_id)
        return empty
    except httpx.HTTPError:
        logger.warning("Upstream unreachable on GET %s: session=%s", path, session_id)
        return empty


async def _proxy_create_session() -> str:
    resp = await _client().post(f"{_opencode_url()}/session", timeout=10.0)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return str(data["id"])


class AgentTurnError(Exception):
    """A provider/model error during a turn, already mapped to a user-facing message.

    ``code`` is a stable, client-switchable identifier (see ``_classify_provider_error``):
    ``tool_calling`` | ``auth`` | ``model_unavailable`` | ``rate_limit`` | ``unknown``.
    """

    def __init__(self, message: str, status: int = 400, code: str = "unknown") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _classify_provider_error(raw: str, name: str = "") -> tuple[str, str]:
    """Classify a raw provider/opencode error into a ``(code, message)`` pair.

    ``code`` is a stable identifier the client switches on *exactly* (never by re-parsing
    the prose); ``message`` is the short chat-facing text. Keyed on OpenCode's structured
    error ``name`` (e.g. ``ProviderAuthError``, ``ProviderModelNotFoundError``) where
    available, with message-text heuristics as a fallback for providers that omit it.
    """
    auth_msg = "That model isn't available with your API key or project (auth/permission)."
    missing_msg = "That model isn't available on this provider/endpoint."
    limited_msg = "The provider is rate-limiting or out of quota — try again shortly."

    # Tier 1: trust OpenCode's structured error ``name`` (unambiguous) over message text, so e.g.
    # a ``ProviderAuthError`` whose message mentions "not found" still classifies as ``auth``.
    structured = {
        "providerautherror": ("auth", auth_msg),
        "providermodelnotfounderror": ("model_unavailable", missing_msg),
    }
    name_l = name.lower()
    for key, result in structured.items():
        if key in name_l:
            return result

    # Tier 2: message-text heuristics for providers that omit a structured name. Check the more
    # specific missing-model signal before auth so "permission denied: model not found" maps to
    # ``model_unavailable`` rather than ``auth``.
    text = raw.lower()
    if "tool" in text and ("choice" in text or "parser" in text or "enable-auto-tool" in text):
        return (
            "tool_calling",
            "This model's server doesn't have tool-calling enabled, which Morty needs. "
            "Pick a tool-calling-capable model.",
        )
    missing = ("not found", "does not exist", "no such model")
    auth = ("unauthorized", "forbidden", "permission", "invalid api key")
    limited = ("rate limit", "too many requests", "quota")
    if any(s in text for s in missing):
        return ("model_unavailable", missing_msg)
    if any(s in text for s in auth):
        return ("auth", auth_msg)
    if any(s in text for s in limited):
        return ("rate_limit", limited_msg)
    return ("unknown", (raw or "").strip()[:180] or "The model provider returned an error.")


def _friendly_provider_error(raw: str, name: str = "") -> str:
    """Back-compat shim: the user-facing message for a raw provider error.

    Prefer ``_classify_provider_error`` when you also need the stable ``code``.
    """
    return _classify_provider_error(raw, name)[1]


async def _proxy_send_message(
    session_id: str,
    text: str,
    agent: str | None = None,
    model: MessageModel | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
    if agent:
        payload["agent"] = agent
    if model:
        payload["model"] = model.model_dump(exclude_none=True)
    resp = await _client().post(
        f"{_opencode_url()}/session/{session_id}/message",
        json=payload,
    )
    resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    # OpenCode returns provider/generation errors as HTTP 200 with the error on the assistant
    # message (info.error) — surface it, else the turn looks blank/successful to the user.
    info = result.get("info")
    err = info.get("error") if isinstance(info, dict) else None
    if isinstance(err, dict):
        data = err.get("data")
        data = data if isinstance(data, dict) else {}
        raw = str(data.get("message") or "")
        code, message = _classify_provider_error(raw, str(err.get("name") or ""))
        raise AgentTurnError(message, code=code)
    return result


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MessagePart(BaseModel):
    type: str
    text: str | None = None


class MessageModel(BaseModel):
    providerID: str | None = None  # noqa: N815
    modelID: str | None = None  # noqa: N815


class MessageRequest(BaseModel):
    agent: str | None = None
    parts: list[MessagePart]
    model: MessageModel | None = None
    # Set by the client to mark a non-user, system-generated turn (e.g.
    # "job_complete" when a job-monitor card fires). The server uses it to steer
    # the turn — e.g. tell an active subagent to hand back instead of continuing.
    event: str | None = None
    # Structured outcome for an `event` turn (e.g. "succeeded"/"failed" for a
    # job_complete). Read instead of sniffing the display text for a "status:
    # succeeded" substring, which silently broke if the card's wording changed.
    outcome: str | None = None


def _extract_user_text(body: MessageRequest) -> str:
    for part in body.parts:
        if part.type == "text" and part.text:
            return part.text
    return ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/session")
async def create_session() -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    try:
        opencode_id = await _proxy_create_session()
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot connect to agent service") from None
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Agent service timed out") from None
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Agent service error: {exc}") from None
    _sessions[session_id] = SessionState(orchestrator_id=opencode_id)
    return {"id": session_id}


# ---------------------------------------------------------------------------
@router.get("/session/{session_id}/message")
async def get_session_messages(session_id: str) -> Any:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")
    target_id = state.subagent_id or state.orchestrator_id
    messages = await _proxy_get(target_id, "message", {"info": {}, "parts": []}, session_id)
    # Drop the cards the interaction gate stripped from the turn result FIRST, so the UI
    # (which re-fetches this raw history) doesn't re-render a card the gate removed — and so
    # the prompt-bind below only ever sees the LIVE parts. Binding before filtering made the
    # bind pass walk suppressed parts too: a stripped show_prompt could feed the approved
    # prompt set, and an errored validate card (now suppressed per the gate fix) could be
    # touched. With the errored card dropped here first, binding lands only on the live one.
    if isinstance(messages, list) and state.suppressed_call_ids:
        for msg in messages:
            parts = msg.get("parts") if isinstance(msg, dict) else None
            if isinstance(parts, list):
                msg["parts"] = [
                    p for p in parts if str(p.get("callID")) not in state.suppressed_call_ids
                ]
    try:
        if isinstance(messages, list):
            _bind_sdg_prompt_to_approved(messages)
    except Exception:
        logger.warning("SDG prompt-bind failed (passing through)", exc_info=True)
    return messages


def _evict_finished_turns(state: SessionState) -> None:
    """Bound per-session turn tracking without dropping a result the client hasn't read.

    A finished turn is evicted only once the client has polled it (``consumed``) or
    its result has gone stale (older than ``UNPOLLED_TURN_TTL``) — so a finished
    result is never discarded merely because newer turns arrived. Active turns are
    never evicted (their background result would be lost; the task is also anchored
    in ``_background_tasks``). ``MAX_TURNS_HARD`` is a last-resort ceiling that drops
    the oldest finished turns regardless, bounding memory against a client that never
    polls.
    """
    now = datetime.now(UTC)

    # Soft cap: evict oldest finished turns that are safe to drop — already polled
    # (consumed) or stale past the TTL. Never drop active or fresh-unpolled results.
    over = len(state.turn_order) - MAX_TURNS_TRACKED
    if over > 0:
        keep: list[str] = []
        for tid in state.turn_order:
            turn = state.turns.get(tid)
            safe = turn is None or (
                not turn.active
                and (
                    turn.consumed
                    or (turn.finished_at is not None and now - turn.finished_at > UNPOLLED_TURN_TTL)
                )
            )
            if over > 0 and safe:
                state.turns.pop(tid, None)
                over -= 1
            else:
                keep.append(tid)
        state.turn_order = keep

    # Hard ceiling: bound memory even if results are still unpolled + fresh (a client
    # that never polls). Drop oldest finished turns regardless of consumed/TTL.
    over_hard = len(state.turn_order) - MAX_TURNS_HARD
    if over_hard > 0:
        keep_hard: list[str] = []
        for tid in state.turn_order:
            turn = state.turns.get(tid)
            if over_hard > 0 and (turn is None or not turn.active):
                state.turns.pop(tid, None)
                over_hard -= 1
            else:
                keep_hard.append(tid)
        state.turn_order = keep_hard


@router.post("/session/{session_id}/message")
async def send_message(session_id: str, body: MessageRequest) -> dict[str, Any]:
    """Accept a message and run the turn in the background.

    Returns immediately with a ``turn_id``; the caller polls
    ``GET /session/{id}/turn/{turn_id}`` for the result. A turn can run for many
    seconds (orchestrator + subagent), which would otherwise exceed intermediate
    proxy timeouts (e.g. the dashboard embed) and 504.
    """
    user_text = _extract_user_text(body)
    if not user_text:
        raise HTTPException(status_code=400, detail="no text part in message")

    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")

    state.last_activity = datetime.now(UTC)

    # Bound pending work per session: active turns are never evicted, so without an
    # admission limit a client could queue unboundedly (state.turns / _background_tasks).
    if sum(1 for t in state.turns.values() if t.active) >= MAX_ACTIVE_TURNS:
        raise HTTPException(
            status_code=429,
            detail="too many pending turns for this session; retry shortly",
        )

    turn_id = str(uuid.uuid4())
    turn = TurnState()
    state.turns[turn_id] = turn
    state.turn_order.append(turn_id)
    _evict_finished_turns(state)
    task = asyncio.create_task(_run_turn(state, session_id, turn_id, user_text, body))
    turn.task = task
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"turn_id": turn_id, "status": "processing"}


async def _run_turn(
    state: SessionState,
    session_id: str,
    turn_id: str,
    user_text: str,
    body: MessageRequest,
) -> None:
    turn = state.turns.get(turn_id)
    # One message-fetch cache per turn, shared by the provenance gate and the metrics
    # recorder so they don't each re-fetch every session over HTTP.
    msg_cache: dict[str, list[dict[str, Any]]] = {}
    try:
        async with state.lock:
            if state.subagent_id:
                result = await _handle_subagent_message(state, session_id, user_text, body)
            else:
                result = await _handle_orchestrator_message(state, session_id, user_text, body)
            # The opencode POST response carries only step markers; the real tool parts
            # (cards, options, show_prompt, signal_phase) live in the GET messages. Every
            # gate below keys off these parts, so make the result carry the ACTIVE
            # session's current-turn parts — otherwise the structural gates scan
            # step-markers, never see a card, and silently no-op (they only ever "worked"
            # in tests, which hand them real parts).
            active_id = state.subagent_id or state.orchestrator_id
            # The gates below INSPECT the active session's real current-turn parts to make
            # decisions (hold a premature card, catch a fabricated job claim); they do not
            # render anything — the studio client re-derives every card/option/prompt from
            # its OWN GET /message, and corrections are posted to the session. So the fetched
            # parts must NOT leak into the client-facing turn.result: the client seeds its
            # tool-dedup from turn.result's parts and would drop the GET's card as a
            # duplicate (issue #505). Snapshot the handler's POST markers for the client,
            # feed the gates the real parts, then restore the client contract after the gates.
            client_parts = result.get("parts") or []
            fetched = _strip_internal_tools(await _fetch_all_assistant_parts(active_id))
            # Only replace on a successful fetch — a transient GET failure returns [],
            # and blanking the parts would needlessly no-op every gate below for that turn
            # (the client re-derives content from its own GET regardless).
            if fetched:
                result["parts"] = fetched
            result = await _apply_provenance_gate(state, result, body, msg_cache)
            result = await _apply_job_claim_gate(state, result, body, msg_cache)
            # Backfill the UI phase uniformly for every path (subagent completion,
            # subagent help hand-back, orchestrator) — not just the one return inside
            # the subagent handler — so the progress bar never strands on a stale
            # phase on a hand-back/delegation turn.
            result = _ensure_phase_signal(state, result)
            result = _enforce_single_interaction(result, state)
            # Hard rail: if the gate held a committing card (unmet precondition), correct
            # the model NOW and re-render, so the premature card never reaches the client
            # this turn — the stale turn falls behind the steer (a user-role message the
            # client never shows), exactly like the provenance gate. Bounded so a driver
            # that keeps jumping the gun can't loop.
            reruns = 0
            while state.pending_rerun_steer and reruns < MAX_INTERACTION_RERUNS:
                reruns += 1
                steer = state.pending_rerun_steer
                state.pending_rerun_steer = None
                steer_agent = state.subagent_target if state.subagent_id else "morty"
                logger.info(
                    "Interaction gate: held a committing card; re-running the model to "
                    "clear it (session=%s agent=%s)",
                    session_id,
                    steer_agent,
                )
                await _proxy_send_message(active_id, steer, agent=steer_agent, model=body.model)
                refetched = _strip_internal_tools(await _fetch_all_assistant_parts(active_id))
                if refetched:
                    result["parts"] = refetched
                result = await _apply_provenance_gate(state, result, body, msg_cache)
                result = await _apply_job_claim_gate(state, result, body, msg_cache)
                result = _ensure_phase_signal(state, result)
                result = _enforce_single_interaction(result, state)
            # Exhausted (the driver re-emitted the premature card despite the correction).
            # Don't silently drop the steer — that dead-ends: the card is held yet the model
            # is never told why. Defer it as a soft interaction steer so the model receives
            # the correction on its NEXT turn and can render the prerequisite step first.
            if state.pending_rerun_steer:
                state.pending_interaction_steer = state.pending_rerun_steer
                state.pending_rerun_steer = None
            # Keep any gate caveat (the provenance/job-claim "couldn't verify / still
            # contradicted" warning) in the client-facing result — it is a tagged text part the
            # Studio client surfaces separately (#498 reads the tag and appends it to the GET
            # content). The restore below otherwise drops everything but markers, so a caveat
            # left out here would never reach the user.
            caveat_parts = [p for p in (result.get("parts") or []) if p.get(_CAVEAT_TAG)]
            # Restore the client contract (see the fetch comment above): ship the handler's POST
            # markers, the gate-synthesized phase backfill, and any caveat text — the parts the
            # client cannot otherwise get from its own GET. Every card/option/prompt the client
            # re-derives from GET, so leaking the fetched copies drops the GET's card (#505).
            phase_backfill = [
                p for p in (result.get("parts") or []) if p.get("tool") == _BACKFILL_PHASE_TOOL
            ]
            result["parts"] = [*client_parts, *phase_backfill, *caveat_parts]
        if turn:
            turn.result = result
    except AgentTurnError as exc:
        logger.warning("Agent turn provider error: session=%s turn=%s", session_id, turn_id)
        if turn:
            turn.error = str(exc)
            turn.error_status = exc.status
            turn.error_code = exc.code
    except httpx.HTTPStatusError as exc:
        logger.warning("Agent turn upstream error: session=%s turn=%s", session_id, turn_id)
        if turn:
            turn.error_status = exc.response.status_code
            turn.error = str(exc)
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning("Agent turn upstream unreachable: session=%s turn=%s", session_id, turn_id)
        if turn:
            turn.error_status = 502
            turn.error = str(exc)
    except Exception as exc:
        logger.exception("Agent turn failed: session=%s turn=%s", session_id, turn_id)
        if turn:
            turn.error = str(exc) or exc.__class__.__name__
    finally:
        if turn:
            turn.active = False
            turn.finished_at = datetime.now(UTC)
            await _record_turn_metrics(state, turn, session_id, turn_id)
        state.last_activity = datetime.now(UTC)


class CompletionRequest(BaseModel):
    turn_id: str | None = None
    outcome: str = "success"  # "success" | "gave_up"
    note: str | None = None


@router.post("/session/{session_id}/monitor/complete")
async def mark_run_complete(session_id: str, body: CompletionRequest) -> dict[str, Any]:
    """Record a human-declared run completion for monitor metrics.

    Studio calls this when the tester marks a test run done. It defines the
    boundary for turns-to-complete and total-tokens; the product itself has no
    workflow-completion signal (the orchestrator loops indefinitely).
    """
    state = _sessions.get(session_id)
    monitor_log.append_record(
        session_id,
        {
            "kind": "completion",
            "session_id": session_id,
            "conversation_root": state.orchestrator_id if state else None,
            "turn_id": body.turn_id,
            "outcome": body.outcome,
            "note": body.note,
        },
    )
    return {"ok": True}


@router.get("/session/{session_id}/turn/{turn_id}")
async def get_turn(session_id: str, turn_id: str) -> dict[str, Any]:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")
    turn = state.turns.get(turn_id)
    if not turn:
        raise HTTPException(status_code=404, detail="unknown turn")
    if not turn.active:
        turn.consumed = True  # client has the final result; safe to evict later
    return {
        "active": turn.active,
        "result": turn.result,
        "error": turn.error,
        "error_status": turn.error_status,
        "error_code": turn.error_code,
    }


def _consume_interaction_steer(state: SessionState, user_text: str) -> str:
    """Prepend any steer the interaction gate parked on a prior held-card turn, so the
    model receives it as the lead of its next message and does the missing step itself.
    The steer is model-directed and is never shown to the user."""
    if state.pending_interaction_steer:
        user_text = f"{system_reminder(state.pending_interaction_steer)}\n\n{user_text}"
        state.pending_interaction_steer = None
    return user_text


async def _handle_subagent_message(
    state: SessionState,
    session_id: str,
    user_text: str,
    body: MessageRequest,
) -> dict[str, Any]:
    # A subagent whose delegated job just finished is done: steer it to hand control back
    # immediately, instead of the generic "suggest next steps" nudge (which left weaker
    # models looping on post-job inspection calls and never signalling completion). BOTH
    # outcomes get a defined hand-back — success resumes the waiting consumer with the
    # dataset, failure resumes it with the bad news. (A failed job previously got no
    # hand-back steer, so the producer kept control and the flow could stall.)
    job_failed = False
    if body.event == "job_complete":
        # Keep the event text (it carries the job id both hand-back prompts must report and
        # the grounding corpus needs) and append the steer — don't replace it.
        if _job_succeeded(body, user_text):
            user_text = f"{user_text}\n\n{system_reminder(_SUBAGENT_JOB_DONE_PROMPT)}"
        else:
            user_text = f"{user_text}\n\n{system_reminder(_SUBAGENT_JOB_FAILED_PROMPT)}"
            job_failed = True
    else:
        # Re-inject the current phase's block each ongoing turn (not just at the first
        # handoff), so the active step's full instructions stay in the freshest position.
        # Skipped on job_complete, which carries its own terminal hand-back steer above.
        target_for_phase = state.subagent_target or ""
        phase_for_turn = subagent_phase(target_for_phase, state.last_signal_step)
        # During gathering, surface the active stepped skill's current step detail (one step
        # per turn) just under the phase block, then advance the cursor so the next turn gets
        # the next step. The skill's titles + body stay resident, so this only moves the
        # freshest detail forward — it never hides the outline. Advanced exactly here (once
        # per turn), not in _ensure_phase_signal (which can run twice on a rerun).
        user_text = _with_active_skill_step(state, target_for_phase, phase_for_turn, user_text)
        user_text = with_block(target_for_phase, phase_for_turn, user_text)
        # Deliver any steer the interaction gate parked on a prior held-card turn as the
        # lead of this message (ahead of the phase block — an urgent correction outranks
        # ambient framing). Skipped on a job_complete event (which appends to user_text
        # above) so the steer stays parked for the next real turn instead of being lost.
        user_text = _consume_interaction_steer(state, user_text)
    logger.info("Routing to subagent: session=%s target=%s", session_id, state.subagent_target)
    try:
        result = await _proxy_send_message(
            state.subagent_id,  # type: ignore[arg-type]
            user_text,
            agent=state.subagent_target,
            model=body.model,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.warning("Subagent session gone, tearing down: session=%s", session_id)
            state.subagent_id = None
            state.subagent_target = None
            state.pending_help_target = None
            state.pending_help_producer = None
        raise
    except Exception:
        logger.exception("Subagent message failed: session=%s", session_id)
        raise

    latest_parts = await _fetch_all_assistant_parts(state.subagent_id)  # type: ignore[arg-type]
    # A weak driver sometimes bundles signal_subagent_completion AND request_orchestrator_help
    # into one turn. A subagent that still names a prerequisite it needs is NOT actually done —
    # taking the completion path there discarded the structured hand-back AND its auto-resume
    # arming (the eval→sdg→eval stall). So when BOTH are present and the help names a real need,
    # help wins; completion only wins when no such help ask rides with it.
    help_request = _detect_help_request(latest_parts)
    help_wins = bool(help_request and help_request[0].strip())
    summary = _detect_completion(latest_parts)

    if summary and not help_wins:
        logger.info("Subagent completion signal received: session=%s", session_id)
        completer = state.subagent_target
        if completer and state.subagent_id:
            state.completed_subagents[completer] = state.subagent_id
        state.subagent_id = None
        state.subagent_target = None

        if state.pending_help_target and completer == state.pending_help_producer:
            # Backend-driven resume: the producer (e.g. sdg) the consumer was WAITING ON just
            # finished the prerequisite it requested via request_orchestrator_help. Resume
            # that consumer directly instead of asking the orchestrator model to infer it —
            # deterministic, so it can't misfire. Gating on the awaited producer (not "any
            # other subagent finished") avoids resurrecting the consumer off an unrelated
            # completion during a detour.
            target = state.pending_help_target
            state.pending_help_target = None
            state.pending_help_producer = None
            logger.info("Auto-resuming help requester: session=%s target=%s", session_id, target)
            if job_failed:
                resume_context = (
                    f"The {target} prerequisite you requested could NOT be produced — the "
                    f"job failed:\n{summary}\n\n"
                    "Tell the user the dataset isn't available and ask how they want to "
                    "proceed. Do NOT assume the data exists or continue as if it succeeded."
                )
            else:
                resume_context = (
                    f"The {target} prerequisite you requested is ready.\n{summary}\n\n"
                    "Continue your own workflow with this result — do not restart it or "
                    "re-ask the user for anything you already have."
                )
            resume_result = await _maybe_delegate(
                state, session_id, (target, resume_context, True), summary, result, body
            )
            # The resumed consumer may itself finish in this same turn (e.g. eval scores and
            # signals completion right after getting its dataset). Detect that and release
            # control — otherwise it stays assigned to the consumer, with its completion
            # stripped from the UI, until an unrelated next user turn.
            if state.subagent_id and _detect_completion(
                await _fetch_all_assistant_parts(state.subagent_id)
            ):
                if state.subagent_target:
                    state.completed_subagents[state.subagent_target] = state.subagent_id
                state.subagent_id = None
                state.subagent_target = None
        else:
            resume_prompt = (
                f"[SUBAGENT COMPLETED]\n{summary}\n\n"
                "REQUIRED: You MUST either delegate_to_subagent (if the user "
                "already expressed intent) or call present_options with next "
                "steps. Do NOT respond with only text."
            )
            # The resume block (Phase 3) leads the orchestrator turn; the clean
            # resume_prompt still flows to _maybe_delegate as the next handoff's text.
            morty_result = await _orchestrator_turn(
                state, with_block("orchestrator", "resume", resume_prompt), body
            )
            delegation = _detect_delegation(await _fetch_all_assistant_parts(state.orchestrator_id))
            resume_result = await _maybe_delegate(
                state, session_id, delegation, resume_prompt, morty_result, body
            )
        response_parts = _strip_internal_tools(result.get("parts", []))
        result["parts"] = response_parts + _strip_internal_tools(resume_result.get("parts", []))
        result["info"] = resume_result.get("info", result.get("info", {}))
        return result

    # A subagent is a leaf — it cannot delegate. When it needs a prerequisite it
    # can't produce itself (e.g. the eval agent needs a held-out dataset), it hands
    # back via request_orchestrator_help. Stash the caller (as a completion does), let the
    # orchestrator run the producer, then arm the auto-resume bound to that producer so when
    # it completes the proxy resumes this caller automatically — the orchestrator only has to
    # make the first hop (delegate to the producer), never the resume.
    if help_request and state.subagent_target and state.subagent_id:
        need, help_context = help_request
        target = state.subagent_target
        logger.info("Subagent help request: session=%s target=%s", session_id, target)
        state.completed_subagents[target] = state.subagent_id
        state.subagent_id = None
        state.subagent_target = None
        # Prefer the externalized fulfill_help phase block (same {target}/{need}/{context}
        # placeholders); fall back to the inline constant if the file isn't present.
        seed_tmpl = phase_block("orchestrator", "fulfill_help") or _SUBAGENT_HELP_PROMPT
        seed = seed_tmpl.format(target=target, need=need, context=help_context)
        morty_result = await _orchestrator_turn(state, seed, body)
        delegation = _detect_delegation(await _fetch_all_assistant_parts(state.orchestrator_id))
        # The leaf's user text is the dataset ask (need), NOT the orchestrator seed —
        # the seed is the orchestrator's instruction sheet ("delegate to sdg…") and must
        # never reach the producer, or a weak model obeys it instead of generating.
        help_result = await _maybe_delegate(state, session_id, delegation, need, morty_result, body)
        # Arm the auto-resume ONLY if the orchestrator actually delegated to a producer (now
        # the active subagent): resume THIS consumer when THAT producer completes. If the
        # orchestrator asked a clarifying question instead of delegating, there is no producer
        # to await — leave the resume disarmed so an unrelated later completion can't
        # resurrect the consumer (it stays stashed for an explicit resume).
        if state.subagent_id and state.subagent_target and state.subagent_target != target:
            state.pending_help_target = target
            state.pending_help_producer = state.subagent_target
        response_parts = _strip_internal_tools(result.get("parts", []))
        help_result["parts"] = response_parts + _strip_internal_tools(help_result.get("parts", []))
        return help_result

    # Ordinary subagent turn (no completion, no help request). Strip internal tools
    # defensively so a stray/denied hand-back call a subagent emits here is never
    # surfaced to the user. Phase backfill is applied uniformly by _run_turn.
    result["parts"] = _strip_internal_tools(result.get("parts", []))
    return result


async def _orchestrator_turn(
    state: SessionState,
    text: str,
    body: MessageRequest,
) -> dict[str, Any]:
    return await _proxy_send_message(state.orchestrator_id, text, agent="morty", model=body.model)


async def _maybe_delegate(
    state: SessionState,
    session_id: str,
    delegation: tuple[str, str, bool] | None,
    user_text: str,
    morty_result: dict[str, Any],
    body: MessageRequest,
) -> dict[str, Any]:
    if not delegation:
        return morty_result

    target, context, resume = delegation

    # A FRESH delegation starts a NEW job for this target, so re-establish its confirm-card
    # preconditions: clear the target's accumulated signals (data confirmed, model/judge
    # offered, VRAM shown, metrics approved) so the new job must earn them again. A RESUME,
    # though, is the same workflow continuing — an auto-resume after a help request, a
    # failed-job retry, or the eval base-model reuse cycle — where the user's earlier
    # confirmations still stand; resetting there wiped signals the driver won't re-offer and
    # held the confirm card until the user prodded. So reset only when NOT resuming.
    # (In-session refinements of an already-shown card do NOT pass through here at all, so
    # they keep their signals regardless.)
    if not resume:
        _reset_confirm_signals(state, target)

    advisory = await _no_data_advisory(state, target)
    if advisory:
        context = f"{advisory}\n\n{context}" if context else advisory

    stashed_id = state.completed_subagents.pop(target, None) if resume else None

    if stashed_id:
        logger.info("Resuming subagent: target=%s session=%s → %s", target, session_id, stashed_id)
        # Inject the block for the phase the subagent last signalled, so a resume lands
        # it back on the right step with that step's full instructions in the turn.
        framing = system_reminder(f"[RESUMED]\n{context}")
        resume_phase = subagent_phase(target, state.last_signal_step)
        resume_msg = _with_active_skill_step(
            state, target, resume_phase, f"{framing}\n\n{user_text}"
        )
        handoff_msg = with_block(target, resume_phase, resume_msg)
        sub_result = await _proxy_send_message(
            stashed_id, handoff_msg, agent=target, model=body.model
        )
        state.subagent_id = stashed_id
    else:
        logger.info("New subagent: target=%s session=%s", target, session_id)
        sub_id = await _proxy_create_session()
        # A fresh subagent starts at its entry phase: clear any step carried over from a
        # previous subagent so subagent_phase() resolves to the first section, and inject
        # that section's block as the lead of the handoff. Also drop any skill the prior
        # session for this target had loaded and reset its step cursor — the fresh subagent
        # re-loads its own skill and walks the steps from the start.
        state.last_signal_step = ""
        state.active_skill_by_target.pop(target, None)
        state.skill_step_idx.pop(target, None)
        framing = system_reminder(f"[CONTEXT]\n{context}")
        handoff_msg = with_block(
            target,
            subagent_phase(target, state.last_signal_step),
            f"{framing}\n\n{user_text}",
        )
        sub_result = await _proxy_send_message(sub_id, handoff_msg, agent=target, model=body.model)
        state.subagent_id = sub_id

    state.subagent_target = target
    return sub_result


async def _handle_orchestrator_message(
    state: SessionState,
    session_id: str,
    user_text: str,
    body: MessageRequest,
) -> dict[str, Any]:
    user_text = _consume_interaction_steer(state, user_text)
    # The orchestrator never calls signal_phase (it is a primary agent); the proxy picks
    # its phase by control-flow path. A fresh user turn is intake→delegate; a job-status
    # event with no active subagent is the monitor case. Inject the block ONLY into the
    # text the orchestrator reads — the clean user_text still flows to _maybe_delegate as
    # the subagent's user text, so orchestrator-only guidance never leaks downstream.
    if body.event == "job_complete":
        orch_text = with_block("orchestrator", "monitor", user_text)
    else:
        orch_text = with_block(
            "orchestrator", "delegate", with_block("orchestrator", "intake", user_text)
        )
    morty_result = await _orchestrator_turn(state, orch_text, body)
    delegation = _detect_delegation(await _fetch_all_assistant_parts(state.orchestrator_id))
    return await _maybe_delegate(state, session_id, delegation, user_text, morty_result, body)


@router.get("/session/{session_id}/pending")
async def get_pending(session_id: str) -> dict[str, Any]:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")
    target_id = state.subagent_id or state.orchestrator_id
    result: dict[str, Any] = await _proxy_get(target_id, "pending", {"messages": []}, session_id)
    return result


# OpenCode bundles a built-in "opencode" provider (its hosted "Zen" free models) that is always
# "connected" even though the user never configured it — and in this deployment the sandbox egress
# only allows the user's own providers, so those models aren't reachable. Hide it from the picker.
_SUPPRESSED_PROVIDERS = frozenset({"opencode"})

# Model ids differ in convention across sources: OpenCode/models.dev use aliases (claude-sonnet-4-5)
# while a provider's own /v1/models may return dated snapshots (claude-sonnet-4-5-20250929). Strip a
# trailing dated suffix so the two compare equal when access-filtering the picker.
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$|-\d{4}-\d{2}-\d{2}$")


def _strip_date_suffix(model_id: str) -> str:
    return _DATE_SUFFIX_RE.sub("", model_id)


@router.get("/provider")
async def list_providers() -> dict[str, Any]:
    """Provider catalog + connection status from OpenCode, for the Studio model picker.

    Proxies OpenCode's ``/provider`` ({all, default, connected}) and reduces it to what the chat
    picker needs: only the ``connected`` providers (those with credentials — ``all`` is the entire
    models.dev catalog of ~225 providers / thousands of models, far too big to ship), each with its
    chat-usable models (normalized to [{id, name}]). "Chat-usable" is judged purely by OpenCode's
    capability flags (emits text + supports tool calls), so the filter is provider-agnostic and new
    models appear automatically. Credentials and all other fields are dropped before the browser.
    """
    empty: dict[str, Any] = {"all": [], "default": {}, "connected": []}
    try:
        resp = await _client().get(f"{_opencode_url()}/provider", timeout=10.0)
        resp.raise_for_status()
        if "application/json" not in resp.headers.get("content-type", ""):
            return empty
        data: dict[str, Any] = resp.json()
    except httpx.HTTPError:
        logger.warning("Upstream unreachable on GET /provider")
        return empty
    except ValueError:
        logger.exception("Bad JSON on GET /provider")
        return empty
    raw_all = data.get("all")
    connected_raw = data.get("connected")
    connected = (
        [c for c in connected_raw if isinstance(c, str) and c not in _SUPPRESSED_PROVIDERS]
        if isinstance(connected_raw, list)
        else []
    )
    connected_set = set(connected)

    # Access-filter: a provider's own /v1/models = what the user's key can actually use
    # (OpenCode's /provider is the whole models.dev catalog). Intersecting drops models the
    # key isn't entitled to. Providers with no server key (vertex ADC) are absent here and
    # left unfiltered; an empty intersection (differing id conventions) also stays unfiltered.
    from amortized.core.model_catalog import accessible_model_ids

    access = await accessible_model_ids()

    def _chat_models(p: dict[str, Any]) -> list[dict[str, str]]:
        # Chat-usable models only, via OpenCode's own capability flags (provider-agnostic, no
        # per-provider/name rules): must emit text AND support tool calls (Morty is an agent).
        # Drops embeddings/image/audio uniformly; new models appear automatically (models.dev).
        raw = p.get("models")
        if isinstance(raw, dict):
            items = list(raw.values())
        elif isinstance(raw, list):
            items = raw
        else:
            items = []
        out: list[dict[str, str]] = []
        for m in items:
            if not isinstance(m, dict) or not m.get("id"):
                continue
            cap = m.get("capabilities")
            cap = cap if isinstance(cap, dict) else {}
            out_mods = cap.get("output")
            out_mods = out_mods if isinstance(out_mods, dict) else {}
            if not (out_mods.get("text") and cap.get("toolcall")):
                continue
            out.append({"id": str(m["id"]), "name": str(m.get("name") or m["id"])})
        # Keep only models the user's key can reach (its own /v1/models). Fall back to the full list
        # when there's no access list for this provider, or the intersection is empty (its id
        # convention differs from models.dev) — never hide everything.
        allowed = access.get(str(p.get("id") or ""))
        if allowed:
            # Match on the raw id OR a date-normalized form, so an alias id (claude-sonnet-4-5)
            # still matches a dated id (claude-sonnet-4-5-20250929) and a reachable model whose
            # convention differs isn't dropped. (Guards against hiding *some* models, not just all.)
            allowed_base = {_strip_date_suffix(a) for a in allowed}
            filtered = [
                m for m in out if m["id"] in allowed or _strip_date_suffix(m["id"]) in allowed_base
            ]
            if filtered:
                out = filtered
        return out

    # Scope to providers the user has credentials for (`connected`): `all` is the full models.dev
    # catalog (hundreds of providers), too big to ship; the picker only shows connected anyway.
    safe_all = (
        [
            {"id": p["id"], "name": p.get("name", p["id"]), "models": _chat_models(p)}
            for p in raw_all
            if isinstance(p, dict) and p.get("id") and p["id"] in connected_set
        ]
        if isinstance(raw_all, list)
        else []
    )
    return {"all": safe_all, "default": {}, "connected": connected}


@router.post("/title")
async def generate_title() -> dict[str, Any]:
    return {"title": ""}


@router.get("/health")
async def agent_health() -> dict[str, Any]:
    try:
        resp = await _client().get(f"{_opencode_url()}/api/health", timeout=5.0)
        upstream = resp.json() if resp.status_code == 200 else {"healthy": False}
    except Exception:
        upstream = {"healthy": False}
    return {"healthy": upstream.get("healthy", False), "proxy": True}
