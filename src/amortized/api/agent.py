"""Agent session proxy with subagent routing.

Proxies chat messages to OpenCode and intercepts delegation/completion
signals to route conversations between the orchestrator and ephemeral
subagent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
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
from amortized.config import settings

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
    # (target, opencode_session_id) pairs for subagents that delegated to
    # another subagent (e.g. eval → sdg). When the delegate completes,
    # control returns to the delegating subagent, not the orchestrator.
    subagent_stack: list[tuple[str, str]] = field(default_factory=list)
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
    # Content signatures of review cards (e.g. an assessor prompt) already rendered
    # to the user standalone. The review gate fires only for a prompt NOT in this
    # set; once shown, a later confirm that re-includes the same prompt is allowed
    # (the redundant review card is dropped), so a driver that re-shows the prompt
    # before every validate_* can't livelock. See _enforce_single_interaction.
    reviewed_card_sigs: set[str] = field(default_factory=set)


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

INTERNAL_TOOLS = {"delegate_to_subagent", "signal_subagent_completion"}

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


def _job_succeeded(body: MessageRequest, user_text: str) -> bool:
    """Whether a job_complete event reports success. Prefers the structured
    `outcome` field; falls back to the legacy display-text substring only when the
    client didn't send one (version skew during a rollout)."""
    if body.outcome is not None:
        return body.outcome.strip().lower() == "succeeded"
    return "status: succeeded" in user_text.lower()


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


def _strip_internal_tools(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in parts if _tool_name(p) not in INTERNAL_TOOLS]


# Tools whose result renders a job CONFIRMATION card (its own Confirm / Cancel).
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


def _enforce_single_interaction(
    result: dict[str, Any], state: SessionState
) -> dict[str, Any]:
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

    Whichever interaction LEADS the turn is kept; every OTHER interactive element
    (extra questions, repeat/stacked confirm cards) is stripped. Passive content
    — explanatory text, show_* display cards, signal_phase — is ALWAYS kept, even
    after the kept interaction, so content the question is about (e.g. a proposed
    metric-set table the driver wrote as trailing text) is never dropped. The
    monitor log is recorded from the raw opencode parts, not this result, so the
    batching stays visible/measurable there."""
    parts = result.get("parts") or []
    first_q = first_card = first_review = None
    for i, part in enumerate(parts):
        name = _tool_name(part)
        if first_q is None and name == "present_options":
            first_q = i
        if first_card is None and name in _CONFIRM_CARD_TOOLS:
            first_card = i
        if first_review is None and name in _REVIEW_CARD_TOOLS:
            first_review = i
    if first_q is None and first_card is None:
        return result

    ask_turn = first_q is not None and (first_card is None or first_q < first_card)

    # Review gate: a review card must be reviewed on its own turn before the job it
    # gates is confirmed — but only the FIRST time the user sees that prompt, so a
    # driver re-showing it before every confirm can't livelock.
    review_sig = _review_card_signature(parts[first_review]) if first_review is not None else None
    new_review = review_sig is not None and review_sig not in state.reviewed_card_sigs
    prompt_gate = not ask_turn and first_card is not None and new_review

    if prompt_gate:
        # No interaction leads: keep passive parts (incl. the review card); drop the
        # confirm card so the prompt stands alone. The job is confirmed next turn.
        keep_index: int | None = None
        drop_review = False
    else:
        # Whichever interaction leads the turn is the one the model is really making;
        # keep exactly that one and strip every OTHER interactive element. If a
        # confirm proceeds with an already-reviewed prompt stacked on it, drop the
        # now-redundant review card (its text is already inside the confirm card).
        keep_index = first_q if ask_turn else first_card
        drop_review = not ask_turn and first_card is not None and review_sig is not None

    kept = [
        part
        for i, part in enumerate(parts)
        if i == keep_index
        or (
            _tool_name(part) != "present_options"
            and _tool_name(part) not in _CONFIRM_CARD_TOOLS
            and not (drop_review and _tool_name(part) in _REVIEW_CARD_TOOLS)
        )
    ]
    # Mark the prompt reviewed once it is actually rendered to the user this turn,
    # so the next confirm that re-includes it proceeds instead of re-gating.
    if review_sig is not None and not drop_review:
        state.reviewed_card_sigs.add(review_sig)
    if len(kept) == len(parts):
        return result
    return {**result, "parts": kept}


# Active subagent → the workflow phase its turns belong to (UI progress bar).
_PHASE_FOR_TARGET = {"sdg": "sdg", "training": "training", "eval": "eval"}


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
    for part in parts:
        if _tool_name(part) == "signal_phase":
            step = _get_tool_input(part).get("step")
            if step:
                state.last_signal_step = str(step)
            return result  # model signalled this turn; nothing to backfill
    payload = {"phase": phase, "step": state.last_signal_step}
    result["parts"] = [
        *parts,
        {
            "type": "tool",
            "tool": "mcp_amortized__signal_phase",
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


async def _record_turn_metrics(
    state: SessionState,
    turn: TurnState,
    session_id: str,
    turn_id: str,
    cache: dict[str, list[dict[str, Any]]] | None = None,
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
        for target, sid in state.subagent_stack:
            role_by_session.setdefault(sid, target)

        new_msgs: list[tuple[int, str, dict[str, Any]]] = []
        for sid, role in role_by_session.items():
            for msg in await _fetch_all_messages(sid, cache):
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
_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{8,}"
)


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
    sessions.update(sid for _t, sid in state.subagent_stack)

    chunks: list[str] = []
    for sid in sessions:
        for msg in await _fetch_all_messages(sid, cache):
            role = (msg.get("info") or {}).get("role")
            for part in msg.get("parts") or []:
                ptype = part.get("type")
                if role == "user" and ptype == "text":
                    chunks.append(str(part.get("text") or ""))
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
                violations, active_id, agent, attempt, MAX_PROVENANCE_RETRIES,
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
            current = await _proxy_send_message(
                active_id, correction, agent=agent, model=body.model
            )
            # The correction adds new messages to the active session; drop its cached
            # copy so re-grounding fetches them (other sessions stay cached).
            if cache is not None:
                cache.pop(active_id, None)
            # Re-ground each attempt: a correction may call a tool that now surfaces
            # the id legitimately.
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
            MAX_PROVENANCE_RETRIES, violations, active_id,
        )
        parts = list(current.get("parts") or [])
        parts.append(
            {
                "type": "text",
                "text": (
                    "\n\n⚠️ I couldn't verify the identifier(s) "
                    f"{', '.join(violations)} against platform data in this "
                    "session — treat them as unconfirmed until I fetch the real "
                    "values."
                ),
            }
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
_VALIDATE_TOOL_BY_TYPE = {
    "sdg": "validate_sdg_job",
    "training": "validate_training_job",
    "eval": "validate_eval_job",
}

# Check C — downstream dispatch tools Morty calls to commit a training/eval run onto
# a parent dataset. The call carries parent_job_id as a structured argument, so
# dispatch provenance is verified from the parent's DB record, with no prose parsing.
_DISPATCH_PARENT_TYPE = {
    "validate_training_job": "training",
    "validate_eval_job": "eval",
}


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
    the active and completed/stashed subagents."""
    sessions = {state.orchestrator_id}
    if state.subagent_id:
        sessions.add(state.subagent_id)
    sessions.update(state.completed_subagents.values())
    sessions.update(sid for _t, sid in state.subagent_stack)
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

# Tools that mean the user is working with an EXISTING dataset (browsing, inspecting,
# or splitting one) — so training/eval off it is not premature.
_DATASET_TOOLS = {"list_datasets", "get_dataset", "get_dataset_samples", "split_dataset"}


async def _conversation_has_dataset(
    state: SessionState,
    context: str,
    user_text: str,
    cache: dict[str, list[dict[str, Any]]] | None,
) -> bool:
    """Whether THIS conversation already has, or points at, a dataset — so a
    training/eval delegation is NOT premature. All signals are structural
    (conversation-scoped, never a lifetime "has this user ever" query):

    1. An SDG job was set up here (`validate_sdg_job`) — fresh data in flight.
    2. A dataset tool was touched here — the user engaged an existing dataset.
    3. An id cited in the handoff/user text resolves to a succeeded SDG/upload job
       — the user pointed training/eval at an existing dataset (the DB is the
       arbiter; id-token extraction, not prose classification).
    """
    if await _session_has_validate(state, "sdg", cache):
        return True
    id_text = [context, user_text]
    for sid in _session_ids(state):
        for msg in await _fetch_all_messages(sid, cache):
            role = (msg.get("info") or {}).get("role")
            for part in msg.get("parts") or []:
                if part.get("type") == "tool":
                    if _tool_name(part) in _DATASET_TOOLS:
                        return True
                elif role == "user" and part.get("type") == "text":
                    id_text.append(str(part.get("text") or ""))
    job_cache: dict[str, dict[str, Any] | None] = {}
    for tok in _extract_id_candidates("\n".join(id_text)):
        job = await _fetch_job_by_token(tok, job_cache)
        if (
            job
            and job.get("type") in {"sdg", "upload"}
            and job.get("status") == "succeeded"
        ):
            return True
    return False


async def _no_data_advisory(
    state: SessionState, target: str, context: str, user_text: str
) -> str | None:
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
        if await _conversation_has_dataset(state, context, user_text, {}):
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
    for token in sorted(tokens):
        job = await _fetch_job_by_token(token, job_cache)
        if not job:
            continue  # not a real job (or ambiguous) — provenance gate covers fabrication
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
    # Check C — a training/eval dispatch this turn onto a stale, undisclosed parent
    # dataset. Keyed on the dispatch tool call's parent_job_id, independent of prose,
    # so it fires even when the reply names no id. Dedup against A/B by parent prefix.
    flagged = {v["token"] for v in violations if v.get("token")}
    for v in await _dispatch_provenance_violations(state, result, text, cache, job_cache):
        if v["token"] not in flagged:
            violations.append(v)
    return violations


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


def _job_claim_correction(violations: list[dict[str, Any]]) -> str:
    lines = [
        "[JOB-CLAIM CHECK — internal system verification, not from the user]",
        "Your reply described job(s) in a way the platform's records contradict:",
    ]
    for v in violations:
        if v["kind"] == "state":
            lines.append(
                f"- You presented job {v['token']} as finished/ready, but its real"
                f" status is '{v['status']}'. Do NOT claim completion until it is"
                " actually succeeded — state the real status or wait for it."
            )
        elif v["kind"] == "dispatch":
            verb = "train on" if v["type"] == "training" else "evaluate on"
            size = f" ({v['num']} records)" if v["num"] is not None else ""
            lines.append(
                f"- You are about to {verb} dataset {v['token']}{size}, but that is an"
                " EXISTING dataset created in an earlier conversation — you did not run"
                " a fresh SDG job for this request this session, nor tell the user you"
                " are reusing it. Either submit a new SDG job for what the user asked"
                f" for, or explicitly tell the user you are REUSING existing dataset"
                f" {v['token']}{size} so they can confirm it matches before it runs."
            )
        else:
            size = f" ({v['num']} records)" if v["num"] is not None else ""
            lines.append(
                f"- You implied job {v['token']} was generated/started in this"
                f" conversation, but it is an EXISTING {v['type']} dataset{size}"
                " created in an earlier conversation and no matching job was"
                " submitted this session. Either submit a fresh job for the user's"
                f" request, or explicitly tell the user you are REUSING existing job"
                f" {v['token']}{size} so they can confirm it matches what they asked for."
            )
    lines.append("Re-send your reply with the accurate status / reuse disclosure.")
    return "\n".join(lines)


def _job_claim_caveat(violations: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for v in violations:
        if v["kind"] == "state":
            parts.append(f"job {v['token']} is '{v['status']}', not finished")
        elif v["kind"] == "dispatch":
            size = f" ({v['num']} records)" if v["num"] is not None else ""
            parts.append(
                f"the {v['type']} run is about to use existing dataset {v['token']}{size}"
                " from an earlier conversation, which was not generated or disclosed as"
                " reused this session"
            )
        else:
            size = f" ({v['num']} records)" if v["num"] is not None else ""
            parts.append(
                f"job {v['token']} is an existing dataset{size} from an earlier"
                " conversation, not a fresh run"
            )
    return (
        "\n\n⚠️ Platform records don't match what I said above: "
        + "; ".join(parts)
        + ". Treat these as unconfirmed."
    )


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
                violations, active_id, agent, attempt, MAX_PROVENANCE_RETRIES,
            )
            current = await _proxy_send_message(
                active_id, _job_claim_correction(violations), agent=agent, model=body.model
            )
            if cache is not None:
                cache.pop(active_id, None)
            violations = await _job_claim_violations(state, current, cache, job_cache)
            if not violations:
                return current

        logger.warning(
            "Job-claim gate: still contradicted after %d correction(s) %s session=%s",
            MAX_PROVENANCE_RETRIES, violations, active_id,
        )
        parts = list(current.get("parts") or [])
        parts.append({"type": "text", "text": _job_claim_caveat(violations)})
        return {**current, "parts": parts}
    except Exception:
        logger.warning("Job-claim gate failed (passing through)", exc_info=True)
        return result


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
    """A provider/model error during a turn, already mapped to a user-facing message."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _friendly_provider_error(raw: str, name: str = "") -> str:
    """Map a raw provider/opencode error to a short, user-facing chat message. Falls back to the
    raw text (capped) so an unmapped error is still more useful than a generic one."""
    text = f"{raw} {name}".lower()
    if "tool" in text and ("choice" in text or "parser" in text or "enable-auto-tool" in text):
        return (
            "This model's server doesn't have tool-calling enabled, which Morty needs. "
            "Pick a tool-calling-capable model."
        )
    auth = ("providerautherror", "unauthorized", "forbidden", "permission", "invalid api key")
    missing = ("not found", "does not exist", "providermodelnotfound", "no such model")
    limited = ("rate limit", "too many requests", "quota")
    if any(s in text for s in auth):
        return "That model isn't available with your API key or project (auth/permission)."
    if any(s in text for s in missing):
        return "That model isn't available on this provider/endpoint."
    if any(s in text for s in limited):
        return "The provider is rate-limiting or out of quota — try again shortly."
    return (raw or "").strip()[:180] or "The model provider returned an error."


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
        raise AgentTurnError(_friendly_provider_error(raw, str(err.get("name") or "")))
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


@router.get("/session/{session_id}/message")
async def get_session_messages(session_id: str) -> Any:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")
    target_id = state.subagent_id or state.orchestrator_id
    return await _proxy_get(target_id, "message", {"info": {}, "parts": []}, session_id)


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
            result = await _apply_provenance_gate(state, result, body, msg_cache)
            result = await _apply_job_claim_gate(state, result, body, msg_cache)
            # Backfill the UI phase uniformly for every path (subagent completion,
            # subagent→subagent delegation, orchestrator) — not just the one return
            # inside the subagent handler — so the progress bar never strands on a
            # stale phase on a hand-back/delegation turn.
            result = _ensure_phase_signal(state, result)
            result = _enforce_single_interaction(result, state)
        if turn:
            turn.result = result
    except AgentTurnError as exc:
        logger.warning("Agent turn provider error: session=%s turn=%s", session_id, turn_id)
        if turn:
            turn.error = str(exc)
            turn.error_status = exc.status
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
            await _record_turn_metrics(state, turn, session_id, turn_id, msg_cache)
        state.last_activity = datetime.now(UTC)


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
    }


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


async def _handle_subagent_message(
    state: SessionState,
    session_id: str,
    user_text: str,
    body: MessageRequest,
) -> dict[str, Any]:
    # A subagent whose delegated job just finished successfully is done: steer it
    # to hand control back immediately, instead of the generic "suggest next
    # steps" nudge (which left weaker models looping on post-job inspection calls
    # and never signalling completion). Only on success — a failed job still needs
    # the subagent's own failure handling.
    if body.event == "job_complete" and _job_succeeded(body, user_text):
        user_text = _SUBAGENT_JOB_DONE_PROMPT
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
            state.subagent_stack.clear()
        raise
    except Exception:
        logger.exception("Subagent message failed: session=%s", session_id)
        raise

    latest_parts = await _fetch_all_assistant_parts(state.subagent_id)  # type: ignore[arg-type]
    summary = _detect_completion(latest_parts)

    if summary:
        logger.info("Subagent completion signal received: session=%s", session_id)
        if state.subagent_target and state.subagent_id:
            state.completed_subagents[state.subagent_target] = state.subagent_id
        state.subagent_id = None
        state.subagent_target = None

        if state.subagent_stack:
            # A subagent delegated to this one (e.g. eval → sdg). Return
            # control to the delegating subagent with the summary so it can
            # continue its own workflow with full context.
            resume_result = await _resume_parent_subagent(state, summary, body)
        else:
            resume_prompt = (
                f"[SUBAGENT COMPLETED]\n{summary}\n\n"
                "REQUIRED: You MUST either delegate_to_subagent (if the user "
                "already expressed intent) or call present_options with next "
                "steps. Do NOT respond with only text."
            )
            morty_result = await _orchestrator_turn(state, resume_prompt, body)
            delegation = _detect_delegation(await _fetch_all_assistant_parts(state.orchestrator_id))
            resume_result = await _maybe_delegate(
                state, session_id, delegation, resume_prompt, morty_result, body
            )
        response_parts = _strip_internal_tools(result.get("parts", []))
        result["parts"] = response_parts + _strip_internal_tools(resume_result.get("parts", []))
        result["info"] = resume_result.get("info", result.get("info", {}))
        return result

    # A subagent may itself delegate (e.g. the eval agent hands off to the
    # SDG agent to build a held-out eval set). Honor it: stash the current
    # subagent on the stack so control returns here when the new one
    # completes, then route to the new subagent.
    delegation = _detect_delegation(latest_parts)
    if delegation:
        target, _, _ = delegation
        logger.info(
            "Subagent delegation: session=%s %s → %s",
            session_id, state.subagent_target, target,
        )
        state.subagent_stack.append((state.subagent_target, state.subagent_id))  # type: ignore[arg-type]
        sub_result = await _maybe_delegate(state, session_id, delegation, user_text, result, body)
        response_parts = _strip_internal_tools(result.get("parts", []))
        sub_result["parts"] = response_parts + _strip_internal_tools(sub_result.get("parts", []))
        return sub_result

    # Phase backfill is applied uniformly by _run_turn for every return path.
    return result


async def _resume_parent_subagent(
    state: SessionState,
    summary: str,
    body: MessageRequest,
) -> dict[str, Any]:
    """Hand control back to the subagent that delegated, with the summary."""
    parent_target, parent_id = state.subagent_stack.pop()
    logger.info("Returning to parent subagent: target=%s", parent_target)
    state.subagent_id = parent_id
    state.subagent_target = parent_target

    resume_prompt = (
        f"[SUBAGENT COMPLETED]\n{summary}\n\n"
        "The delegated workflow finished. Continue your own workflow with "
        "this result — do not restart it or re-ask the user for information "
        "you already have."
    )
    parent_result = await _proxy_send_message(
        parent_id,
        resume_prompt,
        agent=parent_target,
        model=body.model
    )

    # The parent may delegate again (e.g. on to another workflow); honor it
    # the same way as the initial subagent delegation.
    delegation = _detect_delegation(await _fetch_all_assistant_parts(parent_id))
    if delegation:
        state.subagent_stack.append((parent_target, parent_id))
        return await _maybe_delegate(state, "", delegation, resume_prompt, parent_result, body)
    return parent_result


async def _orchestrator_turn(
    state: SessionState,
    text: str,
    body: MessageRequest,
) -> dict[str, Any]:
    return await _proxy_send_message(state.orchestrator_id, text, agent="morty", model=body.model)


# A subagent may hand off to another subagent (e.g. eval → sdg), which
# may hand off again. Bound the chain: each level holds an opencode
# session open, and an agent stuck in a delegate loop would otherwise
# grow the stack without limit.
MAX_SUBAGENT_DEPTH = 4


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

    if len(state.subagent_stack) >= MAX_SUBAGENT_DEPTH:
        logger.warning(
            "Subagent delegation depth limit reached: session=%s depth=%d",
            session_id, len(state.subagent_stack),
        )
        return {
            "info": morty_result.get("info", {}),
            "parts": [
                {
                    "type": "text",
                    "text": (
                        "I can't hand this off further — the workflow has"
                        " nested too many agents. Let's continue here"
                        " instead of delegating again."
                    ),
                }
            ],
        }

    advisory = await _no_data_advisory(state, target, context, user_text)
    if advisory:
        context = f"{advisory}\n\n{context}" if context else advisory

    stashed_id = state.completed_subagents.pop(target, None) if resume else None

    if stashed_id:
        logger.info("Resuming subagent: target=%s session=%s → %s", target, session_id, stashed_id)
        handoff_msg = f"[RESUMED]\n{context}\n\n[USER MESSAGE]\n{user_text}"
        sub_result = await _proxy_send_message(
            stashed_id, handoff_msg, agent=target, model=body.model
        )
        state.subagent_id = stashed_id
    else:
        logger.info("New subagent: target=%s session=%s", target, session_id)
        sub_id = await _proxy_create_session()
        handoff_msg = f"[CONTEXT]\n{context}\n\n[USER MESSAGE]\n{user_text}"
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
    morty_result = await _orchestrator_turn(state, user_text, body)
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
                m for m in out
                if m["id"] in allowed or _strip_date_suffix(m["id"]) in allowed_base
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
