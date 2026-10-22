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
from amortized.api.phases import Verdict, adjudicate, subagent_pointer
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
    # last_signal_step is a single field tracking the CURRENTLY active subagent (so the backfill,
    # adjudicator, and mid-pipeline routing all read one source of truth). A help detour or a
    # later delegation resets it, so stash each stepped-out subagent's own last step here, keyed
    # by target, and restore it when that subagent is resumed — otherwise the resumed consumer is
    # pointed at the wrong section (the detour producer's residue, or its entry). Stash-and-restore
    # rather than making last_signal_step itself per-target: that regressed mid-training routing.
    completed_last_step: dict[str, str] = field(default_factory=dict)
    # Per-target skill state, so the resident-workflow pointer can also name the current SKILL.md
    # step during gathering: the skill the target loaded (via the `skill` tool) and the latest
    # `present_options.step` stamp it emitted. Observed from the turn, used on the next send.
    active_skill_by_target: dict[str, str] = field(default_factory=dict)
    skill_last_step: dict[str, str] = field(default_factory=dict)


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


# Active subagent → the workflow phase its turns belong to (UI progress bar).
_PHASE_FOR_TARGET = {"sdg": "sdg", "training": "training", "eval": "eval"}


# Synthetic tool name for the phase signal _ensure_phase_signal backfills when the model
# forgot to call signal_phase. Deliberately distinct from opencode's real
# `amortized_signal_phase` so _run_turn can keep this backfill (the client's own GET lacks
# it) while dropping the other fetched tool parts it must not ship to the client (issue #505).
_BACKFILL_PHASE_TOOL = "mcp_amortized__signal_phase"


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
    target = state.subagent_target or ""
    parts = result.get("parts") or []
    # One pass over the turn's tool parts: remember the signalled workflow step AND the skill
    # context (which skill is loaded, the latest present_options stamp) so the next turn's pointer
    # can name both the resident workflow section and the current SKILL.md step. Scanned fully (no
    # early return) because a skill load / question card often trails the signal_phase in a turn.
    signalled = False
    for part in parts:
        name = _tool_name(part)
        if name == "signal_phase":
            step = _get_tool_input(part).get("step")
            if step:
                state.last_signal_step = str(step)
            signalled = True
        elif name == "skill":
            skill_name = _get_tool_input(part).get("name")
            if skill_name and target:
                state.active_skill_by_target[target] = str(skill_name)
        elif name == "present_options":
            stamp = _get_tool_input(part).get("step")
            if stamp and target:
                state.skill_last_step[target] = str(stamp)
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


def _pointer_for(state: SessionState, target: str | None) -> str | None:
    """The resident-workflow pointer for a subagent turn: the workflow section (from the last
    signalled step) plus the current SKILL.md step (from the tracked skill + stamp). None for a
    non-subagent, so callers compose it unconditionally. See phases.subagent_pointer."""
    if not target:
        return None
    return subagent_pointer(
        target,
        state.last_signal_step,
        state.active_skill_by_target.get(target),
        state.skill_last_step.get(target),
    )


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


# The validate_* tool each job type exposes — used to detect whether a fresh job of that
# type was set up in this conversation (see _session_has_validate / _conversation_has_dataset).
_VALIDATE_TOOL_BY_TYPE = {
    "sdg": "validate_sdg_job",
    "training": "validate_training_job",
    "eval": "validate_eval_job",
}


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


async def _delegation_advisory(
    state: SessionState, target: str, context: str, user_text: str
) -> str | None:
    """Handoff-CONTEXT advisory for a downstream (training/eval) delegation.

    Returns the data-availability fork when no dataset has been chosen yet — i.e. framing
    the new subagent should surface to the user (choose existing data vs. generate), not a
    mid-conversation user-channel correction. The former UPSTREAM-READY branch was REMOVED:
    that invariant ("don't build a downstream job on a non-'succeeded' upstream") is enforced
    structurally at the tool boundary — validate_training_job / validate_eval_job call
    _require_dependency_succeeded and return a 422 that surfaces to the model as a readable
    tool error (see api/jobs.py) — so re-asserting it here as injected prose was redundant and
    just added another in-band steer. See docs/agent-gates-proposal.md."""
    return await _no_data_advisory(state, target)


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
    system: str | None = None,
) -> dict[str, Any]:
    """POST a turn to opencode. ``text`` is the user-role part; ``system`` (when given) is
    delivered via opencode's message-level ``system`` field — an instruction the model reads
    as SYSTEM context rather than as a user utterance. The proxy's one use of ``system`` is the
    subagent hand-back imperative (on a job_complete event): the imperative rides ``system`` so
    it isn't a fake user turn, while the neutral event fact (e.g. the completed job id) stays in
    the user ``text``."""
    payload: dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
    if agent:
        payload["agent"] = agent
    if model:
        payload["model"] = model.model_dump(exclude_none=True)
    if system:
        payload["system"] = system
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
    try:
        async with state.lock:
            if state.subagent_id:
                result = await _handle_subagent_message(state, session_id, user_text, body)
            else:
                result = await _handle_orchestrator_message(state, session_id, user_text, body)
            # The handler's POST response carries only step markers; the real tool parts
            # (cards, options, show_prompt, signal_phase) live in the GET messages.
            # _ensure_phase_signal needs them to decide whether to backfill the UI progress
            # bar, but they must NOT leak into the client-facing result: the client re-derives
            # every card/option/prompt from its OWN GET /message and would drop the GET's card
            # as a duplicate (issue #505). So snapshot the markers, inspect the fetched parts
            # for the phase backfill, then restore the client contract.
            #
            # There is no gate/correction/stripping layer here anymore: structural invariants
            # are enforced at the tool boundary (validate_*/create_* -> 422 tool error), the
            # UI renders state from tool OUTPUTS (not prose), and fabrication/over-claim/ordering
            # prevention lives in the personas and the tool agent_instruction halt directives —
            # so the agent's context stays equal to what the user sees (AX = UX). See
            # docs/agent-gates-proposal.md.
            active_id = state.subagent_id or state.orchestrator_id
            client_parts = result.get("parts") or []
            fetched = _strip_internal_tools(await _fetch_all_assistant_parts(active_id))
            if fetched:
                result["parts"] = fetched
            result = _ensure_phase_signal(state, result)
            phase_backfill = [
                p for p in (result.get("parts") or []) if p.get("tool") == _BACKFILL_PHASE_TOOL
            ]
            result["parts"] = [*client_parts, *phase_backfill]
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


# A bracketed UPPERCASE tag, matching the PR509 system-channel convention ([SYSTEM EVENT],
# [SUBAGENT NEEDS HELP], [ACTIVE PHASE], …) — delivered raw on opencode's `system` field, never
# wrapped in <system-reminder>.
_STALL_NUDGE_INTRO = "[WORKFLOW STALL] Your last turn did not advance the workflow."


def _adjudicate_subagent_turn(
    state: SessionState, target: str, parts: list[dict[str, Any]]
) -> Verdict:
    """Classify the just-finished subagent turn with the pure adjudicator. `parts` is the turn's
    fetched assistant parts; errored tool calls are excluded (a dead call renders no edge). The
    active stepped-skill step is the present_options stamp the turn-scan tracked (a stalling turn
    emits no step slug of its own)."""
    tools = [_tool_name(p) for p in parts if _tool_name(p) and not _part_errored(p)]
    skill = state.active_skill_by_target.get(target)
    active_step = state.skill_last_step.get(target)
    present_steps = [
        str(_get_tool_input(p).get("step") or "").strip()
        for p in parts
        if _tool_name(p) in {"present_options", "signal_phase"}
    ]
    has_text = any(p.get("type") == "text" and str(p.get("text") or "").strip() for p in parts)
    return adjudicate(
        target,
        state.last_signal_step,
        tools,
        skill=skill,
        active_skill_step=active_step,
        has_text=has_text,
        present_steps=[s for s in present_steps if s],
    )


def _stall_nudge_text(step: str, verdict: Verdict) -> str | None:
    """The system-channel nudge for a non-ok verdict, or None when nothing is owed (ok /
    lenient). Names the owed tool(s) so the driver self-corrects by calling them."""
    where = f"`{step}`" if step else "the current step"
    if verdict.kind == "batching":
        return (
            f"{_STALL_NUDGE_INTRO} You signalled more than one step in a single turn. "
            "Handle ONE step per turn: signal the current step, take only its action, "
            "then wait for the user before moving on."
        )
    if verdict.owed is None:
        return None
    owed = sorted(verdict.owed.all_of or verdict.owed.any_of)
    owed_str = "` / `".join(owed)
    if verdict.kind == "stall":
        return (
            f"{_STALL_NUDGE_INTRO} Phase {where} requires calling `{owed_str}` to proceed, "
            f"but you called no such tool. CALL `{owed_str}` now — do not describe doing it "
            "in prose."
        )
    if verdict.kind == "wrong_edge":
        return (
            f"{_STALL_NUDGE_INTRO} Phase {where} requires `{owed_str}`, not the card you "
            f"emitted. CALL `{owed_str}` to proceed."
        )
    if verdict.kind == "not_solo":
        return (
            f"{_STALL_NUDGE_INTRO} The hand-off `{owed_str}` must be the ONLY action this "
            "turn — no prose and no other tool calls. CALL it alone."
        )
    return None


async def _maybe_nudge_subagent_stall(
    state: SessionState,
    session_id: str,
    body: MessageRequest,
    latest_parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """If the just-finished subagent turn stalled on an owed tool edge, re-prompt ONCE on the
    system channel and return the re-fetched parts; otherwise return `latest_parts` unchanged. Runs
    BEFORE completion/help detection so a nudged request_orchestrator_help flows through the normal
    hand-off path (not stripped as prose). One nudge per user turn — the re-driven turn is not
    re-adjudicated, so it can never loop. This is the live consumer of the pure `adjudicate`
    library (api/phases.py)."""
    if body.event == "job_complete":
        return latest_parts  # carries its own terminal hand-back steer; not a stall
    if not state.subagent_id or not state.subagent_target:
        return latest_parts
    verdict = _adjudicate_subagent_turn(state, state.subagent_target, latest_parts)
    nudge = _stall_nudge_text(state.last_signal_step, verdict)
    if nudge is None:
        return latest_parts
    logger.info(
        "Stall nudge: session=%s target=%s kind=%s step=%s",
        session_id,
        state.subagent_target,
        verdict.kind,
        state.last_signal_step,
    )
    try:
        await _proxy_send_message(
            state.subagent_id,
            "",
            agent=state.subagent_target,
            model=body.model,
            system=nudge,
        )
    except Exception:
        logger.warning("Stall nudge send failed: session=%s", session_id, exc_info=True)
        return latest_parts
    return await _fetch_all_assistant_parts(state.subagent_id)


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
    steer_system: str | None = None
    if body.event == "job_complete":
        # The neutral event fact (the job id, which the hand-back must report and the grounding
        # corpus needs) stays in the user part. The hand-back IMPERATIVE ("your task is done —
        # signal completion now, don't inspect") is delivered on the SYSTEM channel for every
        # driver, not appended to the user text: it is a system-initiated instruction, so it
        # belongs in the system channel rather than masquerading as a user turn. The standing
        # hand-back rule also lives in the subagent persona (agents/*/*.md), so this per-turn
        # system note only reinforces it. See docs/agent-gates-proposal.md.
        succeeded = _job_succeeded(body, user_text)
        job_failed = not succeeded
        steer_system = _SUBAGENT_JOB_DONE_PROMPT if succeeded else _SUBAGENT_JOB_FAILED_PROMPT
    # On a job_complete turn the terminal hand-back steer owns the system channel (it tells the
    # finishing subagent to signal completion now). On every other turn, re-anchor the driver to
    # its resident workflow section with the pointer (see phases.subagent_pointer).
    system = steer_system or _pointer_for(state, state.subagent_target)
    logger.info("Routing to subagent: session=%s target=%s", session_id, state.subagent_target)
    try:
        result = await _proxy_send_message(
            state.subagent_id,  # type: ignore[arg-type]
            user_text,
            agent=state.subagent_target,
            model=body.model,
            system=system,
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
    # If the turn stalled on an owed tool edge (prose instead of the transition), re-prompt ONCE
    # on the system channel and re-fetch, so the corrected turn flows through the completion/help/
    # ordinary handling below (a nudged request_orchestrator_help becomes a real hand-off, not
    # stripped prose). The live consumer of the pure adjudicator — see _maybe_nudge_subagent_stall.
    latest_parts = await _maybe_nudge_subagent_stall(state, session_id, body, latest_parts)
    # Completion is checked before a help request: if a turn somehow carries both
    # signals, "I'm done" wins and the help ask is ignored (a subagent that is
    # finishing has nothing left to be helped with).
    summary = _detect_completion(latest_parts)

    if summary:
        logger.info("Subagent completion signal received: session=%s", session_id)
        completer = state.subagent_target
        if completer and state.subagent_id:
            state.completed_subagents[completer] = state.subagent_id
            state.completed_last_step[completer] = state.last_signal_step
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
            morty_result = await _orchestrator_turn(state, resume_prompt, body)
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
    help_request = _detect_help_request(latest_parts)
    if help_request and state.subagent_target and state.subagent_id:
        need, help_context = help_request
        target = state.subagent_target
        logger.info("Subagent help request: session=%s target=%s", session_id, target)
        state.completed_subagents[target] = state.subagent_id
        state.completed_last_step[target] = state.last_signal_step
        state.subagent_id = None
        state.subagent_target = None
        seed = _SUBAGENT_HELP_PROMPT.format(target=target, need=need, context=help_context)
        morty_result = await _orchestrator_turn(state, seed, body)
        delegation = _detect_delegation(await _fetch_all_assistant_parts(state.orchestrator_id))
        # The leaf's [USER MESSAGE] is the dataset ask (need), NOT the orchestrator seed —
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
        help_result["parts"] = response_parts + _strip_internal_tools(
            help_result.get("parts", [])
        )
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
    system: str | None = None,
) -> dict[str, Any]:
    return await _proxy_send_message(
        state.orchestrator_id, text, agent="morty", model=body.model, system=system
    )


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

    advisory = await _delegation_advisory(state, target, context, user_text)
    if advisory:
        context = f"{advisory}\n\n{context}" if context else advisory

    stashed_id = state.completed_subagents.pop(target, None) if resume else None

    if stashed_id:
        logger.info("Resuming subagent: target=%s session=%s → %s", target, session_id, stashed_id)
        handoff_msg = f"[RESUMED]\n{context}\n\n[USER MESSAGE]\n{user_text}"
        # A resume picks up where it left off, so restore THIS target's own last signalled step
        # (a help detour / later delegation overwrote the shared field) before the pointer names
        # the section. Falls back to "" (entry) when nothing was stashed.
        state.last_signal_step = state.completed_last_step.pop(target, "")
        sub_result = await _proxy_send_message(
            stashed_id,
            handoff_msg,
            agent=target,
            model=body.model,
            system=_pointer_for(state, target),
        )
        state.subagent_id = stashed_id
    else:
        logger.info("New subagent: target=%s session=%s", target, session_id)
        sub_id = await _proxy_create_session()
        handoff_msg = f"[CONTEXT]\n{context}\n\n[USER MESSAGE]\n{user_text}"
        # Fresh delegation: clear any prior subagent's phase and skill state so the pointer (and
        # the UI progress backfill) start at this agent's entry section, not wherever the last one
        # stopped.
        state.last_signal_step = ""
        state.completed_last_step.pop(target, None)
        state.active_skill_by_target.pop(target, None)
        state.skill_last_step.pop(target, None)
        sub_result = await _proxy_send_message(
            sub_id,
            handoff_msg,
            agent=target,
            model=body.model,
            system=_pointer_for(state, target),
        )
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
