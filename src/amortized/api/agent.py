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

from amortized.api.agent_parts import _get_tool_input, _tool_name
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
    skill_optional_step_indices,
    skill_step_block,
    skill_step_count,
    skill_step_index_for_slug,
    skill_step_title,
    subagent_phase,
    system_reminder,
    with_block,
)
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
    # Per-target stepped-skill injection state (phase-injection). active_skill_by_target:
    # the opencode-native skill a target loaded (whose per-step detail is injected one step
    # per gather turn). skill_steps_done: the step indices already covered, so the lowest
    # UNCOVERED step is injected next (a hole left by a jumped-ahead driver is filled first).
    # skill_last_asked: the last present_options.step label seen, so an interaction rerun
    # re-observing the same card does not advance the cursor twice.
    active_skill_by_target: dict[str, str] = field(default_factory=dict)
    skill_steps_done: dict[str, set[int]] = field(default_factory=dict)
    skill_last_asked: dict[str, str] = field(default_factory=dict)


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

# A finished SDG job in PREVIEW mode is only the cheap ~10-row sample, NOT the deliverable.
# The generic done-prompt above would (wrongly) tell the producer its task is complete, so it
# hands back and the orchestrator advances to training on a 10-row preview — the full create
# run never happens. For a preview completion, steer the producer to run the FULL create run
# instead of handing back. Only a create (full) SDG run — or any training/eval job — is a real
# completion. See _handle_subagent_message and _completed_job_is_sdg_preview.
_SUBAGENT_PREVIEW_DONE_PROMPT = (
    "[SYSTEM EVENT] Your PREVIEW sample finished successfully — this was only a small test "
    "run, NOT the full dataset. Show the user the preview output so they can eyeball quality, "
    'then call validate_sdg_job with mode "create" to launch the FULL dataset run. Do NOT call '
    "signal_subagent_completion and do NOT hand control back: your task is NOT complete until "
    "the full create run succeeds."
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


_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{8,}")


def _extract_id_candidates(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _ID_RE.findall(text.lower()):
        if "-" in tok:
            out.add(tok)  # dash-structured UUID — unambiguously an identifier
            continue
        # A bare hex run is only id-like if it mixes digits AND hex letters. This excludes
        # both pure-decimal counts ("12345678" records) and pure-hex English words
        # ("deadbeef") that would otherwise be false-flagged as ids.
        if any(c in "0123456789" for c in tok) and any(c in "abcdef" for c in tok):
            out.add(tok)
    return out


async def _fetch_job_by_token(token: str) -> dict[str, Any] | None:
    """The DB job a stated id-token resolves to (by id prefix), or None. Best-effort —
    any DB issue returns None so the caller simply falls back to default behavior."""
    try:
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository

        async with get_pool().acquire() as conn:
            return await Repository(conn).find_job_by_id_prefix(token)
    except Exception:
        return None


async def _completed_job_is_sdg_preview(user_text: str) -> bool:
    """Whether the job named in a job_complete event is an SDG job run in `preview` mode.

    The event text carries the job id; resolve it and read its config mode. A preview is the
    cheap sample, not the deliverable, so its completion must NOT hand control back (see
    _SUBAGENT_PREVIEW_DONE_PROMPT). Best-effort — any lookup miss returns False, so the normal
    done-prompt applies (fail toward the existing behavior)."""
    for token in sorted(_extract_id_candidates(user_text)):
        job = await _fetch_job_by_token(token)
        if not job or str(job.get("type") or "") != "sdg":
            continue
        cfg = job.get("config")
        if isinstance(cfg, dict) and str(cfg.get("mode") or "").lower() == "preview":
            return True
    return False


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

# The phase during which a stepped skill's per-step detail is surfaced: requirement
# gathering. Per-step injection is confined to this phase so leftover step slices can't
# bleed into the later config-building/validate turns (whose detail stays in the SKILL.md
# body). Matches the `signal_phase.step` slug the sdg/training gather phase reports.
_GATHER_PHASE = "phase-2-gather"

# How many leading steps of a stepped skill ship INSIDE the served SKILL.md (appended by
# gen_native_skills), hence are already present at load and must NOT be re-injected. Keep in
# sync with the Makefile append (currently the first step only).
_RESIDENT_SKILL_STEPS = 1


def _next_skill_step(skill: str | None, done: set[int], count: int) -> int:
    """The index of the next step to inject: the lowest UNCOVERED step, so a hole left by a
    driver that jumped ahead is filled before any later step is re-surfaced. Resident step(s)
    (baked into SKILL.md at build time) are skipped. Required steps come first; an optional step
    is only offered once every required step is covered (it never blocks). Returns `count` when
    nothing is left to inject."""
    optional = skill_optional_step_indices(skill)
    injectable = range(_RESIDENT_SKILL_STEPS, count)
    required_gap = [i for i in injectable if i not in done and i not in optional]
    if required_gap:
        return min(required_gap)
    optional_gap = [i for i in injectable if i not in done and i in optional]
    return min(optional_gap) if optional_gap else count


def _with_active_skill_step(state: SessionState, target: str, phase: str | None, text: str) -> str:
    """Prepend the active stepped skill's next step-detail block — plus a short list of the
    steps still outstanding — to a gather turn.

    "Next" is the lowest UNCOVERED step (see _next_skill_step), so a driver that jumped ahead is
    pulled back to fill the skipped step rather than losing it. Inject-only: it reads the coverage
    set but never mutates it — advancing is content-driven and happens when the model's RESPONSE
    is processed (_advance_skill_cursor).

    No-op outside the gather phase, when no skill is active, or when the active skill is not
    stepped (opt-in by `steps/` presence) — so callers wrap every gather turn without first
    checking skill state."""
    if phase != _GATHER_PHASE:
        return text
    skill = state.active_skill_by_target.get(target)
    count = skill_step_count(skill)
    if not count:
        return text
    done = state.skill_steps_done.get(target, set())
    nxt = _next_skill_step(skill, done, count)
    block = skill_step_block(skill, nxt)
    # The HOLES a driver left by jumping ahead: required steps it skipped (uncovered and below
    # the furthest step it reached), excluding the one being injected now. Future steps it simply
    # hasn't reached yet are NOT listed (the resident outline already shows those) — only the
    # ones it owes. So a jumped-ahead driver is told exactly what to come back for.
    optional = skill_optional_step_indices(skill)
    frontier = max(done) if done else _RESIDENT_SKILL_STEPS - 1
    outstanding: list[str] = []
    for i in range(_RESIDENT_SKILL_STEPS, frontier):
        if i == nxt or i in done or i in optional:
            continue
        title = skill_step_title(skill, i)
        if title:
            outstanding.append(title)
    if not block and not outstanding:
        return text
    sections: list[str] = []
    if outstanding:
        sections.append("Steps still to gather before building: " + "; ".join(outstanding) + ".")
    if block:
        sections.append(block)
    injected = system_reminder("\n\n".join(sections))
    return f"{injected}\n\n{text}" if text else injected


def _advance_skill_cursor(state: SessionState, target: str, parts: list[dict[str, Any]]) -> None:
    """Mark the steps the model actually reached this turn as covered.

    The driver stamps each gather question's `present_options.step` with the current step's slug
    (the step blocks instruct it to). Resolve those slugs to indices and add them to the coverage
    set — so a bundled turn marks several, a jumped-to step is recorded (leaving the skipped ones
    uncovered for re-injection), and a stall/re-ask is a no-op (adding an index already present).

    Fallback: when a stamped value matches no step slug (the driver used its own label), degrade
    to marking the step we were injecting as covered — but only on a NEW label, so the interaction
    rerun loop re-observing the same card does not advance twice."""
    skill = state.active_skill_by_target.get(target)
    count = skill_step_count(skill)
    if not count:
        return
    asked = [
        str(_get_tool_input(p).get("step") or "").strip()
        for p in parts
        if _tool_name(p) == "present_options"
    ]
    asked = [s for s in asked if s]
    if not asked:
        return  # no question posed → no evidence of progress; coverage unchanged
    done = state.skill_steps_done.setdefault(target, set())
    recognized = {i for s in asked if (i := skill_step_index_for_slug(skill, s)) >= 0}
    if recognized:
        done.update(recognized)  # crisp content anchor: record exactly the steps reached
    elif asked[-1] != state.skill_last_asked.get(target):
        # unrecognized NEW label → cover the step currently being injected
        nxt = _next_skill_step(skill, done, count)
        if nxt < count:
            done.add(nxt)
    # else: unrecognized label already counted (rerun / unchanged) → hold
    state.skill_last_asked[target] = asked[-1]


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
    # Bind the stepped-skill cursor to the step this turn reached, from the question cards'
    # stamped step id. Only in the gather phase (where steps are injected), and after the loop
    # so a skill loaded / phase signalled in the same turn is already recorded.
    target = state.subagent_target
    if target and subagent_phase(target, state.last_signal_step) == _GATHER_PHASE:
        _advance_skill_cursor(state, target, parts)
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
        "error_code": turn.error_code,
    }


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
        if not succeeded:
            steer_system = _SUBAGENT_JOB_FAILED_PROMPT
        elif await _completed_job_is_sdg_preview(user_text):
            # A PREVIEW completion is not a real completion — the full create run hasn't
            # happened. Steer the producer to run it instead of handing back (which would
            # advance the orchestrator to training on the ~10-row preview). Any other success
            # (create SDG, training, eval) is a genuine hand-back.
            steer_system = _SUBAGENT_PREVIEW_DONE_PROMPT
        else:
            steer_system = _SUBAGENT_JOB_DONE_PROMPT
    else:
        # Re-inject the current phase's block each ongoing turn (not just at the first
        # handoff), so the active step's full instructions stay in the freshest position —
        # the thin workflow.md carries only the phase titles. Skipped on job_complete, which
        # carries its own terminal hand-back steer on the system channel above.
        target_for_phase = state.subagent_target or ""
        phase_for_turn = subagent_phase(target_for_phase, state.last_signal_step)
        # During gathering, surface the active stepped skill's current step detail (one step
        # per turn) just under the phase block, then the cursor advances when this turn's
        # response is processed (_advance_skill_cursor). The skill's titles + body stay
        # resident, so this only moves the freshest detail forward — it never hides the outline.
        user_text = _with_active_skill_step(state, target_for_phase, phase_for_turn, user_text)
        user_text = with_block(target_for_phase, phase_for_turn, user_text)
    logger.info("Routing to subagent: session=%s target=%s", session_id, state.subagent_target)
    try:
        result = await _proxy_send_message(
            state.subagent_id,  # type: ignore[arg-type]
            user_text,
            agent=state.subagent_target,
            model=body.model,
            system=steer_system,
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
    # Completion is checked before a help request: if a turn somehow carries both
    # signals, "I'm done" wins and the help ask is ignored (a subagent that is
    # finishing has nothing left to be helped with).
    summary = _detect_completion(latest_parts)

    if summary:
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
    help_request = _detect_help_request(latest_parts)
    if help_request and state.subagent_target and state.subagent_id:
        need, help_context = help_request
        target = state.subagent_target
        logger.info("Subagent help request: session=%s target=%s", session_id, target)
        state.completed_subagents[target] = state.subagent_id
        state.subagent_id = None
        state.subagent_target = None
        # The fulfill_help phase block is the orchestrator's seed template when present (it
        # carries the same {target}/{need}/{context} placeholders); fall back to the inline
        # prompt for an un-migrated orchestrator.
        seed_tmpl = phase_block("orchestrator", "fulfill_help") or _SUBAGENT_HELP_PROMPT
        seed = seed_tmpl.format(target=target, need=need, context=help_context)
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
        # Inject the block for the phase the subagent last signalled, so a resume lands it
        # back on the right step with that step's full instructions in the turn (the thin
        # workflow.md carries only titles).
        resume_phase = subagent_phase(target, state.last_signal_step)
        handoff_msg = f"[RESUMED]\n{context}\n\n[USER MESSAGE]\n{user_text}"
        handoff_msg = _with_active_skill_step(state, target, resume_phase, handoff_msg)
        handoff_msg = with_block(target, resume_phase, handoff_msg)
        sub_result = await _proxy_send_message(
            stashed_id, handoff_msg, agent=target, model=body.model
        )
        state.subagent_id = stashed_id
    else:
        logger.info("New subagent: target=%s session=%s", target, session_id)
        sub_id = await _proxy_create_session()
        # A fresh subagent starts at its entry phase: clear any step/skill state carried over
        # from a previous subagent so subagent_phase() resolves to the first section, then
        # inject that section's block as the lead of the handoff.
        state.last_signal_step = ""
        state.active_skill_by_target.pop(target, None)
        state.skill_steps_done.pop(target, None)
        state.skill_last_asked.pop(target, None)
        handoff_msg = f"[CONTEXT]\n{context}\n\n[USER MESSAGE]\n{user_text}"
        handoff_msg = with_block(
            target, subagent_phase(target, state.last_signal_step), handoff_msg
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
    # The orchestrator never calls signal_phase (it is a primary agent); the proxy picks its
    # phase by control-flow path. A fresh user turn is intake→delegate; a job-status event with
    # no active subagent is the monitor case. Inject the block ONLY into the text the
    # orchestrator reads — the clean user_text still flows to _maybe_delegate as the subagent's
    # user text, so orchestrator-only guidance never leaks downstream.
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
