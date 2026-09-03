"""Agent session proxy with subagent routing.

Proxies chat messages to OpenCode and intercepts delegation/completion
signals to route conversations between the orchestrator and ephemeral
subagent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import ssl
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
    finished_at: datetime | None = None
    task: asyncio.Task[None] | None = None


@dataclass
class SessionState:
    orchestrator_id: str
    subagent_id: str | None = None
    subagent_target: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    completed_subagents: dict[str, str] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    turns: dict[str, TurnState] = field(default_factory=dict)
    turn_order: list[str] = field(default_factory=list)


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


_VALID_TARGETS = {"sdg", "training"}


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
        if turn:
            turn.result = result
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
    }


async def _handle_subagent_message(
    state: SessionState,
    session_id: str,
    user_text: str,
    body: MessageRequest,
) -> dict[str, Any]:
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
        result["parts"] = response_parts + resume_result.get("parts", [])
        result["info"] = resume_result.get("info", result.get("info", {}))

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
