"""Agent session proxy with subagent routing.

Proxies chat messages to OpenCode and intercepts delegation/completion
signals to route conversations between the orchestrator and ephemeral
subagent sessions.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from amortized.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])

OPENCODE_TIMEOUT = 300.0

_current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_session", default=None
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    orchestrator_id: str
    subagent_id: str | None = None
    subagent_target: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_delegation: tuple[str, str] | None = None
    pending_completion: str | None = None


_sessions: dict[str, SessionState] = {}

# ---------------------------------------------------------------------------
# Shared HTTP client
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


async def startup() -> None:
    global _http_client
    _http_client = httpx.AsyncClient(timeout=OPENCODE_TIMEOUT)


async def shutdown() -> None:
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


def _client() -> httpx.AsyncClient:
    assert _http_client is not None, "agent proxy not started"
    return _http_client


# ---------------------------------------------------------------------------
# Signal queues (called from MCP endpoints in ui.py)
# ---------------------------------------------------------------------------


def queue_delegation(target: str, context: str) -> None:
    sid = _current_session.get()
    if not sid or sid not in _sessions:
        return
    _sessions[sid].pending_delegation = (target, context)
    logger.info("Delegation queued: target=%s session=%s", target, sid)


def queue_completion(summary: str) -> None:
    sid = _current_session.get()
    if not sid or sid not in _sessions:
        return
    _sessions[sid].pending_completion = summary
    logger.info("Completion queued: session=%s", sid)


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
        if not resp.content:
            return empty
        return resp.json()
    except httpx.HTTPStatusError:
        logger.exception("Upstream error on GET %s: session=%s", path, session_id)
        return empty
    except httpx.HTTPError:
        logger.warning("Upstream unreachable on GET %s: session=%s", path, session_id)
        return empty


async def _proxy_create_session() -> str:
    resp = await _client().post(f"{_opencode_url()}/session", timeout=10.0)
    resp.raise_for_status()
    return resp.json()["id"]


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
    return resp.json()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MessagePart(BaseModel):
    type: str
    text: str | None = None


class MessageModel(BaseModel):
    providerID: str | None = None
    modelID: str | None = None


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
    opencode_id = await _proxy_create_session()
    _sessions[session_id] = SessionState(orchestrator_id=opencode_id)
    return {"id": session_id}


@router.get("/session/{session_id}/message")
async def get_session_messages(session_id: str) -> Any:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")
    target_id = state.subagent_id or state.orchestrator_id
    return await _proxy_get(target_id, "message", {"info": {}, "parts": []}, session_id)


@router.post("/session/{session_id}/message")
async def send_message(session_id: str, body: MessageRequest) -> dict[str, Any]:
    user_text = _extract_user_text(body)
    if not user_text:
        raise HTTPException(status_code=400, detail="no text part in message")

    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")

    async with state.lock:
        if state.subagent_id:
            return await _handle_subagent_message(state, session_id, user_text, body)
        return await _handle_orchestrator_message(state, session_id, user_text, body)


async def _handle_subagent_message(
    state: SessionState,
    session_id: str,
    user_text: str,
    body: MessageRequest,
) -> dict[str, Any]:
    logger.info("Routing to subagent: session=%s target=%s", session_id, state.subagent_target)
    state.pending_completion = None
    _current_session.set(session_id)
    try:
        result = await _proxy_send_message(
            state.subagent_id,  # type: ignore[arg-type]
            user_text,
            agent=state.subagent_target,
            model=body.model,
        )
    except Exception:
        logger.exception("Subagent message failed, tearing down: session=%s", session_id)
        state.subagent_id = None
        state.subagent_target = None
        raise
    finally:
        _current_session.set(None)

    if state.pending_completion:
        completion = state.pending_completion
        state.pending_completion = None
        logger.info("Subagent completion signal received: session=%s", session_id)
        state.subagent_id = None
        state.subagent_target = None

        resume_prompt = (
            f"[SUBAGENT COMPLETED]\n{completion}\n\n"
            "Present contextual next steps to the user via present_options."
        )
        orch_result = await _proxy_send_message(
            state.orchestrator_id, resume_prompt, agent="morty", model=body.model
        )
        result.setdefault("parts", []).extend(orch_result.get("parts", []))
        result["info"] = orch_result.get("info", result.get("info", {}))

    return result


async def _handle_orchestrator_message(
    state: SessionState,
    session_id: str,
    user_text: str,
    body: MessageRequest,
) -> dict[str, Any]:
    state.pending_delegation = None
    _current_session.set(session_id)
    try:
        result = await _proxy_send_message(
            state.orchestrator_id, user_text, agent="morty", model=body.model
        )
    finally:
        _current_session.set(None)

    if state.pending_delegation:
        target, context = state.pending_delegation
        state.pending_delegation = None
        logger.info("Delegation detected: target=%s session=%s", target, session_id)

        sub_id = await _proxy_create_session()
        handoff_msg = f"[CONTEXT]\n{context}\n\n[USER MESSAGE]\n{user_text}"
        sub_result = await _proxy_send_message(sub_id, handoff_msg, agent=target, model=body.model)
        state.subagent_id = sub_id
        state.subagent_target = target
        logger.info("Subagent active: %s → %s (%s)", session_id, sub_id, target)
        return sub_result

    return result


@router.get("/session/{session_id}/pending")
async def get_pending(session_id: str) -> dict[str, Any]:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")
    target_id = state.subagent_id or state.orchestrator_id
    return await _proxy_get(target_id, "pending", {"messages": []}, session_id)


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
