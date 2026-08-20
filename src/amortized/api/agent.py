"""Agent session proxy with subagent routing.

Proxies chat messages to OpenCode and intercepts delegation/completion
signals to route conversations between the orchestrator and ephemeral
subagent sessions.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from amortized.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])

OPENCODE_TIMEOUT = 300.0

_orchestrator_sessions: dict[str, str] = {}
_active_subagents: dict[str, str] = {}
_subagent_targets: dict[str, str] = {}
_session_locks: dict[str, asyncio.Lock] = {}
_session_locks_lock = asyncio.Lock()

# Per-session signal queues — keyed by external session_id, not "latest"
_pending_delegations: dict[str, tuple[str, str]] = {}
_pending_completions: dict[str, str] = {}

_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=OPENCODE_TIMEOUT)
    return _http_client


def queue_delegation(target: str, context: str, session_id: str) -> None:
    """Called by the MCP delegate_to_subagent endpoint."""
    _pending_delegations[session_id] = (target, context)
    logger.info("Delegation queued: target=%s session=%s", target, session_id)


def queue_completion(summary: str, session_id: str) -> None:
    """Called by the MCP signal_subagent_completion endpoint."""
    _pending_completions[session_id] = summary
    logger.info("Completion queued: session=%s", session_id)


def _resolve_session_id_for_mcp() -> str | None:
    """Resolve the external session_id for the currently active MCP call.

    When an MCP tool fires during _proxy_send_message, we need to know
    which external session triggered it. We track this via _active_mcp_session.
    """
    return _active_mcp_session


_active_mcp_session: str | None = None


def _opencode_url() -> str:
    return settings.agent_upstream_url.rstrip("/")


async def _get_lock(session_id: str) -> asyncio.Lock:
    async with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]


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


async def _proxy_create_session() -> str:
    client = _get_client()
    resp = await client.post(f"{_opencode_url()}/session", timeout=10.0)
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
    client = _get_client()
    resp = await client.post(
        f"{_opencode_url()}/session/{session_id}/message",
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_user_text(body: MessageRequest) -> str:
    for part in body.parts:
        if part.type == "text" and part.text:
            return part.text
    return ""


@router.post("/session")
async def create_session() -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    opencode_id = await _proxy_create_session()
    _orchestrator_sessions[session_id] = opencode_id
    return {"id": session_id}


@router.get("/session/{session_id}/message")
async def get_session_messages(session_id: str) -> Any:
    """Proxy GET to the active OpenCode session to fetch message history."""
    active_sub = _active_subagents.get(session_id)
    target_id = active_sub or _orchestrator_sessions.get(session_id)
    if not target_id:
        raise HTTPException(status_code=404, detail="unknown session")
    try:
        client = _get_client()
        resp = await client.get(f"{_opencode_url()}/session/{target_id}/message", timeout=10.0)
        if resp.status_code == 404:
            return {"info": {}, "parts": []}
        resp.raise_for_status()
        if not resp.content:
            return {"info": {}, "parts": []}
        return resp.json()
    except httpx.HTTPStatusError:
        logger.exception("Upstream error fetching messages for session %s", session_id)
        return {"info": {}, "parts": []}
    except httpx.HTTPError:
        logger.warning("Upstream unreachable fetching messages for session %s", session_id)
        return {"info": {}, "parts": []}


@router.post("/session/{session_id}/message")
async def send_message(session_id: str, body: MessageRequest) -> dict[str, Any]:
    global _active_mcp_session

    user_text = _extract_user_text(body)
    if not user_text:
        raise HTTPException(status_code=400, detail="no text part in message")

    lock = await _get_lock(session_id)
    async with lock:
        active_sub = _active_subagents.get(session_id)

        if active_sub:
            target = _subagent_targets.get(session_id, "")
            logger.info("Routing to subagent: session=%s target=%s", session_id, target)
            _pending_completions.pop(session_id, None)
            _active_mcp_session = session_id
            try:
                result = await _proxy_send_message(
                    active_sub, user_text, agent=target, model=body.model
                )
            except Exception:
                logger.exception("Subagent message failed, tearing down: session=%s", session_id)
                _active_subagents.pop(session_id, None)
                _subagent_targets.pop(session_id, None)
                raise
            finally:
                _active_mcp_session = None

            completion = _pending_completions.pop(session_id, None)
            if completion:
                logger.info("Subagent completion signal received: session=%s", session_id)
                _active_subagents.pop(session_id, None)
                _subagent_targets.pop(session_id, None)
                orch_id = _orchestrator_sessions.get(session_id)
                if orch_id:
                    resume_prompt = (
                        f"[SUBAGENT COMPLETED]\n{completion}\n\n"
                        "Present contextual next steps to the user via present_options."
                    )
                    orch_result = await _proxy_send_message(
                        orch_id, resume_prompt, agent="morty", model=body.model
                    )
                    result.setdefault("parts", []).extend(orch_result.get("parts", []))
                    result["info"] = orch_result.get("info", result.get("info", {}))

            return result

        orch_id = _orchestrator_sessions.get(session_id)
        if not orch_id:
            raise HTTPException(status_code=404, detail="unknown session")

        _pending_delegations.pop(session_id, None)
        _active_mcp_session = session_id
        try:
            result = await _proxy_send_message(orch_id, user_text, agent="morty", model=body.model)
        finally:
            _active_mcp_session = None

        delegation = _pending_delegations.pop(session_id, None)
        if delegation:
            target, context = delegation
            logger.info("Delegation detected: target=%s session=%s", target, session_id)
            sub_id = await _proxy_create_session()
            handoff_msg = f"[CONTEXT]\n{context}\n\n[USER MESSAGE]\n{user_text}"
            sub_result = await _proxy_send_message(
                sub_id, handoff_msg, agent=target, model=body.model
            )
            _active_subagents[session_id] = sub_id
            _subagent_targets[session_id] = target
            logger.info("Subagent active: %s → %s (%s)", session_id, sub_id, target)
            return sub_result

        return result


@router.get("/session/{session_id}/pending")
async def get_pending(session_id: str) -> dict[str, Any]:
    active_sub = _active_subagents.get(session_id)
    target_id = active_sub or _orchestrator_sessions.get(session_id)
    if not target_id:
        raise HTTPException(status_code=404, detail="unknown session")
    try:
        client = _get_client()
        resp = await client.get(f"{_opencode_url()}/session/{target_id}/pending", timeout=10.0)
        if resp.status_code == 404:
            return {"messages": []}
        resp.raise_for_status()
        if not resp.content:
            return {"messages": []}
        return resp.json()
    except httpx.HTTPStatusError:
        logger.exception("Upstream error fetching pending for session %s", session_id)
        return {"messages": []}
    except httpx.HTTPError:
        logger.warning("Upstream unreachable fetching pending for session %s", session_id)
        return {"messages": []}


@router.post("/title")
async def generate_title() -> dict[str, Any]:
    return {"title": ""}


@router.get("/health")
async def agent_health() -> dict[str, Any]:
    try:
        client = _get_client()
        resp = await client.get(f"{_opencode_url()}/api/health", timeout=5.0)
        upstream = resp.json() if resp.status_code == 200 else {"healthy": False}
    except Exception:
        upstream = {"healthy": False}
    return {"healthy": upstream.get("healthy", False), "proxy": True}
