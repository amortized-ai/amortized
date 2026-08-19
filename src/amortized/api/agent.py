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
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{_opencode_url()}/session")
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
    async with httpx.AsyncClient(timeout=OPENCODE_TIMEOUT) as client:
        resp = await client.post(
            f"{_opencode_url()}/session/{session_id}/message",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


def _detect_delegation(parts: list[dict[str, Any]]) -> tuple[str, str] | None:
    for part in parts:
        if part.get("type") != "tool" or part.get("state", {}).get("status") != "completed":
            continue
        tool = part.get("tool", "")
        if tool.endswith("delegate_to_subagent"):
            inp = part.get("state", {}).get("input", {})
            return inp.get("target", ""), inp.get("context", "")
    return None


def _detect_completion(parts: list[dict[str, Any]]) -> str | None:
    for part in parts:
        if part.get("type") != "tool" or part.get("state", {}).get("status") != "completed":
            continue
        tool = part.get("tool", "")
        if tool.endswith("signal_subagent_completion"):
            inp = part.get("state", {}).get("input", {})
            return inp.get("summary", "")
    return None


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
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_opencode_url()}/session/{target_id}/message")
            if resp.status_code == 404:
                return {"info": {}, "parts": []}
            resp.raise_for_status()
            if not resp.content:
                return {"info": {}, "parts": []}
            return resp.json()
    except Exception:
        return {"info": {}, "parts": []}


@router.post("/session/{session_id}/message")
async def send_message(session_id: str, body: MessageRequest) -> dict[str, Any]:
    user_text = _extract_user_text(body)
    if not user_text:
        raise HTTPException(status_code=400, detail="no text part in message")

    lock = await _get_lock(session_id)
    async with lock:
        active_sub = _active_subagents.get(session_id)

        if active_sub:
            target = _subagent_targets.get(session_id, "")
            logger.info("Routing to subagent: session=%s target=%s", session_id, target)
            result = await _proxy_send_message(
                active_sub, user_text, agent=target, model=body.model
            )
            parts = result.get("parts", [])

            summary = _detect_completion(parts)
            if summary:
                _active_subagents.pop(session_id, None)
                _subagent_targets.pop(session_id, None)
                orch_id = _orchestrator_sessions.get(session_id)
                if orch_id:
                    resume_prompt = (
                        f"[SUBAGENT COMPLETED]\n{summary}\n\n"
                        "Present contextual next steps to the user via present_options."
                    )
                    orch_result = await _proxy_send_message(
                        orch_id, resume_prompt, agent="morty", model=body.model
                    )
                    parts.extend(orch_result.get("parts", []))
                    result["info"] = orch_result.get("info", result.get("info", {}))

            return result

        orch_id = _orchestrator_sessions.get(session_id)
        if not orch_id:
            raise HTTPException(status_code=404, detail="unknown session")

        result = await _proxy_send_message(orch_id, user_text, agent="morty", model=body.model)
        parts = result.get("parts", [])

        for p in parts:
            if p.get("type") == "tool":
                logger.info(
                    "Tool part: tool=%s state=%s",
                    p.get("tool"),
                    p.get("state"),
                )

        delegation = _detect_delegation(parts)
        if delegation:
            target, context = delegation
            logger.info("Delegation detected: target=%s session=%s", target, session_id)
            sub_id = await _proxy_create_session()
            logger.info("Subagent session created: %s for %s", sub_id, session_id)
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_opencode_url()}/session/{target_id}/pending")
            if resp.status_code == 404:
                return {"messages": []}
            resp.raise_for_status()
            if not resp.content:
                return {"messages": []}
            return resp.json()
    except Exception:
        return {"messages": []}


@router.post("/title")
async def generate_title() -> dict[str, Any]:
    return {"title": ""}


@router.get("/health")
async def agent_health() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_opencode_url()}/api/health")
            upstream = resp.json() if resp.status_code == 200 else {"healthy": False}
    except Exception:
        upstream = {"healthy": False}
    return {"healthy": upstream.get("healthy", False), "proxy": True}
