"""Agent session proxy with subagent routing.

Proxies chat messages to OpenCode and intercepts delegation/completion
signals to route conversations between the orchestrator and ephemeral
subagent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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


@dataclass
class SessionState:
    orchestrator_id: str
    subagent_id: str | None = None
    subagent_target: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    completed_subagents: dict[str, str] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))


_sessions: dict[str, SessionState] = {}

# ---------------------------------------------------------------------------
# Shared HTTP client
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None
_cleanup_task: asyncio.Task[None] | None = None


async def startup() -> None:
    global _http_client, _cleanup_task
    _http_client = httpx.AsyncClient(timeout=OPENCODE_TIMEOUT)
    _cleanup_task = asyncio.create_task(_session_cleanup_loop())


async def shutdown() -> None:
    global _http_client, _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _cleanup_task
        _cleanup_task = None
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


def _detect_delegation(parts: list[dict[str, Any]]) -> tuple[str, str, bool] | None:
    for part in parts:
        if part.get("type") != "tool":
            continue
        if _tool_name(part) == "delegate_to_subagent":
            inp = _get_tool_input(part)
            return (inp.get("target", ""), inp.get("context", ""), inp.get("resume", False))
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


@router.post("/session/{session_id}/message")
async def send_message(session_id: str, body: MessageRequest) -> dict[str, Any]:
    user_text = _extract_user_text(body)
    if not user_text:
        raise HTTPException(status_code=400, detail="no text part in message")

    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")

    state.last_activity = datetime.now(UTC)

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
            "Present contextual next steps to the user via present_options."
        )
        orch_result = await _proxy_send_message(
            state.orchestrator_id, resume_prompt, agent="morty", model=body.model
        )
        response_parts = _strip_internal_tools(result.get("parts", []))
        result["parts"] = response_parts + orch_result.get("parts", [])
        result["info"] = orch_result.get("info", result.get("info", {}))

    return result


async def _handle_orchestrator_message(
    state: SessionState,
    session_id: str,
    user_text: str,
    body: MessageRequest,
) -> dict[str, Any]:
    result = await _proxy_send_message(
        state.orchestrator_id, user_text, agent="morty", model=body.model
    )

    latest_parts = await _fetch_all_assistant_parts(state.orchestrator_id)
    delegation = _detect_delegation(latest_parts)

    if delegation:
        target, context, resume = delegation

        stashed_id = state.completed_subagents.pop(target, None) if resume else None

        if stashed_id:
            logger.info(
                "Resuming subagent: target=%s session=%s → %s", target, session_id, stashed_id
            )
            handoff_msg = f"[RESUMED]\n{context}\n\n[USER MESSAGE]\n{user_text}"
            sub_result = await _proxy_send_message(
                stashed_id, handoff_msg, agent=target, model=body.model
            )
            state.subagent_id = stashed_id
        else:
            logger.info("New subagent: target=%s session=%s", target, session_id)
            sub_id = await _proxy_create_session()
            handoff_msg = f"[CONTEXT]\n{context}\n\n[USER MESSAGE]\n{user_text}"
            sub_result = await _proxy_send_message(
                sub_id, handoff_msg, agent=target, model=body.model
            )
            state.subagent_id = sub_id

        state.subagent_target = target
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
