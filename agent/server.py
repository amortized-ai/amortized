"""OpenCode-compatible wrapper around the Claude Agent SDK.

Exposes the same HTTP API that OpenCode's `serve` command provides,
so Amortized Studio can talk to either backend without code changes.

Endpoints:
  POST /session              → create a session
  POST /session/{id}/message → send a message, get a synchronous JSON response
  POST /session/{id}/event   → receive a job event, generate follow-up response
  GET  /session/{id}/pending → drain pending follow-up messages
  GET  /api/health           → health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

MORTY_PROMPT_PATH = Path(os.environ.get("MORTY_PROMPT_PATH", "/app/morty.md"))
CONFIG_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR", "/data"))
SESSION_MAP_PATH = CONFIG_DIR / "session_map.json"

MCP_AMORTIZED_URL = os.environ.get(
    "MCP_AMORTIZED_URL",
    "http://amortized-server.amortized.svc.cluster.local:8000/mcp",
)
MCP_MLFLOW_URL = os.environ.get("MCP_MLFLOW_URL", "http://127.0.0.1:5002/sse")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/app/workspace")

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
MAX_PENDING_PER_SESSION = 20
EVENT_COOLDOWN_SECONDS = 10


def _detect_provider_id() -> str:
    if os.environ.get("CLAUDE_CODE_USE_VERTEX"):
        return "google-vertex-anthropic"
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK"):
        return "amazon-bedrock-anthropic"
    if os.environ.get("CLAUDE_CODE_USE_FOUNDRY"):
        return "microsoft-foundry-anthropic"
    return "anthropic"


PROVIDER_ID = _detect_provider_id()
logger = logging.getLogger(__name__)

_session_map: dict[str, str] = {}
_map_lock = asyncio.Lock()
_morty_prompt: str = ""

_pending_messages: dict[str, list[dict[str, Any]]] = defaultdict(list)
_pending_lock = asyncio.Lock()
_session_locks: dict[str, asyncio.Lock] = {}
_session_locks_lock = asyncio.Lock()
_last_event_time: dict[str, float] = {}
_background_tasks: set[asyncio.Task[None]] = set()


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    async with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]


def _strip_frontmatter(text: str) -> str:
    """Strip YAML frontmatter (between --- markers) from a markdown file."""
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end == -1:
        return text
    return text[end + 3 :].lstrip("\n")


def _load_morty_prompt() -> str:
    if MORTY_PROMPT_PATH.is_dir():
        parts = []
        for f in sorted(MORTY_PROMPT_PATH.glob("*.md")):
            parts.append(f.read_text())
        if not parts:
            raise RuntimeError(f"No .md files found in {MORTY_PROMPT_PATH}")
        raw = "\n".join(parts)
    elif MORTY_PROMPT_PATH.exists():
        raw = MORTY_PROMPT_PATH.read_text()
    else:
        raise RuntimeError(f"Morty prompt not found at {MORTY_PROMPT_PATH}")
    return _strip_frontmatter(raw)


app = FastAPI(title="Claude Code Agent (OpenCode-compatible)")


PENDING_SWEEP_INTERVAL = 300


async def _sweep_pending() -> None:
    """Periodically remove pending messages for sessions that no longer exist."""
    while True:
        await asyncio.sleep(PENDING_SWEEP_INTERVAL)
        async with _pending_lock:
            stale = [sid for sid in _pending_messages if sid not in _session_map]
            for sid in stale:
                del _pending_messages[sid]
            if stale:
                logger.info("Swept %d stale pending queues", len(stale))


@app.on_event("startup")
async def _startup() -> None:
    global _morty_prompt
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if SESSION_MAP_PATH.exists():
        _session_map.update(json.loads(SESSION_MAP_PATH.read_text()))
    _morty_prompt = _load_morty_prompt()
    task = asyncio.create_task(_sweep_pending())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class SessionResponse(BaseModel):
    id: str


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


class JobEvent(BaseModel):
    type: str
    job_id: str
    status: str
    job_type: str = ""
    error: str | None = None


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"healthy": True, "version": "0.1.0"}


@app.post("/session")
async def create_session() -> SessionResponse:
    session_id = str(uuid.uuid4())
    return SessionResponse(id=session_id)


async def _run_agent(
    session_id: str, prompt: str, model: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run an agent query and return (response_parts, result_info)."""
    resolved_model = model or MODEL
    sdk_session_id = _session_map.get(session_id)

    options = ClaudeAgentOptions(
        system_prompt=_morty_prompt,
        allowed_tools=["mcp__*"],
        permission_mode="acceptEdits",
        setting_sources=[],
        model=resolved_model,
        cwd=WORKSPACE_DIR,
        mcp_servers={
            "amortized": {
                "type": "http",
                "url": MCP_AMORTIZED_URL,
            },
            "mlflow": {
                "type": "sse",
                "url": MCP_MLFLOW_URL,
            },
        },
    )

    if sdk_session_id:
        options.resume = sdk_session_id

    response_parts: list[dict[str, Any]] = []
    result_info: dict[str, Any] = {}
    new_sdk_session_id: str | None = None

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    response_parts.append(
                        {
                            "type": "tool",
                            "tool": getattr(block, "name", ""),
                            "callID": getattr(block, "id", ""),
                            "state": "running",
                            "input": getattr(block, "input", {}),
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    tool_use_id = getattr(block, "tool_use_id", None)
                    for rp in response_parts:
                        if rp.get("type") == "tool" and rp.get("callID") == tool_use_id:
                            rp["state"] = "completed"
                            rp["output"] = getattr(block, "content", None)
                            break

        elif isinstance(message, ResultMessage):
            new_sdk_session_id = getattr(message, "session_id", None)
            cost = getattr(message, "total_cost_usd", 0.0) or 0.0
            usage = getattr(message, "usage", None)
            input_tokens = 0
            output_tokens = 0
            if usage:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0

            result_info = {
                "providerID": PROVIDER_ID,
                "modelID": resolved_model,
                "cost": cost,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "reasoning": 0,
                },
                "finish": getattr(message, "subtype", "stop"),
                "id": str(uuid.uuid4()),
                "sessionID": session_id,
            }

    if new_sdk_session_id:
        await _persist_session(session_id, new_sdk_session_id)

    if not result_info:
        result_info = {
            "providerID": PROVIDER_ID,
            "modelID": resolved_model,
            "cost": 0,
            "tokens": {"input": 0, "output": 0, "reasoning": 0},
            "finish": "stop",
            "id": str(uuid.uuid4()),
            "sessionID": session_id,
        }

    return response_parts, result_info


def _extract_job_id_from_output(output: Any) -> str | None:
    """Extract job ID from MCP tool output in various formats."""
    if isinstance(output, str):
        try:
            obj = json.loads(output)
            if isinstance(obj, dict) and obj.get("id"):
                return obj["id"]
        except (json.JSONDecodeError, TypeError):
            pass
    if isinstance(output, list):
        for block in output:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    obj = json.loads(block.get("text", ""))
                    if isinstance(obj, dict) and obj.get("id"):
                        return obj["id"]
                except (json.JSONDecodeError, TypeError):
                    continue
    if isinstance(output, dict) and output.get("id"):
        return output["id"]
    return None


async def _auto_watch_jobs(session_id: str, response_parts: list[dict[str, Any]]) -> None:
    """Scan response for job creation tools and auto-register watches."""
    for part in response_parts:
        if part.get("type") != "tool" or part.get("state") != "completed":
            continue
        raw = (part.get("tool") or "")
        tool_name = raw.replace("mcp_amortized__", "").replace("amortized_", "")
        if tool_name not in (
            "create_sdg_job", "create_training_job", "submit_recipe_job",
        ):
            continue
        job_id = _extract_job_id_from_output(part.get("output"))
        if not job_id:
            continue
        base_url = MCP_AMORTIZED_URL.replace("/mcp", "")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = os.environ.get("AMORTIZED_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{base_url}/api/v1/ui/watch_job",
                    json={"job_id": job_id, "session_id": session_id},
                    headers=headers,
                )
                if resp.status_code != 200:
                    logger.warning(
                        "watch_job failed: %s %s", resp.status_code, resp.text[:200]
                    )
        except Exception:
            logger.warning(
                "watch_job request failed for job %s", job_id, exc_info=True
            )


def _sanitize_error(error: str | None, max_len: int = 200) -> str:
    if not error:
        return ""
    truncated = error[:max_len]
    if len(error) > max_len:
        truncated += "..."
    return truncated.replace("\n", " ").strip()


def _build_event_prompt(event: JobEvent) -> str:
    short_id = event.job_id[:8]
    jtype = event.job_type.upper() if event.job_type else ""
    job_label = f"{jtype} job #{short_id}" if jtype else f"job #{short_id}"

    if event.status == "running":
        return (
            f"[SYSTEM EVENT] {job_label} is now running. "
            f"Acknowledge briefly in 1 sentence. Do NOT call present_options."
        )
    elif event.status == "succeeded":
        return (
            f"[SYSTEM EVENT] {job_label} succeeded. "
            f"Congratulate briefly and call present_options with relevant next steps. "
            f"For SDG jobs: offer 'Preview dataset', 'Start training with this data'. "
            f"For training jobs: offer 'View model', 'View training metrics'. "
            f"Call signal_phase with step='review'. Keep the message to 1-2 sentences."
        )
    elif event.status in ("failed", "cancelled"):
        sanitized = _sanitize_error(event.error)
        error_detail = f" Error: {sanitized}" if sanitized else ""
        return (
            f"[SYSTEM EVENT] {job_label} {event.status}.{error_detail} "
            f"Explain briefly and call present_options with recovery options: "
            f"'View logs', 'Try again with different settings', 'Start fresh'."
        )
    return f"[SYSTEM EVENT] {job_label} status changed to {event.status}."


@app.post("/session/{session_id}/message")
async def send_message(session_id: str, body: MessageRequest) -> dict[str, Any]:
    user_text = ""
    for part in body.parts:
        if part.type == "text" and part.text:
            user_text = part.text
            break

    if not user_text:
        raise HTTPException(status_code=400, detail="no text part in message")

    model = MODEL
    if body.model and body.model.modelID:
        model = body.model.modelID.replace("@default", "") or MODEL

    lock = await _get_session_lock(session_id)
    async with lock:
        response_parts, result_info = await _run_agent(session_id, user_text, model)

    task = asyncio.create_task(_auto_watch_jobs(session_id, response_parts))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"info": result_info, "parts": response_parts}


@app.post("/session/{session_id}/event")
async def receive_event(session_id: str, body: JobEvent, request: Request) -> dict[str, Any]:
    expected_secret = os.environ.get("AGENT_EVENT_SECRET", "")
    if expected_secret:
        provided = request.headers.get("X-Event-Secret", "")
        if provided != expected_secret:
            raise HTTPException(status_code=403, detail="invalid event secret")

    if session_id not in _session_map:
        raise HTTPException(status_code=404, detail="unknown session")

    now = time.monotonic()
    last = _last_event_time.get(session_id, 0.0)
    if now - last < EVENT_COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail="event cooldown active")

    prompt = _build_event_prompt(body)

    lock = await _get_session_lock(session_id)
    async with lock:
        response_parts, result_info = await _run_agent(session_id, prompt)

    _last_event_time[session_id] = time.monotonic()

    follow_up = {"info": result_info, "parts": response_parts}

    async with _pending_lock:
        msgs = _pending_messages[session_id]
        if len(msgs) >= MAX_PENDING_PER_SESSION:
            msgs.pop(0)
        msgs.append(follow_up)

    return {"status": "queued", "message_count": len(_pending_messages[session_id])}


@app.get("/session/{session_id}/pending")
async def get_pending(session_id: str) -> dict[str, Any]:
    if session_id not in _session_map:
        raise HTTPException(status_code=404, detail="unknown session")
    async with _pending_lock:
        messages = _pending_messages.pop(session_id, [])
    return {"messages": messages}


async def _persist_session(external_id: str, sdk_session_id: str) -> None:
    async with _map_lock:
        if _session_map.get(external_id) == sdk_session_id:
            return
        _session_map[external_id] = sdk_session_id
        tmp = SESSION_MAP_PATH.with_suffix(".json.tmp")
        await asyncio.to_thread(tmp.write_text, json.dumps(_session_map))
        await asyncio.to_thread(tmp.replace, SESSION_MAP_PATH)
