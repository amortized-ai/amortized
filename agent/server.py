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
SDG_PROMPT_PATH = Path(os.environ.get("SDG_PROMPT_PATH", "/app/morty-sdg.md"))
TRAINING_PROMPT_PATH = Path(os.environ.get("TRAINING_PROMPT_PATH", "/app/morty-training.md"))
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

_orchestrator_sessions: dict[str, str] = {}
_active_subagents: dict[str, str] = {}
_subagent_targets: dict[str, str] = {}
_map_lock = asyncio.Lock()
_prompts: dict[str, str] = {}

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


def _load_prompt(path: Path) -> str:
    if path.is_dir():
        parts = []
        for f in sorted(path.glob("*.md")):
            parts.append(f.read_text())
        if not parts:
            raise RuntimeError(f"No .md files found in {path}")
        raw = "\n".join(parts)
    elif path.exists():
        raw = path.read_text()
    else:
        raise RuntimeError(f"Prompt not found at {path}")
    return _strip_frontmatter(raw)


def _load_all_prompts() -> dict[str, str]:
    prompts = {"orchestrator": _load_prompt(MORTY_PROMPT_PATH)}
    if SDG_PROMPT_PATH.exists():
        prompts["sdg"] = _load_prompt(SDG_PROMPT_PATH)
    else:
        logger.warning("SDG prompt not found at %s, delegation disabled", SDG_PROMPT_PATH)
    if TRAINING_PROMPT_PATH.exists():
        prompts["training"] = _load_prompt(TRAINING_PROMPT_PATH)
    else:
        logger.warning("Training prompt not found at %s, delegation disabled", TRAINING_PROMPT_PATH)
    return prompts


app = FastAPI(title="Claude Code Agent (OpenCode-compatible)")


PENDING_SWEEP_INTERVAL = 300


async def _sweep_pending() -> None:
    """Periodically remove pending messages for sessions that no longer exist."""
    while True:
        await asyncio.sleep(PENDING_SWEEP_INTERVAL)
        async with _pending_lock:
            stale = [sid for sid in _pending_messages if sid not in _orchestrator_sessions]
            for sid in stale:
                del _pending_messages[sid]
            if stale:
                logger.info("Swept %d stale pending queues", len(stale))


@app.on_event("startup")
async def _startup() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if SESSION_MAP_PATH.exists():
        saved = json.loads(SESSION_MAP_PATH.read_text())
        if saved.get("_version") == 2:
            _orchestrator_sessions.update(saved.get("orchestrator_sessions", {}))
        else:
            _orchestrator_sessions.update(saved)
    _prompts.update(_load_all_prompts())
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
    session_id: str,
    prompt: str,
    model: str | None = None,
    system_prompt: str | None = None,
    sdk_session_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Run an agent query and return (response_parts, result_info, new_sdk_session_id)."""
    resolved_model = model or MODEL
    resolved_prompt = system_prompt or _prompts["orchestrator"]

    options = ClaudeAgentOptions(
        system_prompt=resolved_prompt,
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

    return response_parts, result_info, new_sdk_session_id


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
        raw = part.get("tool") or ""
        tool_name = raw.replace("mcp_amortized__", "").replace("amortized_", "")
        if tool_name not in (
            "create_sdg_job",
            "create_training_job",
            "submit_recipe_job",
            "validate_sdg_job",
            "validate_training_job",
            "validate_recipe_job",
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
                    logger.warning("watch_job failed: %s %s", resp.status_code, resp.text[:200])
        except Exception:
            logger.warning("watch_job request failed for job %s", job_id, exc_info=True)


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


def _detect_delegation(response_parts: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Scan response for a delegate_to_subagent tool call. Returns (target, context) or None."""
    for part in response_parts:
        if part.get("type") != "tool" or part.get("state") != "completed":
            continue
        tool_name = (
            (part.get("tool") or "").replace("mcp_amortized__", "").replace("amortized_", "")
        )
        if tool_name == "delegate_to_subagent":
            inp = part.get("input", {})
            return inp.get("target", ""), inp.get("context", "")
    return None


def _detect_completion(response_parts: list[dict[str, Any]]) -> str | None:
    """Scan response for a signal_subagent_completion tool call. Returns summary or None."""
    for part in response_parts:
        if part.get("type") != "tool" or part.get("state") != "completed":
            continue
        tool_name = (
            (part.get("tool") or "").replace("mcp_amortized__", "").replace("amortized_", "")
        )
        if tool_name == "signal_subagent_completion":
            inp = part.get("input", {})
            return inp.get("summary", "")
    return None


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
        active_sub = _active_subagents.get(session_id)

        if active_sub:
            # Route to active subagent
            sub_prompt = _prompts.get(_subagent_targets.get(session_id, ""), "")
            response_parts, result_info, new_sdk_id = await _run_agent(
                session_id, user_text, model, system_prompt=sub_prompt, sdk_session_id=active_sub
            )
            if new_sdk_id:
                _active_subagents[session_id] = new_sdk_id

            # Check if subagent is signaling completion
            summary = _detect_completion(response_parts)
            if summary:
                _active_subagents.pop(session_id, None)
                _subagent_targets.pop(session_id, None)
                # Send summary to orchestrator and get its response
                orch_sdk_id = _orchestrator_sessions.get(session_id)
                resume_prompt = (
                    f"[SUBAGENT COMPLETED]\n{summary}\n\n"
                    "Present contextual next steps to the user via present_options."
                )
                orch_parts, orch_info, new_orch_id = await _run_agent(
                    session_id,
                    resume_prompt,
                    model,
                    sdk_session_id=orch_sdk_id,
                )
                if new_orch_id:
                    _orchestrator_sessions[session_id] = new_orch_id
                    await _persist_session_state()
                # Append orchestrator response to subagent's final response
                response_parts.extend(orch_parts)
                result_info = orch_info
        else:
            # Route to orchestrator
            orch_sdk_id = _orchestrator_sessions.get(session_id)
            response_parts, result_info, new_sdk_id = await _run_agent(
                session_id, user_text, model, sdk_session_id=orch_sdk_id
            )
            if new_sdk_id:
                _orchestrator_sessions[session_id] = new_sdk_id
                await _persist_session_state()

            # Check if orchestrator is delegating
            delegation = _detect_delegation(response_parts)
            if delegation:
                target, context = delegation
                if target in _prompts:
                    # Create fresh subagent session
                    sub_prompt = _prompts[target]
                    handoff_msg = f"[CONTEXT FROM ORCHESTRATOR]\n{context}"
                    sub_parts, sub_info, sub_sdk_id = await _run_agent(
                        session_id,
                        handoff_msg,
                        model,
                        system_prompt=sub_prompt,
                    )
                    if sub_sdk_id:
                        _active_subagents[session_id] = sub_sdk_id
                        _subagent_targets[session_id] = target
                    # Replace orchestrator response with subagent's first response
                    response_parts = sub_parts
                    result_info = sub_info
                else:
                    logger.warning("Unknown delegation target: %s", target)

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

    if session_id not in _orchestrator_sessions:
        raise HTTPException(status_code=404, detail="unknown session")

    now = time.monotonic()
    last = _last_event_time.get(session_id, 0.0)
    if now - last < EVENT_COOLDOWN_SECONDS:
        raise HTTPException(status_code=429, detail="event cooldown active")

    prompt = _build_event_prompt(body)

    lock = await _get_session_lock(session_id)
    async with lock:
        # Route event to active session (subagent or orchestrator)
        active_sub = _active_subagents.get(session_id)
        if active_sub:
            sub_prompt = _prompts.get(_subagent_targets.get(session_id, ""), "")
            response_parts, result_info, new_sdk_id = await _run_agent(
                session_id, prompt, system_prompt=sub_prompt, sdk_session_id=active_sub
            )
            if new_sdk_id:
                _active_subagents[session_id] = new_sdk_id
        else:
            orch_sdk_id = _orchestrator_sessions.get(session_id)
            response_parts, result_info, new_sdk_id = await _run_agent(
                session_id, prompt, sdk_session_id=orch_sdk_id
            )
            if new_sdk_id:
                _orchestrator_sessions[session_id] = new_sdk_id
                await _persist_session_state()

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
    if session_id not in _orchestrator_sessions:
        raise HTTPException(status_code=404, detail="unknown session")
    async with _pending_lock:
        messages = _pending_messages.pop(session_id, [])
    return {"messages": messages}


async def _persist_session_state() -> None:
    async with _map_lock:
        state = {
            "_version": 2,
            "orchestrator_sessions": dict(_orchestrator_sessions),
        }
        tmp = SESSION_MAP_PATH.with_suffix(".json.tmp")
        await asyncio.to_thread(tmp.write_text, json.dumps(state))
        await asyncio.to_thread(tmp.replace, SESSION_MAP_PATH)
