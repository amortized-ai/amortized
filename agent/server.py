"""OpenCode-compatible wrapper around the Claude Agent SDK.

Exposes the same HTTP API that OpenCode's `serve` command provides,
so Amortized Studio can talk to either backend without code changes.

Endpoints:
  POST /session              → create a session
  POST /session/{id}/message → send a message, get a synchronous JSON response
  GET  /api/health           → health check
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MORTY_PROMPT_PATH = Path(os.environ.get("MORTY_PROMPT_PATH", "/app/morty.md"))
CONFIG_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR", "/data"))
SESSION_MAP_PATH = CONFIG_DIR / "session_map.json"

MCP_AMORTIZED_URL = os.environ.get(
    "MCP_AMORTIZED_URL",
    "http://amortized-server.amortized.svc.cluster.local:8000/mcp",
)
MCP_MLFLOW_URL = os.environ.get("MCP_MLFLOW_URL", "http://127.0.0.1:5002/sse")

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")

_session_map: dict[str, str] = {}
_map_lock = asyncio.Lock()
_morty_prompt: str = ""


def _strip_frontmatter(text: str) -> str:
    """Strip YAML frontmatter (between --- markers) from a markdown file."""
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end == -1:
        return text
    return text[end + 3 :].lstrip("\n")


def _load_morty_prompt() -> str:
    if not MORTY_PROMPT_PATH.exists():
        raise RuntimeError(f"morty.md not found at {MORTY_PROMPT_PATH}")
    raw = MORTY_PROMPT_PATH.read_text()
    return _strip_frontmatter(raw)


app = FastAPI(title="Claude Code Agent (OpenCode-compatible)")


@app.on_event("startup")
async def _startup() -> None:
    global _morty_prompt
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if SESSION_MAP_PATH.exists():
        _session_map.update(json.loads(SESSION_MAP_PATH.read_text()))
    _morty_prompt = _load_morty_prompt()


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


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"healthy": True, "version": "0.1.0"}


@app.post("/session")
async def create_session() -> SessionResponse:
    session_id = str(uuid.uuid4())
    return SessionResponse(id=session_id)


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
        model = body.model.modelID.replace("@default", "")

    sdk_session_id = _session_map.get(session_id)

    options = ClaudeAgentOptions(
        system_prompt=_morty_prompt,
        allowed_tools=["mcp__*"],
        permission_mode="acceptEdits",
        setting_sources=[],
        model=model,
        cwd="/app/workspace",
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

    async for message in query(prompt=user_text, options=options):
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
                "providerID": "google-vertex-anthropic",
                "modelID": model,
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
            "providerID": "google-vertex-anthropic",
            "modelID": model,
            "cost": 0,
            "tokens": {"input": 0, "output": 0, "reasoning": 0},
            "finish": "stop",
            "id": str(uuid.uuid4()),
            "sessionID": session_id,
        }

    return {"info": result_info, "parts": response_parts}


async def _persist_session(external_id: str, sdk_session_id: str) -> None:
    async with _map_lock:
        if _session_map.get(external_id) == sdk_session_id:
            return
        _session_map[external_id] = sdk_session_id
        tmp = SESSION_MAP_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_session_map))
        tmp.replace(SESSION_MAP_PATH)
