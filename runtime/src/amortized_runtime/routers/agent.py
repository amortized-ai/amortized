"""Agent chat API endpoints."""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from amortized_runtime.agent import process_message, stream_message
from amortized_runtime.db import (
    create_conversation,
    create_message,
    get_conversation,
    list_conversations,
    list_messages,
    update_conversation,
)
from amortized_runtime.db import get_db as _get_db
from amortized_runtime.models import (
    ChatRequest,
    ChatResponse,
    Conversation,
    ConversationDetail,
    Message,
    MessageRole,
)

logger = logging.getLogger("amortized_runtime.agent_router")

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


def _history_from_messages(msgs: list[dict[str, object]]) -> list[dict[str, str]]:
    """Convert stored messages into the history format expected by the agent."""
    history: list[dict[str, str]] = []
    for m in msgs:
        role = str(m["role"])
        content = m["content"]
        text = str(content.get("message", content)) if isinstance(content, dict) else str(content)
        history.append({"role": role, "content": text})
    return history


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> ChatResponse:
    """Send a message to the agent and get a response."""
    now = datetime.now(UTC).isoformat()

    # Create or fetch conversation
    if request.conversation_id:
        conv = await get_conversation(db, request.conversation_id)
        if conv is None:
            conv = await create_conversation(
                db,
                conversation_id=request.conversation_id,
                title=request.message[:50],
                created_at=now,
            )
        else:
            await update_conversation(
                db, request.conversation_id, updated_at=now
            )
        conversation_id = request.conversation_id
    else:
        conversation_id = str(uuid.uuid4())
        await create_conversation(
            db,
            conversation_id=conversation_id,
            title=request.message[:50],
            created_at=now,
        )

    # Load conversation history for multi-turn
    existing_msgs = await list_messages(db, conversation_id)
    history = _history_from_messages(existing_msgs)

    # Save user message
    await create_message(
        db,
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.user.value,
        content=request.message,
        created_at=now,
    )

    # Process with Claude Code CLI agent
    response_text = await process_message(request.message, history=history)

    # Save assistant message
    await create_message(
        db,
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.assistant.value,
        content=response_text,
        created_at=datetime.now(UTC).isoformat(),
    )

    return ChatResponse(
        conversation_id=conversation_id,
        message=response_text,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> EventSourceResponse:
    """Send a message to the agent and stream the response via SSE."""
    now = datetime.now(UTC).isoformat()

    # Create or fetch conversation
    if request.conversation_id:
        conv = await get_conversation(db, request.conversation_id)
        if conv is None:
            await create_conversation(
                db,
                conversation_id=request.conversation_id,
                title=request.message[:50],
                created_at=now,
            )
        else:
            await update_conversation(
                db, request.conversation_id, updated_at=now
            )
        conversation_id = request.conversation_id
    else:
        conversation_id = str(uuid.uuid4())
        await create_conversation(
            db,
            conversation_id=conversation_id,
            title=request.message[:50],
            created_at=now,
        )

    # Load conversation history
    existing_msgs = await list_messages(db, conversation_id)
    history = _history_from_messages(existing_msgs)

    # Save user message
    await create_message(
        db,
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.user.value,
        content=request.message,
        created_at=now,
    )

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        # Send conversation_id first
        yield {"event": "metadata", "data": json.dumps({"conversation_id": conversation_id})}

        full_text = ""
        got_done = False
        try:
            proc = await stream_message(request.message, history=history)
            assert proc.stdout is not None

            async for raw_line in proc.stdout:
                line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "assistant":
                    content = event.get("message", {}).get("content", [])
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block["text"]
                            full_text += text
                            yield {
                                "event": "delta",
                                "data": json.dumps({"text": text}),
                            }
                elif event.get("type") == "result":
                    result_text = event.get("result", "")
                    if result_text:
                        full_text = result_text
                    got_done = True
                    yield {
                        "event": "done",
                        "data": json.dumps({"full_text": full_text or result_text}),
                    }

            await proc.wait()

            # If we never got a result event, send done with what we have
            if not got_done:
                yield {
                    "event": "done",
                    "data": json.dumps({"full_text": full_text}),
                }

        except FileNotFoundError:
            error_msg = "The Claude CLI is not installed or not in PATH."
            yield {"event": "error", "data": json.dumps({"error": error_msg})}
            full_text = full_text or error_msg
        except Exception:
            logger.exception("Streaming error")
            error_msg = "Sorry, something went wrong. Please try again."
            yield {"event": "error", "data": json.dumps({"error": error_msg})}
            full_text = full_text or error_msg

        # Save assistant response
        await create_message(
            db,
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.assistant.value,
            content=full_text,
            created_at=datetime.now(UTC).isoformat(),
        )

    return EventSourceResponse(event_generator())


@router.get("/conversations", response_model=list[Conversation])
async def get_conversations(
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Conversation]:
    """List all conversations."""
    rows = await list_conversations(db)
    return [Conversation(**row) for row in rows]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(
    conversation_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> ConversationDetail:
    """Get a conversation with its message history."""
    conv = await get_conversation(db, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = await list_messages(db, conversation_id)
    messages = [Message(**m) for m in msgs]

    return ConversationDetail(
        id=conv["id"],
        title=conv["title"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=messages,
    )
