"""Agent chat API endpoints with OpenAI function-calling backend."""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from amortized_runtime.agent import AgentResult, process_message, stream_message
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
    SuggestedAction,
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


async def _ensure_conversation(
    db: aiosqlite.Connection,
    conversation_id: str | None,
    title: str,
) -> str:
    """Create or update a conversation, returning the conversation ID."""
    now = datetime.now(UTC).isoformat()
    if conversation_id:
        conv = await get_conversation(db, conversation_id)
        if conv is None:
            await create_conversation(
                db, conversation_id=conversation_id, title=title, created_at=now
            )
        else:
            await update_conversation(db, conversation_id, updated_at=now)
        return conversation_id

    new_id = str(uuid.uuid4())
    await create_conversation(db, conversation_id=new_id, title=title, created_at=now)
    return new_id


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> ChatResponse:
    """Send a message to the agent and get a response."""
    conversation_id = await _ensure_conversation(
        db, request.conversation_id, request.message[:50]
    )

    existing_msgs = await list_messages(db, conversation_id)
    history = _history_from_messages(existing_msgs)

    await create_message(
        db,
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.user.value,
        content=request.message,
        created_at=datetime.now(UTC).isoformat(),
    )

    result: AgentResult = await process_message(request.message, history=history)

    suggested = None
    if result.proposed_action:
        suggested = SuggestedAction(
            type=result.proposed_action["type"],
            config=result.proposed_action["config"],
            label=result.proposed_action["label"],
        )

    await create_message(
        db,
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.assistant.value,
        content=result.text,
        created_at=datetime.now(UTC).isoformat(),
    )

    return ChatResponse(
        conversation_id=conversation_id,
        message=result.text,
        suggested_action=suggested,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: aiosqlite.Connection = Depends(_get_db),
) -> EventSourceResponse:
    """Send a message to the agent and stream the response via SSE.

    Agent execution is decoupled from SSE streaming: the full agent loop runs
    first, the response is persisted to the DB, and only then are the collected
    events replayed to the client.  This guarantees the assistant response is
    always saved regardless of whether the client disconnects mid-stream.
    """
    conversation_id = await _ensure_conversation(
        db, request.conversation_id, request.message[:50]
    )

    existing_msgs = await list_messages(db, conversation_id)
    history = _history_from_messages(existing_msgs)

    await create_message(
        db,
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.user.value,
        content=request.message,
        created_at=datetime.now(UTC).isoformat(),
    )

    # 1. Run the full agent loop and collect all events
    events: list[dict[str, object]] = []
    full_text = ""

    try:
        async for event in stream_message(request.message, history=history):
            events.append({"type": event.type, "data": event.data})
            if event.type == "delta":
                full_text += event.data.get("text", "")
            elif event.type == "done":
                full_text = event.data.get("full_text", full_text)
    except Exception:
        logger.exception("Agent execution error")
        error_msg = "Sorry, something went wrong. Please try again."
        events.append({"type": "error", "data": {"error": error_msg}})
        full_text = full_text or error_msg

    # 2. Save assistant response BEFORE streaming to client
    if full_text:
        await create_message(
            db,
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.assistant.value,
            content=full_text,
            created_at=datetime.now(UTC).isoformat(),
        )

    # 3. Replay collected events to the client via SSE
    async def replay_events() -> AsyncIterator[dict[str, str]]:
        yield {"event": "metadata", "data": json.dumps({"conversation_id": conversation_id})}
        for evt in events:
            yield {"event": str(evt["type"]), "data": json.dumps(evt["data"])}

    return EventSourceResponse(replay_events())


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
