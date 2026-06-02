"""Agent chat API endpoints."""

import json
import uuid
from datetime import UTC, datetime

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from amortized_runtime.agent import agent
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

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


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

    # Save user message
    await create_message(
        db,
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.user.value,
        content=request.message,
        created_at=now,
    )

    # Process with agent
    response = agent.process_message(request.message)

    # Save assistant message
    await create_message(
        db,
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.assistant.value,
        content=json.dumps(response.model_dump()),
        created_at=datetime.now(UTC).isoformat(),
    )

    return ChatResponse(
        conversation_id=conversation_id,
        message=response.message,
        suggested_action=response.suggested_action,
        context=response.context,
    )


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
