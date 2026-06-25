"""Conversation history endpoints.

Chat is handled by OpenCode (external agent via MCP).
These endpoints manage conversation persistence for the Studio UI.
"""

import logging

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from amortized.db import Repository
from amortized.db import get_db as _get_db
from amortized.models import (
    Conversation,
    ConversationDetail,
    Message,
)

logger = logging.getLogger("amortized.conversations")

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.get("/conversations", response_model=list[Conversation])
async def get_conversations(
    db: aiosqlite.Connection = Depends(_get_db),
) -> list[Conversation]:
    """List all conversations."""
    rows = await Repository(db).list_conversations()
    return [Conversation(**row) for row in rows]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(
    conversation_id: str,
    db: aiosqlite.Connection = Depends(_get_db),
) -> ConversationDetail:
    """Get a conversation with its message history."""
    repo = Repository(db)
    conv = await repo.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = await repo.list_messages(conversation_id)
    messages = [Message(**m) for m in msgs]

    return ConversationDetail(
        id=conv["id"],
        title=conv["title"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=messages,
    )
