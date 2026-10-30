"""Prompt store read endpoints (#541).

Prompts are written by show_prompt (agent path) and by create_sdg_job's refify
(recipe path); these endpoints let the frontend resolve a config's `<prompt:...>`
refs back to text for display and for the editable recipe form. Resolution is by
id alone (a capability token — the id is unguessable, like a document/model ref),
so a job's prompts render regardless of who owns the row; listing is user-scoped.
"""

from __future__ import annotations

from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from amortized.db import get_db as _get_db
from amortized.db.repository import Repository

router = APIRouter(prefix="/api/v1/prompts", tags=["prompts"])


class PromptRecord(BaseModel):
    id: str
    label: str = ""
    purpose: str = ""
    role: str = ""
    content: str


class ResolvePromptsRequest(BaseModel):
    prompt_ids: list[str] = Field(default_factory=list)


class ResolvePromptsResponse(BaseModel):
    # id -> text, for the ids that resolved (missing ids are omitted).
    prompts: dict[str, str] = Field(default_factory=dict)


@router.post(
    "/resolve",
    response_model=ResolvePromptsResponse,
    operation_id="resolve_prompts",
    summary="Resolve prompt reference ids to their stored text (for rendering ref-form configs).",
)
async def resolve_prompts(
    body: ResolvePromptsRequest,
    db: asyncpg.Connection = Depends(_get_db),
) -> ResolvePromptsResponse:
    repo = Repository(db)
    resolved: dict[str, str] = {}
    for prompt_id in body.prompt_ids:
        text = await repo.get_prompt_content(prompt_id)
        if text is not None:
            resolved[prompt_id] = text
    return ResolvePromptsResponse(prompts=resolved)


@router.get(
    "",
    response_model=list[PromptRecord],
    operation_id="list_prompts",
    summary="List the prompts owned by the current user.",
)
async def list_prompts(
    http_request: Request,
    db: asyncpg.Connection = Depends(_get_db),
) -> list[PromptRecord]:
    user_id = http_request.headers.get("X-Forwarded-User", "")
    repo = Repository(db)
    rows = await repo.list_prompts(user_id)
    return [PromptRecord(**_record(row)) for row in rows]


@router.get(
    "/{prompt_id}",
    response_model=PromptRecord,
    operation_id="get_prompt",
    summary="Fetch one stored prompt by id.",
)
async def get_prompt(
    prompt_id: str,
    db: asyncpg.Connection = Depends(_get_db),
) -> PromptRecord:
    repo = Repository(db)
    row = await repo.get_prompt(prompt_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"prompt {prompt_id} not found")
    return PromptRecord(**_record(row))


def _record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "label": row.get("label") or "",
        "purpose": row.get("purpose") or "",
        "role": row.get("role") or "",
        "content": row.get("content") or "",
    }
