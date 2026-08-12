"""Experiment listing endpoints."""

import asyncpg
from fastapi import APIRouter, Depends

from amortized.core.lineage import list_lineage_chains
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import LineageChainSummary

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.get("", response_model=list[LineageChainSummary], operation_id="list_lineage_chains")
async def list_chains(
    type: str = "",
    status: str = "",
    db: asyncpg.Connection = Depends(_get_db),
) -> list[LineageChainSummary]:
    repo = Repository(db)
    return await list_lineage_chains(repo, job_type=type, status=status)
