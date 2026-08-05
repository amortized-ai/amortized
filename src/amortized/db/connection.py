"""PostgreSQL database layer for job persistence.

Uses a connection pool so the API acquires/releases per-request
and the worker acquires/releases per-operation.
"""

import logging
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg

from amortized.config import settings

logger = logging.getLogger("amortized.db")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_pool: asyncpg.Pool | None = None  # type: ignore[type-arg]


def get_pool() -> asyncpg.Pool:  # type: ignore[type-arg]
    assert _pool is not None, "Database not initialized — call init_db() first"
    return _pool


async def get_db() -> AsyncIterator[asyncpg.Connection]:  # type: ignore[type-arg]
    pool = get_pool()
    async with pool.acquire() as conn:
        yield conn


async def init_db() -> None:
    global _pool
    _pool = await asyncpg.create_pool(dsn=settings.database_url, min_size=2, max_size=10)
    schema_sql = _SCHEMA_PATH.read_text()
    async with _pool.acquire() as conn:
        await conn.execute(schema_sql)
    logger.info("Database initialized at %s", settings.database_url.split("@")[-1])


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
