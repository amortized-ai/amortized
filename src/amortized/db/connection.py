"""PostgreSQL database layer for job persistence.

Uses a connection pool with retry logic and health checking.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import asyncpg

from amortized.config import settings

logger = logging.getLogger("amortized.db")

_pool: asyncpg.Pool | None = None

_MAX_RETRIES = 5
_BASE_DELAY = 1.0


def get_pool() -> asyncpg.Pool:
    assert _pool is not None, "Database not initialized — call init_db() first"
    return _pool


async def get_db() -> AsyncIterator[asyncpg.Connection]:
    pool = get_pool()
    async with pool.acquire() as conn:
        yield conn


async def init_db() -> None:
    global _pool
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=2,
                max_size=10,
                command_timeout=30.0,
                server_settings={"application_name": "amortized"},
            )
            logger.info("Database pool created (%s)", settings.database_url.split("@")[-1])
            return
        except (OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            delay = _BASE_DELAY * (2**attempt)
            logger.warning(
                "DB connect attempt %d/%d failed, retrying in %.0fs: %s",
                attempt + 1,
                _MAX_RETRIES,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    raise RuntimeError(
        f"Failed to connect to database after {_MAX_RETRIES} attempts"
    ) from last_error


async def check_db_health() -> bool:
    if _pool is None:
        return False
    try:
        async with _pool.acquire(timeout=2.0) as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
