"""SQLite database layer for job persistence.

Uses a single shared connection so the API and worker always see the same
data — separate connections in WAL mode can have snapshot-visibility gaps.
"""

import logging
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite

from amortized.config import settings

logger = logging.getLogger("amortized.db")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_shared_db: aiosqlite.Connection | None = None


async def _get_shared_db() -> aiosqlite.Connection:
    global _shared_db
    if _shared_db is None:
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        _shared_db = await aiosqlite.connect(str(settings.db_path))
        _shared_db.row_factory = aiosqlite.Row
        logger.info("Opened shared DB connection to %s", settings.db_path)
    return _shared_db


async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    yield await _get_shared_db()


async def init_db() -> None:
    db = await _get_shared_db()
    schema_sql = _SCHEMA_PATH.read_text()
    await db.execute("PRAGMA journal_mode=WAL")
    await db.executescript(schema_sql)
    await db.commit()
    logger.info("Database initialized at %s", settings.db_path)


async def close_db() -> None:
    global _shared_db
    if _shared_db is not None:
        await _shared_db.close()
        _shared_db = None
