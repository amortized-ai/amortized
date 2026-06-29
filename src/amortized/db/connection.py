"""SQLite database layer for job persistence."""

import logging
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite

from amortized.config import settings

logger = logging.getLogger("amortized.db")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(settings.db_path))
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


async def init_db() -> None:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = _SCHEMA_PATH.read_text()
    async with aiosqlite.connect(str(settings.db_path)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executescript(schema_sql)
        await db.commit()
    logger.info("Database initialized at %s", settings.db_path)
