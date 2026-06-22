"""SQLite database layer for job and artifact persistence."""

import logging
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite

from amortized.config import settings

logger = logging.getLogger("amortized.db")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Get a database connection (FastAPI dependency)."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(settings.db_path))
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


_MIGRATIONS: list[tuple[str, str, str]] = [
    ("jobs", "backend_handle", "ALTER TABLE jobs ADD COLUMN backend_handle TEXT"),
    ("artifacts", "producer_job", "ALTER TABLE artifacts ADD COLUMN producer_job TEXT"),
    ("jobs", "mlflow_run_id", "ALTER TABLE jobs ADD COLUMN mlflow_run_id TEXT DEFAULT ''"),
]


async def _get_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    """Return the set of column names for *table*."""
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return {row[1] for row in rows}


async def _migrate(db: aiosqlite.Connection) -> None:
    """Add any missing columns to existing databases."""
    for table, column, ddl in _MIGRATIONS:
        cols = await _get_columns(db, table)
        if column not in cols:
            await db.execute(ddl)
            logger.info("Migration: added %s.%s", table, column)
    await db.commit()


async def init_db() -> None:
    """Create tables if they don't exist."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = _SCHEMA_PATH.read_text()
    async with aiosqlite.connect(str(settings.db_path)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.executescript(schema_sql)
        await _migrate(db)
        await db.commit()
    logger.info("Database initialized at %s", settings.db_path)
