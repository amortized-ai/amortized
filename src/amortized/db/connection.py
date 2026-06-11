"""SQLite database layer for job and artifact persistence."""

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiosqlite

from amortized.config import settings
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType

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


# Backward-compatible thin wrappers — delegate to Repository.
# Existing call sites use these; new code should use Repository directly.


async def create_job(
    db: aiosqlite.Connection,
    *,
    job_id: str,
    job_type: JobType,
    config: dict[str, Any],
    created_at: str,
    output_dir: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await Repository(db).create_job(
        job_id=job_id,
        job_type=job_type,
        config=config,
        created_at=created_at,
        output_dir=output_dir,
        metadata=metadata,
    )


async def get_job(db: aiosqlite.Connection, job_id: str) -> dict[str, Any] | None:
    return await Repository(db).get_job(job_id)


async def list_jobs(
    db: aiosqlite.Connection,
    *,
    status: JobStatus | None = None,
    job_type: JobType | None = None,
) -> list[dict[str, Any]]:
    return await Repository(db).list_jobs(status=status, job_type=job_type)


async def update_job_status(
    db: aiosqlite.Connection,
    job_id: str,
    *,
    status: JobStatus,
    updated_at: str,
    started_at: str | None = None,
    completed_at: str | None = None,
    error: str | None = None,
    pid: int | None = None,
) -> dict[str, Any] | None:
    return await Repository(db).update_job_status(
        job_id,
        status=status,
        updated_at=updated_at,
        started_at=started_at,
        completed_at=completed_at,
        error=error,
        pid=pid,
    )


async def create_artifact(
    db: aiosqlite.Connection,
    *,
    artifact_id: str,
    job_id: str,
    artifact_type: str,
    path: str,
    size: int,
    created_at: str,
) -> dict[str, Any]:
    return await Repository(db).create_artifact(
        artifact_id=artifact_id,
        job_id=job_id,
        artifact_type=artifact_type,
        path=path,
        size=size,
        created_at=created_at,
    )


async def get_artifact(db: aiosqlite.Connection, artifact_id: str) -> dict[str, Any] | None:
    return await Repository(db).get_artifact(artifact_id)


async def list_artifacts(db: aiosqlite.Connection, job_id: str) -> list[dict[str, Any]]:
    return await Repository(db).list_artifacts(job_id)


async def create_conversation(
    db: aiosqlite.Connection,
    *,
    conversation_id: str,
    title: str,
    created_at: str,
) -> dict[str, Any]:
    return await Repository(db).create_conversation(
        conversation_id=conversation_id,
        title=title,
        created_at=created_at,
    )


async def get_conversation(db: aiosqlite.Connection, conversation_id: str) -> dict[str, Any] | None:
    return await Repository(db).get_conversation(conversation_id)


async def list_conversations(
    db: aiosqlite.Connection,
) -> list[dict[str, Any]]:
    return await Repository(db).list_conversations()


async def update_conversation(
    db: aiosqlite.Connection,
    conversation_id: str,
    *,
    updated_at: str,
    title: str | None = None,
) -> dict[str, Any] | None:
    return await Repository(db).update_conversation(
        conversation_id,
        updated_at=updated_at,
        title=title,
    )


async def create_message(
    db: aiosqlite.Connection,
    *,
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    created_at: str,
) -> dict[str, Any]:
    return await Repository(db).create_message(
        message_id=message_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=created_at,
    )


async def list_messages(db: aiosqlite.Connection, conversation_id: str) -> list[dict[str, Any]]:
    return await Repository(db).list_messages(conversation_id)
