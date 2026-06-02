"""SQLite database layer for job and artifact persistence."""

import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiosqlite

from amortized_runtime.config import settings
from amortized_runtime.models import JobStatus, JobType

logger = logging.getLogger("amortized_runtime.db")

_CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    config TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    pid INTEGER,
    output_dir TEXT
)
"""

_CREATE_ARTIFACTS_TABLE = """
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
)
"""

_CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
)
"""


async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Get a database connection (FastAPI dependency)."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(settings.db_path))
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


async def init_db() -> None:
    """Create tables if they don't exist."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(settings.db_path)) as db:
        await db.execute(_CREATE_JOBS_TABLE)
        await db.execute(_CREATE_ARTIFACTS_TABLE)
        await db.execute(_CREATE_CONVERSATIONS_TABLE)
        await db.execute(_CREATE_MESSAGES_TABLE)
        await db.commit()
    logger.info("Database initialized at %s", settings.db_path)


async def create_job(
    db: aiosqlite.Connection,
    *,
    job_id: str,
    job_type: JobType,
    config: dict[str, Any],
    created_at: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Insert a new job record."""
    await db.execute(
        """INSERT INTO jobs (id, type, status, config, created_at, updated_at, output_dir)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            job_id,
            job_type.value,
            JobStatus.pending.value,
            json.dumps(config),
            created_at,
            created_at,
            output_dir,
        ),
    )
    await db.commit()
    result = await get_job(db, job_id)
    assert result is not None
    return result


async def get_job(db: aiosqlite.Connection, job_id: str) -> dict[str, Any] | None:
    """Fetch a single job by ID."""
    cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_job(row)


async def list_jobs(
    db: aiosqlite.Connection,
    *,
    status: JobStatus | None = None,
    job_type: JobType | None = None,
) -> list[dict[str, Any]]:
    """List jobs with optional filters."""
    query = "SELECT * FROM jobs"
    params: list[str] = []
    conditions: list[str] = []

    if status is not None:
        conditions.append("status = ?")
        params.append(status.value)
    if job_type is not None:
        conditions.append("type = ?")
        params.append(job_type.value)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"

    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [_row_to_job(row) for row in rows]


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
    """Update job status and related fields."""
    fields = ["status = ?", "updated_at = ?"]
    params: list[Any] = [status.value, updated_at]

    if started_at is not None:
        fields.append("started_at = ?")
        params.append(started_at)
    if completed_at is not None:
        fields.append("completed_at = ?")
        params.append(completed_at)
    if error is not None:
        fields.append("error = ?")
        params.append(error)
    if pid is not None:
        fields.append("pid = ?")
        params.append(pid)

    params.append(job_id)
    await db.execute(
        f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    await db.commit()
    return await get_job(db, job_id)


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
    """Insert a new artifact record."""
    await db.execute(
        """INSERT INTO artifacts (id, job_id, artifact_type, path, size, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (artifact_id, job_id, artifact_type, path, size, created_at),
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
    row = await cursor.fetchone()
    assert row is not None
    return dict(row)


async def list_artifacts(
    db: aiosqlite.Connection, job_id: str
) -> list[dict[str, Any]]:
    """List artifacts for a given job."""
    cursor = await db.execute(
        "SELECT * FROM artifacts WHERE job_id = ? ORDER BY created_at", (job_id,)
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def create_conversation(
    db: aiosqlite.Connection,
    *,
    conversation_id: str,
    title: str,
    created_at: str,
) -> dict[str, Any]:
    """Insert a new conversation record."""
    await db.execute(
        """INSERT INTO conversations (id, title, created_at, updated_at)
           VALUES (?, ?, ?, ?)""",
        (conversation_id, title, created_at, created_at),
    )
    await db.commit()
    cursor = await db.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
    )
    row = await cursor.fetchone()
    assert row is not None
    return dict(row)


async def get_conversation(
    db: aiosqlite.Connection, conversation_id: str
) -> dict[str, Any] | None:
    """Fetch a single conversation by ID."""
    cursor = await db.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def list_conversations(
    db: aiosqlite.Connection,
) -> list[dict[str, Any]]:
    """List all conversations ordered by most recent."""
    cursor = await db.execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC"
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def update_conversation(
    db: aiosqlite.Connection,
    conversation_id: str,
    *,
    updated_at: str,
    title: str | None = None,
) -> dict[str, Any] | None:
    """Update conversation metadata."""
    fields = ["updated_at = ?"]
    params: list[Any] = [updated_at]
    if title is not None:
        fields.append("title = ?")
        params.append(title)
    params.append(conversation_id)
    await db.execute(
        f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?",
        params,
    )
    await db.commit()
    return await get_conversation(db, conversation_id)


async def create_message(
    db: aiosqlite.Connection,
    *,
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    created_at: str,
) -> dict[str, Any]:
    """Insert a new message record."""
    await db.execute(
        """INSERT INTO messages (id, conversation_id, role, content, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (message_id, conversation_id, role, content, created_at),
    )
    await db.commit()
    cursor = await db.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
    row = await cursor.fetchone()
    assert row is not None
    return _row_to_message(row)


async def list_messages(
    db: aiosqlite.Connection, conversation_id: str
) -> list[dict[str, Any]]:
    """List messages for a conversation ordered by creation time."""
    cursor = await db.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_message(row) for row in rows]


def _row_to_message(row: Any) -> dict[str, Any]:
    """Convert a database row to a message dict with parsed content."""
    d = dict(row)
    if isinstance(d["content"], str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d["content"] = json.loads(d["content"])
    return d


def _row_to_job(row: Any) -> dict[str, Any]:
    """Convert a database row to a job dict with parsed config."""
    d = dict(row)
    d["config"] = json.loads(d["config"]) if isinstance(d["config"], str) else d["config"]
    return d
