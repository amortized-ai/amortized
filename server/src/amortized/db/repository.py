"""Repository pattern wrapping all CRUD operations on the database."""

import contextlib
import json
from typing import Any

import aiosqlite

from amortized.models import JobStatus, JobType


class Repository:
    """Wraps an aiosqlite connection with typed CRUD methods."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    # ---- Jobs ----

    async def create_job(
        self,
        *,
        job_id: str,
        job_type: JobType,
        config: dict[str, Any],
        created_at: str,
        output_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.conn.execute(
            """INSERT INTO jobs
               (id, type, status, config, metadata, created_at, updated_at, output_dir)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                job_type.value,
                JobStatus.pending.value,
                json.dumps(config),
                json.dumps(metadata or {}),
                created_at,
                created_at,
                output_dir,
            ),
        )
        await self.conn.commit()
        result = await self.get_job(job_id)
        assert result is not None
        return result

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_job(row)

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
    ) -> list[dict[str, Any]]:
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

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_job(row) for row in rows]

    async def update_job_status(
        self,
        job_id: str,
        *,
        status: JobStatus,
        updated_at: str,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: str | None = None,
        pid: int | None = None,
    ) -> dict[str, Any] | None:
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
        await self.conn.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        await self.conn.commit()
        return await self.get_job(job_id)

    # ---- Artifacts ----

    async def create_artifact(
        self,
        *,
        artifact_id: str,
        job_id: str | None = None,
        artifact_type: str,
        path: str = "",
        size: int = 0,
        created_at: str,
        name: str = "",
        location: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.conn.execute(
            """INSERT INTO artifacts
               (id, job_id, artifact_type, path, size, created_at, name, location, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                job_id,
                artifact_type,
                path,
                size,
                created_at,
                name,
                location,
                json.dumps(metadata or {}),
            ),
        )
        await self.conn.commit()
        cursor = await self.conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        )
        row = await cursor.fetchone()
        assert row is not None
        return _row_to_artifact(row)

    async def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        )
        row = await cursor.fetchone()
        return _row_to_artifact(row) if row else None

    async def list_artifacts(
        self,
        job_id: str | None = None,
        *,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM artifacts"
        params: list[str] = []
        conditions: list[str] = []

        if job_id is not None:
            conditions.append("job_id = ?")
            params.append(job_id)
        if artifact_type is not None:
            conditions.append("artifact_type = ?")
            params.append(artifact_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at"

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_artifact(row) for row in rows]

    async def delete_artifact(self, artifact_id: str) -> bool:
        cursor = await self.conn.execute(
            "DELETE FROM artifacts WHERE id = ?", (artifact_id,)
        )
        await self.conn.commit()
        return cursor.rowcount > 0

    # ---- Events ----

    async def create_event(
        self,
        *,
        event_id: str,
        job_id: str,
        event_type: str,
        timestamp: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.conn.execute(
            """INSERT INTO events (id, job_id, type, timestamp, data)
               VALUES (?, ?, ?, ?, ?)""",
            (event_id, job_id, event_type, timestamp, json.dumps(data) if data else None),
        )
        await self.conn.commit()
        cursor = await self.conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        )
        row = await cursor.fetchone()
        assert row is not None
        return _row_to_event(row)

    async def get_latest_event(self, job_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute(
            "SELECT * FROM events WHERE job_id = ? ORDER BY timestamp DESC LIMIT 1",
            (job_id,),
        )
        row = await cursor.fetchone()
        return _row_to_event(row) if row else None

    async def list_events(
        self,
        job_id: str,
        *,
        since: str | None = None,
        types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM events WHERE job_id = ?"
        params: list[Any] = [job_id]

        if since is not None:
            query += " AND timestamp > ?"
            params.append(since)
        if types:
            placeholders = ", ".join("?" for _ in types)
            query += f" AND type IN ({placeholders})"
            params.extend(types)

        query += " ORDER BY timestamp"
        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_event(row) for row in rows]

    # ---- Conversations ----

    async def create_conversation(
        self,
        *,
        conversation_id: str,
        title: str,
        created_at: str,
    ) -> dict[str, Any]:
        await self.conn.execute(
            """INSERT INTO conversations (id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, title, created_at, created_at),
        )
        await self.conn.commit()
        cursor = await self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        )
        row = await cursor.fetchone()
        assert row is not None
        return dict(row)

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_conversations(self) -> list[dict[str, Any]]:
        cursor = await self.conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        updated_at: str,
        title: str | None = None,
    ) -> dict[str, Any] | None:
        fields = ["updated_at = ?"]
        params: list[Any] = [updated_at]
        if title is not None:
            fields.append("title = ?")
            params.append(title)
        params.append(conversation_id)
        await self.conn.execute(
            f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        await self.conn.commit()
        return await self.get_conversation(conversation_id)

    # ---- Messages ----

    async def create_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        role: str,
        content: str,
        created_at: str,
    ) -> dict[str, Any]:
        await self.conn.execute(
            """INSERT INTO messages (id, conversation_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (message_id, conversation_id, role, content, created_at),
        )
        await self.conn.commit()
        cursor = await self.conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        )
        row = await cursor.fetchone()
        assert row is not None
        return _row_to_message(row)

    async def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        cursor = await self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_message(row) for row in rows]


def _row_to_message(row: Any) -> dict[str, Any]:
    d = dict(row)
    if isinstance(d["content"], str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d["content"] = json.loads(d["content"])
    return d


def _row_to_job(row: Any) -> dict[str, Any]:
    d = dict(row)
    d["config"] = json.loads(d["config"]) if isinstance(d["config"], str) else d["config"]
    raw_meta = d.get("metadata")
    if isinstance(raw_meta, str):
        d["metadata"] = json.loads(raw_meta) if raw_meta else {}
    elif raw_meta is None:
        d["metadata"] = {}
    return d


def _row_to_artifact(row: Any) -> dict[str, Any]:
    d = dict(row)
    if isinstance(d.get("metadata"), str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d["metadata"] = json.loads(d["metadata"])
    if d.get("metadata") is None:
        d["metadata"] = {}
    d["producer_job"] = d.get("job_id")
    return d


def _row_to_event(row: Any) -> dict[str, Any]:
    d = dict(row)
    if isinstance(d.get("data"), str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d["data"] = json.loads(d["data"])
    return d
