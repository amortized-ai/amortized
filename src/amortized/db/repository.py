"""Repository wrapping all CRUD operations on the jobs table."""

import json
from typing import Any, ClassVar

import aiosqlite

from amortized.models import JobStatus, JobType


class Repository:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def create_job(
        self,
        *,
        job_id: str,
        job_type: JobType,
        config: dict[str, Any],
        created_at: str,
        recipe: str = "",
        parent_job_id: str = "",
        user_id: str = "",
    ) -> dict[str, Any]:
        await self.conn.execute(
            """INSERT INTO jobs
               (id, type, status, config, recipe, parent_job_id, user_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                job_type.value,
                JobStatus.queued.value,
                json.dumps(config),
                recipe,
                parent_job_id,
                user_id,
                created_at,
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

    _UPDATABLE_COLUMNS: ClassVar[set[str]] = {
        "status",
        "config",
        "recipe",
        "user_id",
        "k8s_job_name",
        "k8s_namespace",
        "mlflow_run_id",
        "mlflow_experiment",
        "parent_job_id",
        "error",
        "started_at",
        "completed_at",
        "backend_handle",
    }

    async def update_job(
        self,
        job_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if not kwargs:
            return await self.get_job(job_id)

        fields: list[str] = []
        params: list[Any] = []

        for key, value in kwargs.items():
            if key not in self._UPDATABLE_COLUMNS:
                raise ValueError(f"Cannot update column: {key!r}")
            fields.append(f"{key} = ?")
            params.append(value)

        params.append(job_id)
        await self.conn.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        await self.conn.commit()
        return await self.get_job(job_id)

    async def pick_pending_job(self) -> dict[str, Any] | None:
        cursor = await self.conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
            (JobStatus.queued.value,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_job(row)


    async def list_documents(self) -> list[dict[str, Any]]:
        cursor = await self.conn.execute(
            "SELECT document_id, mlflow_run_id, filename, format, created_at FROM documents ORDER BY created_at DESC"
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def create_document(
        self, *, document_id: str, filename: str, fmt: str, content: str, created_at: str,
        mlflow_run_id: str = "",
    ) -> dict[str, Any]:
        await self.conn.execute(
            "INSERT INTO documents (document_id, mlflow_run_id, filename, format, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, mlflow_run_id, filename, fmt, content, created_at),
        )
        await self.conn.commit()
        return {"document_id": document_id, "mlflow_run_id": mlflow_run_id, "filename": filename, "format": fmt, "content": content, "created_at": created_at}


def _row_to_job(row: Any) -> dict[str, Any]:
    d = dict(row)
    d["config"] = json.loads(d["config"]) if isinstance(d["config"], str) else d["config"]
    if d.get("error") in ("", "None"):
        d["error"] = None
    if d.get("started_at") == "":
        d["started_at"] = None
    if d.get("completed_at") == "":
        d["completed_at"] = None
    return d
