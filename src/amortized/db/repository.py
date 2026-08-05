"""Repository wrapping all CRUD operations on the jobs table."""

import json
from typing import Any, ClassVar

import asyncpg

from amortized.models import JobStatus, JobType


class Repository:
    def __init__(self, conn: asyncpg.Connection) -> None:  # type: ignore[type-arg]
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
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            job_id,
            job_type.value,
            JobStatus.queued.value,
            json.dumps(config),
            recipe,
            parent_job_id,
            user_id,
            created_at,
        )
        result = await self.get_job(job_id)
        assert result is not None
        return result

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = await self.conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
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
        params: list[Any] = []
        conditions: list[str] = []
        idx = 1

        if status is not None:
            conditions.append(f"status = ${idx}")
            params.append(status.value)
            idx += 1
        if job_type is not None:
            conditions.append(f"type = ${idx}")
            params.append(job_type.value)
            idx += 1

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        rows = await self.conn.fetch(query, *params)
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

        for i, (key, value) in enumerate(kwargs.items(), 1):
            if key not in self._UPDATABLE_COLUMNS:
                raise ValueError(f"Cannot update column: {key!r}")
            fields.append(f"{key} = ${i}")
            params.append(value)

        params.append(job_id)
        await self.conn.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE id = ${len(params)}",
            *params,
        )
        return await self.get_job(job_id)

    async def pick_pending_job(self) -> dict[str, Any] | None:
        row = await self.conn.fetchrow(
            "SELECT * FROM jobs WHERE status = $1 ORDER BY created_at ASC LIMIT 1",
            JobStatus.queued.value,
        )
        if row is None:
            return None
        return _row_to_job(row)

    async def delete_job(self, job_id: str) -> bool:
        result = await self.conn.execute("DELETE FROM jobs WHERE id = $1", job_id)
        return result == "DELETE 1"


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
