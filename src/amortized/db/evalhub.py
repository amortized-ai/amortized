"""Durable dispatch intent and last-observed Hub state, separate from request config."""

import json
from typing import Any

from amortized.db.repository import Repository


class EvalHubRepository:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.conn = repo.conn

    async def create(self, job_id: str, tenant: str, hub_url: str) -> None:
        await self.conn.execute(
            """INSERT INTO evalhub_executions (job_id, tenant, hub_url, correlation)
               VALUES ($1, $2, $3, $4) ON CONFLICT (job_id) DO NOTHING""",
            job_id,
            tenant,
            hub_url.rstrip("/"),
            f"amortized-{job_id}",
        )

    async def get(self, job_id: str) -> dict[str, Any] | None:
        row = await self.conn.fetchrow("SELECT * FROM evalhub_executions WHERE job_id = $1", job_id)
        if row is None:
            return None
        result = dict(row)
        for key in ("request", "snapshot", "inputs"):
            if isinstance(result[key], str):
                result[key] = json.loads(result[key])
        return result

    async def update(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "state",
            "remote_id",
            "request",
            "snapshot",
            "cancel_requested",
            "sync_error",
            "observed_at",
            "serving_handle",
            "secret_name",
            "inputs",
        }
        if not fields or not set(fields) <= allowed:
            raise ValueError("Invalid execution update")
        assignments = []
        values = []
        for i, (key, value) in enumerate(fields.items(), 1):
            suffix = "::jsonb" if key in {"request", "snapshot", "inputs"} else ""
            assignments.append(f"{key} = ${i}{suffix}")
            values.append(json.dumps(value) if suffix else value)
        values.append(job_id)
        await self.conn.execute(
            f"UPDATE evalhub_executions SET {', '.join(assignments)} WHERE job_id = ${len(values)}",
            *values,
        )

    async def cancel(self, job_id: str) -> None:
        await self.update(job_id, cancel_requested=True)
