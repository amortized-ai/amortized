"""Repository pattern wrapping all CRUD operations on the database."""

import contextlib
import json
from typing import Any

import aiosqlite

from amortized.core.crypto import decrypt_value, encrypt_value
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
                JobStatus.validating.value,
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
        mlflow_run_id: str | None = None,
        backend_handle: str | None = None,
        output_dir: str | None = None,
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
        if mlflow_run_id is not None:
            fields.append("mlflow_run_id = ?")
            params.append(mlflow_run_id)
        if backend_handle is not None:
            fields.append("backend_handle = ?")
            params.append(backend_handle)
        if output_dir is not None:
            fields.append("output_dir = ?")
            params.append(output_dir)

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
            ("queued",),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_job(row)

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
        cursor = await self.conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        row = await cursor.fetchone()
        assert row is not None
        return _row_to_artifact(row)

    async def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
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

    async def get_artifact_with_job_context(self, artifact_id: str) -> dict[str, Any] | None:
        """Return artifact path along with producing job's backend_handle and output_dir."""
        cursor = await self.conn.execute(
            "SELECT a.path, j.backend_handle, j.output_dir "
            "FROM artifacts a LEFT JOIN jobs j ON a.job_id = j.id "
            "WHERE a.id = ?",
            (artifact_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {"path": row[0], "backend_handle": row[1], "output_dir": row[2]}

    async def delete_artifact(self, artifact_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
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
        cursor = await self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
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
        cursor = await self.conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
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
        cursor = await self.conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
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

    # ---- Evaluators ----

    async def create_evaluator(
        self,
        *,
        evaluator_id: str,
        name: str,
        description: str,
        type: str,
        prompt: str,
        judgment_type: str,
        response_format: str,
        variables: list[str],
        model: str | None,
        inference_params: dict[str, Any],
        rule_config: dict[str, Any] | None,
        created_at: str,
    ) -> dict[str, Any]:
        await self.conn.execute(
            """INSERT INTO evaluators
               (id, name, description, type, prompt, judgment_type, response_format,
                variables, model, inference_params, rule_config, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evaluator_id,
                name,
                description,
                type,
                prompt,
                judgment_type,
                response_format,
                json.dumps(variables),
                model,
                json.dumps(inference_params),
                json.dumps(rule_config) if rule_config is not None else None,
                created_at,
                created_at,
            ),
        )
        await self.conn.commit()
        result = await self.get_evaluator(evaluator_id)
        assert result is not None
        return result

    async def get_evaluator(self, evaluator_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute("SELECT * FROM evaluators WHERE id = ?", (evaluator_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_evaluator(row)

    async def list_evaluators(self) -> list[dict[str, Any]]:
        cursor = await self.conn.execute("SELECT * FROM evaluators ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [_row_to_evaluator(row) for row in rows]

    async def update_evaluator(
        self, evaluator_id: str, *, updated_at: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        fields = ["updated_at = ?"]
        params: list[Any] = [updated_at]

        for key, value in kwargs.items():
            if key == "variables":
                fields.append("variables = ?")
                params.append(json.dumps(value))
            elif key == "inference_params":
                fields.append("inference_params = ?")
                params.append(json.dumps(value))
            elif key == "rule_config":
                fields.append("rule_config = ?")
                params.append(json.dumps(value) if value is not None else None)
            else:
                fields.append(f"{key} = ?")
                params.append(value)

        params.append(evaluator_id)
        await self.conn.execute(
            f"UPDATE evaluators SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        await self.conn.commit()
        return await self.get_evaluator(evaluator_id)

    async def delete_evaluator(self, evaluator_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM evaluators WHERE id = ?", (evaluator_id,))
        await self.conn.commit()
        return cursor.rowcount > 0

    # ---- Evaluations ----

    async def create_evaluation(
        self,
        *,
        evaluation_id: str,
        evaluator_id: str,
        dataset_artifact_id: str | None = None,
        job_id: str | None = None,
        created_at: str,
    ) -> dict[str, Any]:
        await self.conn.execute(
            """INSERT INTO evaluations
               (id, evaluator_id, dataset_artifact_id, job_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (evaluation_id, evaluator_id, dataset_artifact_id, job_id, created_at),
        )
        await self.conn.commit()
        result = await self.get_evaluation(evaluation_id)
        assert result is not None
        return result

    async def get_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_evaluation(row)

    async def list_evaluations(self, *, evaluator_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM evaluations"
        params: list[str] = []

        if evaluator_id is not None:
            query += " WHERE evaluator_id = ?"
            params.append(evaluator_id)

        query += " ORDER BY created_at DESC"
        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_evaluation(row) for row in rows]

    # ---- API Keys ----

    async def create_api_key(
        self,
        *,
        key_id: str,
        name: str,
        provider: str,
        key_value: str,
        created_at: str,
    ) -> dict[str, Any]:
        await self.conn.execute(
            "INSERT INTO api_keys (id, name, provider, key_value, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (key_id, name, provider, encrypt_value(key_value), created_at),
        )
        await self.conn.commit()
        result = await self.get_api_key(key_id)
        assert result is not None
        return result

    async def get_api_key(self, key_id: str) -> dict[str, Any] | None:
        cursor = await self.conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,))
        row = await cursor.fetchone()
        return _row_to_api_key(row) if row else None

    async def list_api_keys(self) -> list[dict[str, Any]]:
        cursor = await self.conn.execute("SELECT * FROM api_keys ORDER BY created_at")
        rows = await cursor.fetchall()
        return [_row_to_api_key(row) for row in rows]

    async def delete_api_key(self, key_id: str) -> bool:
        cursor = await self.conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        await self.conn.commit()
        return cursor.rowcount > 0

    async def get_api_key_for_provider(self, provider: str) -> dict[str, Any] | None:
        """Return the most recent API key for a provider.

        WARNING: Returns the full key_value for internal use (worker key injection).
        NEVER expose this in API responses — use list_api_keys() instead.
        """
        cursor = await self.conn.execute(
            "SELECT id, provider, key_value FROM api_keys"
            " WHERE provider = ? ORDER BY created_at DESC LIMIT 1",
            (provider,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["key_value"] = decrypt_value(d["key_value"])
        return d

    async def update_evaluation(
        self,
        evaluation_id: str,
        *,
        status: str | None = None,
        results: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        fields: list[str] = []
        params: list[Any] = []

        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if results is not None:
            fields.append("results = ?")
            params.append(json.dumps(results))

        if not fields:
            return await self.get_evaluation(evaluation_id)

        params.append(evaluation_id)
        await self.conn.execute(
            f"UPDATE evaluations SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        await self.conn.commit()
        return await self.get_evaluation(evaluation_id)


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
    if d.get("error") == "None":
        d["error"] = None
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


def _row_to_evaluator(row: Any) -> dict[str, Any]:
    d = dict(row)
    if isinstance(d.get("variables"), str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d["variables"] = json.loads(d["variables"])
    if isinstance(d.get("inference_params"), str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d["inference_params"] = json.loads(d["inference_params"])
    if isinstance(d.get("rule_config"), str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d["rule_config"] = json.loads(d["rule_config"])
    return d


def _row_to_api_key(row: Any) -> dict[str, Any]:
    d = dict(row)
    raw_key = d.pop("key_value", "")
    plaintext = decrypt_value(raw_key)
    d["key_preview"] = f"...{plaintext[-4:]}" if len(plaintext) >= 4 else "***"
    return d


def _row_to_evaluation(row: Any) -> dict[str, Any]:
    d = dict(row)
    if isinstance(d.get("results"), str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            d["results"] = json.loads(d["results"])
    if d.get("results") is None:
        d["results"] = {}
    return d
