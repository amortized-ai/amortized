"""Async Python SDK client for the Amortized API."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

_TERMINAL_STATUSES = frozenset({"succeeded", "completed", "failed", "cancelled"})


def _discover_url() -> str:
    env = os.environ.get("AMORTIZED_API_URL")
    if env:
        return env.rstrip("/")

    config_path = Path.home() / ".amortized" / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            data = yaml.safe_load(config_path.read_text())
            if isinstance(data, dict) and data.get("api_url"):
                return str(data["api_url"]).rstrip("/")
        except Exception:
            pass

    return "http://localhost:8000"


class Job:
    """Wraps a job API response with convenience methods."""

    def __init__(self, data: dict[str, Any], client: Client) -> None:
        self._data = data
        self._client = client

    @property
    def id(self) -> str:
        return str(self._data["id"])

    @property
    def type(self) -> str:
        return str(self._data["type"])

    @property
    def status(self) -> str:
        return str(self._data["status"])

    @property
    def config(self) -> dict[str, Any]:
        result: dict[str, Any] = self._data.get("config", {})
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = self._data.get("metadata", {})
        return result

    @property
    def created_at(self) -> str:
        return str(self._data.get("created_at", ""))

    @property
    def error(self) -> str | None:
        return self._data.get("error")

    @property
    def artifacts(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._data.get("_artifacts", [])
        return result

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    async def refresh(self) -> Job:
        updated = await self._client.get_job(self.id)
        self._data = updated._data
        return self

    async def wait(self, poll_interval: float = 2.0) -> Job:
        while self.status not in _TERMINAL_STATUSES:
            await asyncio.sleep(poll_interval)
            await self.refresh()
        return self

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        url = f"/api/v1/jobs/{self.id}/events"
        async with self._client._http.stream("GET", url, params={"stream": "true"}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    import json

                    yield json.loads(line[5:].strip())

    async def cancel(self) -> Job:
        return await self._client.cancel_job(self.id)

    def artifact_ref(self, name: str) -> str:
        """Return an artifact reference string for use in job configs."""
        return f"artifact:{self.id}/{name}"

    def __repr__(self) -> str:
        return f"Job(id={self.id!r}, type={self.type!r}, status={self.status!r})"


class Client:
    """Async client for the Amortized API.

    Usage::

        async with Client() as c:
            job = await c.submit(type="training", config={...})
            await job.wait()
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or _discover_url()
        self._http = httpx.AsyncClient(base_url=self.base_url)

    async def submit(
        self,
        type: str,
        config: dict[str, Any],
        compute: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> Job:
        body: dict[str, Any] = {"type": type, "config": config, "dry_run": dry_run}
        if compute is not None:
            body["compute"] = compute
        if metadata is not None:
            body["metadata"] = metadata
        resp = await self._http.post("/api/v1/jobs", json=body)
        resp.raise_for_status()
        return Job(resp.json(), self)

    async def validate(
        self,
        type: str,
        config: dict[str, Any],
        compute: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate a job configuration without creating it."""
        body: dict[str, Any] = {"type": type, "config": config, "dry_run": True}
        if compute is not None:
            body["compute"] = compute
        resp = await self._http.post("/api/v1/jobs/validate", json=body)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def submit_recipe(
        self,
        recipe: str,
        overrides: dict[str, Any] | None = None,
    ) -> Job:
        body: dict[str, Any] = {"recipe": recipe}
        if overrides is not None:
            body["overrides"] = overrides
        resp = await self._http.post("/api/v1/jobs/recipe", json=body)
        resp.raise_for_status()
        return Job(resp.json(), self)

    async def get_job(self, job_id: str) -> Job:
        resp = await self._http.get(f"/api/v1/jobs/{job_id}")
        resp.raise_for_status()
        return Job(resp.json(), self)

    async def list_jobs(
        self,
        status: str | None = None,
        type: str | None = None,
    ) -> list[Job]:
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = status
        if type is not None:
            params["type"] = type
        resp = await self._http.get("/api/v1/jobs", params=params)
        resp.raise_for_status()
        return [Job(j, self) for j in resp.json()]

    async def cancel_job(self, job_id: str) -> Job:
        resp = await self._http.delete(f"/api/v1/jobs/{job_id}")
        resp.raise_for_status()
        return Job(resp.json(), self)

    async def list_job_types(self) -> list[dict[str, Any]]:
        resp = await self._http.get("/api/v1/job-types")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def get_job_type_schema(self, type: str) -> dict[str, Any]:
        resp = await self._http.get(f"/api/v1/job-types/{type}/schema")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def list_recipes(self) -> list[dict[str, Any]]:
        resp = await self._http.get("/api/v1/recipes")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def get_recipe(self, name: str) -> dict[str, Any]:
        resp = await self._http.get(f"/api/v1/recipes/{name}")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def upload(
        self,
        file_path: str,
        artifact_type: str = "dataset",
        name: str | None = None,
    ) -> dict[str, Any]:
        """Register a local file as an artifact."""
        p = Path(file_path)
        artifact_name = name or p.name
        body: dict[str, Any] = {
            "name": artifact_name,
            "artifact_type": artifact_type,
            "location": str(p.resolve()),
            "metadata": {"original_filename": p.name},
        }
        resp = await self._http.post("/api/v1/artifacts", json=body)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def list_artifacts(
        self,
        type: str | None = None,
        producer_job: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if type is not None:
            params["type"] = type
        if producer_job is not None:
            params["producer_job"] = producer_job
        resp = await self._http.get("/api/v1/artifacts", params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def list_backends(self) -> list[dict[str, Any]]:
        resp = await self._http.get("/api/v1/compute")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def health(self) -> dict[str, Any]:
        resp = await self._http.get("/api/v1/health")
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


class SyncClient:
    """Synchronous wrapper around the async :class:`Client`.

    Every method delegates to the async counterpart via ``asyncio.run()``.
    Useful for scripts and notebooks where an event loop is not already running.

    Usage::

        client = SyncClient()
        job = client.submit(type="training", config={...})
        job.wait()
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._async = Client(base_url)

    @property
    def base_url(self) -> str:
        return self._async.base_url

    def submit(
        self,
        type: str,
        config: dict[str, Any],
        compute: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> Job:
        return asyncio.run(
            self._async.submit(type, config, compute=compute, metadata=metadata, dry_run=dry_run)
        )

    def validate(
        self,
        type: str,
        config: dict[str, Any],
        compute: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(self._async.validate(type, config, compute=compute))

    def submit_recipe(
        self,
        recipe: str,
        overrides: dict[str, Any] | None = None,
    ) -> Job:
        return asyncio.run(self._async.submit_recipe(recipe, overrides=overrides))

    def get_job(self, job_id: str) -> Job:
        return asyncio.run(self._async.get_job(job_id))

    def list_jobs(
        self,
        status: str | None = None,
        type: str | None = None,
    ) -> list[Job]:
        return asyncio.run(self._async.list_jobs(status=status, type=type))

    def cancel_job(self, job_id: str) -> Job:
        return asyncio.run(self._async.cancel_job(job_id))

    def list_job_types(self) -> list[dict[str, Any]]:
        return asyncio.run(self._async.list_job_types())

    def get_job_type_schema(self, type: str) -> dict[str, Any]:
        return asyncio.run(self._async.get_job_type_schema(type))

    def list_recipes(self) -> list[dict[str, Any]]:
        return asyncio.run(self._async.list_recipes())

    def get_recipe(self, name: str) -> dict[str, Any]:
        return asyncio.run(self._async.get_recipe(name))

    def upload(
        self,
        file_path: str,
        artifact_type: str = "dataset",
        name: str | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(self._async.upload(file_path, artifact_type=artifact_type, name=name))

    def list_artifacts(
        self,
        type: str | None = None,
        producer_job: str | None = None,
    ) -> list[dict[str, Any]]:
        return asyncio.run(self._async.list_artifacts(type=type, producer_job=producer_job))

    def list_backends(self) -> list[dict[str, Any]]:
        return asyncio.run(self._async.list_backends())

    def health(self) -> dict[str, Any]:
        return asyncio.run(self._async.health())

    def close(self) -> None:
        asyncio.run(self._async.close())

    def __enter__(self) -> SyncClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
