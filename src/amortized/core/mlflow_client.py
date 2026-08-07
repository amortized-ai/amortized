"""Async MLflow REST API client — shared primitives for experiment/run/artifact lifecycle."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger("amortized.core.mlflow_client")


class MLflowClient:
    """Thin async wrapper around the MLflow REST API."""

    def __init__(self, tracking_uri: str, timeout: float = 30.0) -> None:
        self._base = tracking_uri.rstrip("/")
        self._timeout = timeout
        self._artifact_prefix_cache: dict[str, str] = {}

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    async def list_experiment_ids(self, max_results: int = 200) -> list[str]:
        """Return all experiment IDs."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("/api/2.0/mlflow/experiments/search"),
                json={"max_results": max_results},
            )
            resp.raise_for_status()
            return [e["experiment_id"] for e in resp.json().get("experiments", [])]

    async def get_experiment(self, name: str) -> str | None:
        """Get an experiment ID by name. Returns None if it doesn't exist."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url("/api/2.0/mlflow/experiments/get-by-name"),
                params={"experiment_name": name},
            )
            if resp.status_code == 404 or "RESOURCE_DOES_NOT_EXIST" in resp.text:
                return None
            resp.raise_for_status()
            exp = resp.json()["experiment"]
            if exp.get("lifecycle_stage") == "deleted":
                return None
            experiment_id: str = exp["experiment_id"]
            return experiment_id

    async def ensure_experiment(self, name: str) -> str:
        """Get or create an experiment by name. Returns the experiment ID."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url("/api/2.0/mlflow/experiments/get-by-name"),
                params={"experiment_name": name},
            )
            if resp.status_code == 404 or "RESOURCE_DOES_NOT_EXIST" in resp.text:
                create_resp = await client.post(
                    self._url("/api/2.0/mlflow/experiments/create"),
                    json={"name": name},
                )
                if create_resp.status_code == 409:
                    refetch = await client.get(
                        self._url("/api/2.0/mlflow/experiments/get-by-name"),
                        params={"experiment_name": name},
                    )
                    refetch.raise_for_status()
                    experiment_id: str = refetch.json()["experiment"]["experiment_id"]
                    return experiment_id
                create_resp.raise_for_status()
                experiment_id = create_resp.json()["experiment_id"]
                return experiment_id

            resp.raise_for_status()
            exp = resp.json()["experiment"]
            experiment_id = exp["experiment_id"]
            if exp.get("lifecycle_stage") == "deleted":
                restore_resp = await client.post(
                    self._url("/api/2.0/mlflow/experiments/restore"),
                    json={"experiment_id": experiment_id},
                )
                restore_resp.raise_for_status()
                logger.info("Restored deleted experiment %s", name)
            return experiment_id

    async def _resolve_artifact_prefix(self, run_id: str) -> str:
        """Resolve the artifact path prefix for the mlflow-artifacts proxy.

        The proxy serves from the artifact root (--default-artifact-root).
        Each run's artifact_uri includes experiment_id/run_id/artifacts under
        that root.  We strip the root to get the relative prefix the proxy needs.
        """
        if run_id in self._artifact_prefix_cache:
            return self._artifact_prefix_cache[run_id]
        try:
            run = await self.get_run(run_id)
        except httpx.HTTPStatusError:
            logger.error("Failed to resolve artifact prefix for run %s", run_id)
            raise
        artifact_uri = run["info"]["artifact_uri"]
        prefix = self._extract_artifact_prefix(run_id, artifact_uri)
        self._artifact_prefix_cache[run_id] = prefix
        return prefix

    @staticmethod
    def _extract_artifact_prefix(run_id: str, artifact_uri: str) -> str:
        suffix = f"/{run_id}/artifacts"
        if not artifact_uri.endswith(suffix):
            raise ValueError(
                f"Cannot extract artifact prefix: artifact_uri '{artifact_uri}' "
                f"does not end with expected suffix '{suffix}'"
            )
        experiment_location = artifact_uri[: -len(suffix)]
        artifact_root = experiment_location.rsplit("/", 1)[0] + "/"
        return artifact_uri[len(artifact_root) :]

    async def create_run(
        self,
        experiment_id: str,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Create a run in an experiment. Returns the run ID."""
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        tag_list = [{"key": k, "value": v} for k, v in (tags or {}).items()]
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("/api/2.0/mlflow/runs/create"),
                json={
                    "experiment_id": experiment_id,
                    "run_name": name,
                    "start_time": now_ms,
                    "tags": tag_list,
                },
            )
            resp.raise_for_status()
            run_info = resp.json()["run"]["info"]
            run_id: str = run_info["run_id"]
            artifact_uri = run_info.get("artifact_uri", "")
            if artifact_uri:
                self._artifact_prefix_cache[run_id] = self._extract_artifact_prefix(
                    run_id, artifact_uri
                )
            return run_id

    async def upload_artifact(
        self,
        run_id: str,
        path: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload an artifact to a run."""
        prefix = await self._resolve_artifact_prefix(run_id)
        full_path = f"{prefix}/{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.put(
                self._url(f"/api/2.0/mlflow-artifacts/artifacts/{full_path}"),
                content=content,
                headers={"Content-Type": content_type},
            )
            resp.raise_for_status()

    async def finish_run(self, run_id: str, status: str = "FINISHED") -> None:
        """Mark a run as finished or failed."""
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        payload: dict[str, Any] = {"run_id": run_id, "status": status}
        if status == "FINISHED":
            payload["end_time"] = now_ms
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("/api/2.0/mlflow/runs/update"),
                json=payload,
            )
            resp.raise_for_status()

    async def fail_run_quiet(self, run_id: str) -> None:
        """Best-effort mark a run as FAILED. Swallows all errors."""
        try:
            await self.finish_run(run_id, status="FAILED")
        except Exception:
            logger.warning("Could not mark MLflow run %s as FAILED", run_id)

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch a run by ID. Returns the full run dict."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url("/api/2.0/mlflow/runs/get"),
                params={"run_id": run_id},
            )
            resp.raise_for_status()
            run: dict[str, Any] = resp.json()["run"]
            return run

    async def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str = "",
        order_by: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search runs across experiments."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            payload: dict[str, Any] = {
                "experiment_ids": experiment_ids,
                "max_results": max_results,
            }
            if filter_string:
                payload["filter"] = filter_string
            if order_by:
                payload["order_by"] = order_by
            resp = await client.post(
                self._url("/api/2.0/mlflow/runs/search"),
                json=payload,
            )
            resp.raise_for_status()
            runs: list[dict[str, Any]] = resp.json().get("runs", [])
            return runs

    async def delete_run(self, run_id: str) -> None:
        """Delete a run. Silently ignores 404."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("/api/2.0/mlflow/runs/delete"),
                json={"run_id": run_id},
            )
            if resp.status_code != 404:
                resp.raise_for_status()

    async def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a tag on a run."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("/api/2.0/mlflow/runs/set-tag"),
                json={"run_id": run_id, "key": key, "value": value},
            )
            resp.raise_for_status()

    async def list_artifacts(self, run_id: str, path: str = "") -> list[dict[str, Any]]:
        """List artifacts under *path* for a run. Returns empty list on 404."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            params: dict[str, str] = {"run_id": run_id}
            if path:
                params["path"] = path
            resp = await client.get(
                self._url("/api/2.0/mlflow/artifacts/list"),
                params=params,
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return resp.json().get("files", [])  # type: ignore[no-any-return]

    async def artifact_exists(self, run_id: str, path: str) -> bool:
        """Check if an artifact exists without downloading it."""
        try:
            prefix = await self._resolve_artifact_prefix(run_id)
            full_path = f"{prefix}/{path}"
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.head(
                    self._url(f"/api/2.0/mlflow-artifacts/artifacts/{full_path}"),
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def get_artifact(self, run_id: str, path: str) -> bytes:
        """Download an artifact's raw bytes."""
        prefix = await self._resolve_artifact_prefix(run_id)
        full_path = f"{prefix}/{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url(f"/api/2.0/mlflow-artifacts/artifacts/{full_path}"),
            )
            resp.raise_for_status()
            return resp.content

    async def get_artifact_text(self, run_id: str, path: str) -> str | None:
        """Download an artifact as text. Returns None if artifact doesn't exist."""
        try:
            data = await self.get_artifact(run_id, path)
            return data.decode("utf-8")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            logger.warning(
                "MLflow artifact fetch failed for run %s path %s: %d",
                run_id,
                path,
                exc.response.status_code,
            )
            raise
        except Exception:
            logger.warning("Failed to fetch artifact %s from run %s", path, run_id, exc_info=True)
            raise

    async def register_model(
        self,
        name: str,
        run_id: str,
        description: str = "",
    ) -> bool:
        """Register a model version from a run. Returns True on success."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._url("/api/2.0/mlflow/registered-models/create"),
                json={"name": name},
            )
            if resp.status_code == 409:
                logger.info("Model %s already registered", name)
            elif resp.is_error:
                logger.warning("Failed to create registered model: %s", resp.text)
                return False

            source = f"runs:/{run_id}/model"
            version_resp = await client.post(
                self._url("/api/2.0/mlflow/model-versions/create"),
                json={
                    "name": name,
                    "source": source,
                    "run_id": run_id,
                    "description": description,
                },
            )
            version_resp.raise_for_status()
            logger.info("Registered model version %s from run %s", name, run_id)
        return True

    async def list_gateway_endpoints(self) -> list[dict[str, Any]]:
        """Fetch raw MLflow AI Gateway endpoint dicts."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url("/api/3.0/mlflow/gateway/endpoints/list"),
            )
            resp.raise_for_status()
            endpoints: list[dict[str, Any]] = resp.json().get("endpoints", [])
            return endpoints

    async def list_gateway_models(self) -> list[dict[str, str]]:
        """Parse gateway endpoints into model records.

        Returns a list of dicts with keys: name, provider, model_name.
        """
        endpoints = await self.list_gateway_endpoints()
        models: list[dict[str, str]] = []
        for ep in endpoints:
            provider = ""
            model_name = ""
            for mapping in ep.get("model_mappings", []):
                model_def = mapping.get("model_definition", {})
                if model_def:
                    provider = model_def.get("provider", "")
                    model_name = model_def.get("model_name", "")
                    break
            models.append(
                {
                    "name": ep.get("name", ""),
                    "provider": provider,
                    "model_name": model_name,
                }
            )
        return models
