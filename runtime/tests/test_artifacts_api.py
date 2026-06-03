"""Tests for top-level artifact CRUD API endpoints."""

import os

import httpx
import pytest

from amortized_runtime.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized_runtime.config as config_mod
    import amortized_runtime.db as db_mod
    import amortized_runtime.db.connection as db_conn_mod

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized_runtime.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


class TestCreateArtifact:
    @pytest.mark.asyncio
    async def test_create_artifact(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/artifacts",
            json={
                "name": "my-model",
                "artifact_type": "adapter_weights",
                "location": "/tmp/adapter_model.safetensors",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my-model"
        assert data["artifact_type"] == "adapter_weights"
        assert data["location"] == "/tmp/adapter_model.safetensors"
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_with_producer_job(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/artifacts",
            json={
                "name": "generated-data",
                "artifact_type": "dataset",
                "location": "s3://bucket/data.jsonl",
                "producer_job": "job-123",
                "metadata": {"rows": 500},
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["producer_job"] == "job-123"
        assert data["metadata"] == {"rows": 500}

    @pytest.mark.asyncio
    async def test_create_missing_required_fields(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/api/v1/artifacts", json={"name": "x"})
        assert resp.status_code == 422


class TestListArtifacts:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/artifacts")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_all(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/artifacts",
            json={"name": "a", "artifact_type": "dataset", "location": "/a"},
        )
        await client.post(
            "/api/v1/artifacts",
            json={"name": "b", "artifact_type": "model", "location": "/b"},
        )
        resp = await client.get("/api/v1/artifacts")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @pytest.mark.asyncio
    async def test_filter_by_type(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/artifacts",
            json={"name": "a", "artifact_type": "dataset", "location": "/a"},
        )
        await client.post(
            "/api/v1/artifacts",
            json={"name": "b", "artifact_type": "model", "location": "/b"},
        )
        resp = await client.get("/api/v1/artifacts", params={"type": "dataset"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["artifact_type"] == "dataset"

    @pytest.mark.asyncio
    async def test_filter_by_producer_job(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/artifacts",
            json={
                "name": "a",
                "artifact_type": "dataset",
                "location": "/a",
                "producer_job": "j1",
            },
        )
        await client.post(
            "/api/v1/artifacts",
            json={"name": "b", "artifact_type": "model", "location": "/b"},
        )
        resp = await client.get("/api/v1/artifacts", params={"producer_job": "j1"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "a"


class TestGetArtifact:
    @pytest.mark.asyncio
    async def test_get_existing(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/artifacts",
            json={"name": "a", "artifact_type": "model", "location": "/a"},
        )
        artifact_id = create_resp.json()["id"]
        resp = await client.get(f"/api/v1/artifacts/{artifact_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "a"

    @pytest.mark.asyncio
    async def test_get_not_found(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/artifacts/nonexistent")
        assert resp.status_code == 404


class TestDeleteArtifact:
    @pytest.mark.asyncio
    async def test_delete_existing(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/artifacts",
            json={"name": "a", "artifact_type": "model", "location": "/a"},
        )
        artifact_id = create_resp.json()["id"]
        resp = await client.delete(f"/api/v1/artifacts/{artifact_id}")
        assert resp.status_code == 204

        resp = await client.get(f"/api/v1/artifacts/{artifact_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, client: httpx.AsyncClient) -> None:
        resp = await client.delete("/api/v1/artifacts/nonexistent")
        assert resp.status_code == 404


class TestDownloadArtifact:
    @pytest.mark.asyncio
    async def test_download_local_file(
        self, client: httpx.AsyncClient, tmp_path: object
    ) -> None:
        file_path = str(tmp_path) + "/test_file.txt"
        with open(file_path, "w") as f:
            f.write("test content")

        create_resp = await client.post(
            "/api/v1/artifacts",
            json={"name": "test", "artifact_type": "log", "location": file_path},
        )
        artifact_id = create_resp.json()["id"]
        resp = await client.get(f"/api/v1/artifacts/{artifact_id}/download")
        assert resp.status_code == 200
        assert resp.text == "test content"

    @pytest.mark.asyncio
    async def test_download_uri_returns_location(
        self, client: httpx.AsyncClient
    ) -> None:
        create_resp = await client.post(
            "/api/v1/artifacts",
            json={
                "name": "remote",
                "artifact_type": "model",
                "location": "s3://bucket/model.bin",
            },
        )
        artifact_id = create_resp.json()["id"]
        resp = await client.get(f"/api/v1/artifacts/{artifact_id}/download")
        assert resp.status_code == 200
        assert resp.json()["location"] == "s3://bucket/model.bin"

    @pytest.mark.asyncio
    async def test_download_not_found(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/artifacts/nonexistent/download")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_download_missing_file(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/artifacts",
            json={
                "name": "gone",
                "artifact_type": "model",
                "location": "/nonexistent/file.bin",
            },
        )
        artifact_id = create_resp.json()["id"]
        resp = await client.get(f"/api/v1/artifacts/{artifact_id}/download")
        assert resp.status_code == 404


class TestDatasetsDeprecation:
    @pytest.mark.asyncio
    async def test_create_dataset_has_deprecation_headers(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post(
            "/api/v1/datasets",
            json={"filename": "test.jsonl", "rows": [{"a": 1}]},
        )
        assert resp.status_code == 200
        assert resp.headers.get("Deprecation") == "true"
        assert "Sunset" in resp.headers

    @pytest.mark.asyncio
    async def test_convert_dataset_has_deprecation_headers(
        self, client: httpx.AsyncClient, tmp_path: object
    ) -> None:
        src = str(tmp_path) + "/src.jsonl"
        with open(src, "w") as f:
            f.write('{"question": "q", "answer": "a"}\n')

        resp = await client.post(
            "/api/v1/datasets/convert",
            json={"source_path": src, "output_filename": "out.jsonl"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("Deprecation") == "true"


class TestJobArtifactsStillWork:
    @pytest.mark.asyncio
    async def test_job_artifacts_endpoint(self, client: httpx.AsyncClient) -> None:
        job_resp = await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test/model",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "./outputs",
            },
        )
        assert job_resp.status_code == 201
        job_id = job_resp.json()["id"]

        resp = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert resp.status_code == 200
        assert resp.json() == []
