"""Tests for job CRUD API endpoints."""

import os

import httpx
import pytest

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db.connection as db_conn_mod

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


TRAINING_BODY = {
    "algorithm": "sft",
    "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "data_path": "./data.jsonl",
}

SDG_BODY = {
    "num_records": 10,
    "columns": [
        {
            "column_type": "sampler",
            "name": "q",
            "sampler_type": "category",
            "params": {"values": ["A", "B"]},
        }
    ],
}


async def _create_training(client: httpx.AsyncClient, **overrides: object) -> httpx.Response:
    body = {**TRAINING_BODY, **overrides}
    return await client.post("/api/v1/jobs/training", json=body)


async def _create_sdg(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post("/api/v1/jobs/sdg", json=SDG_BODY)


class TestCreateJob:
    @pytest.mark.asyncio
    async def test_create_training_job(self, client: httpx.AsyncClient) -> None:
        response = await _create_training(client)
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "training"
        assert data["status"] == "queued"
        assert data["config"]["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_sdg_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "num_records": 10,
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                        "system_prompt": "Generate a question.",
                        "prompt": "Context: {{ content }}",
                    }
                ],
                "model_configs": [{"alias": "text", "model": "gpt-4o", "provider": "gateway"}],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "sdg"
        assert data["status"] == "queued"
        assert data["config"]["num_records"] == 10

    @pytest.mark.asyncio
    async def test_create_training_with_parent_job_id(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        response = await _create_training(client, parent_job_id="abc-123")
        assert response.status_code == 201
        assert response.json()["parent_job_id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_training_missing_algorithm(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={"model_name_or_path": "test"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_training_missing_model(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={"algorithm": "sft"},
        )
        assert response.status_code == 422


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client: httpx.AsyncClient) -> None:
        await _create_training(client)
        await _create_sdg(client)
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_filter_by_type(self, client: httpx.AsyncClient) -> None:
        await _create_training(client)
        await _create_sdg(client)
        response = await client.get("/api/v1/jobs?type=training")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["type"] == "training"


class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_existing_job(self, client: httpx.AsyncClient) -> None:
        create_resp = await _create_training(client)
        job_id = create_resp.json()["id"]
        response = await client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["id"] == job_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404


class TestCancelJob:
    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, client: httpx.AsyncClient) -> None:
        create_resp = await _create_training(client)
        job_id = create_resp.json()["id"]
        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        response = await client.delete("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404


class TestErrorFieldSerialization:
    @pytest.mark.asyncio
    async def test_error_is_null_not_string_none(self, client: httpx.AsyncClient) -> None:
        create_resp = await _create_training(client)
        job_id = create_resp.json()["id"]
        response = await client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_has_gpu_info(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "gpu" in data
        assert "timestamp" in data


class TestConfig:
    @pytest.mark.asyncio
    async def test_config_endpoint(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert "default_compute_backend" in data
        assert "compute_namespace" in data
        assert "mlflow_tracking_uri" in data
        assert "mlflow_gateway_uri" in data
        assert "available_backends" in data
        assert "version" in data


class TestJobLogs:
    @pytest.mark.asyncio
    async def test_logs_no_handle(self, client: httpx.AsyncClient) -> None:
        create_resp = await _create_training(client)
        job_id = create_resp.json()["id"]
        response = await client.get(f"/api/v1/jobs/{job_id}/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["logs"] == []

    @pytest.mark.asyncio
    async def test_logs_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs/nonexistent/logs")
        assert response.status_code == 404
