"""Tests for job CRUD API endpoints."""

import os

import httpx
import pytest

from amortized.main import app

TEST_DATABASE_URL = os.environ.get(
    "AMORTIZED_TEST_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized_test",
)


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db.connection as db_conn_mod

    os.environ["AMORTIZED_DATABASE_URL"] = TEST_DATABASE_URL
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


class TestCreateJob:
    @pytest.mark.asyncio
    async def test_create_training_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "data_path": "./data.jsonl",
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "training"
        assert data["status"] == "queued"
        assert data["config"]["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_sdg_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
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
                    "model_configs": [{"alias": "text", "model": "gpt-4o"}],
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "sdg"
        assert data["status"] == "queued"
        assert data["config"]["num_records"] == 10

    @pytest.mark.asyncio
    async def test_create_with_recipe(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
                "recipe": "models/qwen-1.5b-lora",
            },
        )
        assert response.status_code == 201
        assert response.json()["recipe"] == "models/qwen-1.5b-lora"

    @pytest.mark.asyncio
    async def test_create_with_parent_job_id(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
                "parent_job_id": "abc-123",
            },
        )
        assert response.status_code == 201
        assert response.json()["parent_job_id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_create_unknown_type(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "unknown",
                "config": {},
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_dry_run(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
                "dry_run": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["dry_run"] is True
        assert data["valid"] is True

    @pytest.mark.asyncio
    async def test_sdg_missing_columns(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={"type": "sdg", "config": {}},
        )
        assert response.status_code == 422
        assert "columns" in str(response.json())

    @pytest.mark.asyncio
    async def test_sdg_empty_columns(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={"type": "sdg", "config": {"columns": []}},
        )
        assert response.status_code == 422
        assert "non-empty" in str(response.json())

    @pytest.mark.asyncio
    async def test_sdg_column_missing_fields(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={"type": "sdg", "config": {"columns": [{"name": "q"}]}},
        )
        assert response.status_code == 422
        assert "column_type" in str(response.json())

    @pytest.mark.asyncio
    async def test_sdg_llm_text_missing_model_configs(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
                    "columns": [
                        {
                            "column_type": "llm-text",
                            "name": "question",
                            "model_alias": "text",
                            "system_prompt": "Generate a question.",
                            "prompt": "{{ content }}",
                        }
                    ],
                },
            },
        )
        assert response.status_code == 422
        assert "model_configs" in str(response.json())

    @pytest.mark.asyncio
    async def test_sdg_model_alias_mismatch(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
                    "columns": [
                        {
                            "column_type": "llm-text",
                            "name": "question",
                            "model_alias": "nonexistent",
                            "system_prompt": "Generate.",
                            "prompt": "{{ content }}",
                        }
                    ],
                    "model_configs": [
                        {"alias": "text", "model": "gpt-4o"},
                    ],
                },
            },
        )
        assert response.status_code == 422
        assert "nonexistent" in str(response.json())

    @pytest.mark.asyncio
    async def test_sdg_unknown_column_type(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
                    "columns": [{"column_type": "llm_text", "name": "q"}],
                },
            },
        )
        assert response.status_code == 422
        assert "llm_text" in str(response.json())

    @pytest.mark.asyncio
    async def test_sdg_uuid_sampler_valid(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
                    "columns": [
                        {
                            "column_type": "sampler",
                            "name": "id",
                            "sampler_type": "uuid",
                            "params": {},
                        }
                    ],
                },
                "dry_run": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

    @pytest.mark.asyncio
    async def test_sdg_valid_config(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
                    "columns": [
                        {
                            "column_type": "sampler",
                            "name": "difficulty",
                            "sampler_type": "category",
                            "params": {"values": ["Easy", "Hard"]},
                        },
                        {
                            "column_type": "llm-text",
                            "name": "question",
                            "model_alias": "text",
                            "system_prompt": "Generate a question.",
                            "prompt": "Difficulty: {{ difficulty }}",
                        },
                    ],
                    "model_configs": [{"alias": "text", "model": "gpt-4o"}],
                },
            },
        )
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_sdg_dry_run_invalid(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={"type": "sdg", "config": {"columns": []}, "dry_run": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    @pytest.mark.asyncio
    async def test_sdg_dry_run_valid(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
                    "columns": [
                        {
                            "column_type": "sampler",
                            "name": "topic",
                            "sampler_type": "category",
                            "params": {"values": ["A", "B"]},
                        }
                    ],
                },
                "dry_run": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
            },
        )
        await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
                    "num_records": 10,
                    "columns": [
                        {
                            "column_type": "sampler",
                            "name": "q",
                            "sampler_type": "category",
                            "params": {"values": ["A", "B"]},
                        }
                    ],
                },
            },
        )
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_filter_by_type(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
            },
        )
        await client.post(
            "/api/v1/jobs",
            json={
                "type": "sdg",
                "config": {
                    "num_records": 10,
                    "columns": [
                        {
                            "column_type": "sampler",
                            "name": "q",
                            "sampler_type": "category",
                            "params": {"values": ["A", "B"]},
                        }
                    ],
                },
            },
        )
        response = await client.get("/api/v1/jobs?type=training")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["type"] == "training"


class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_existing_job(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
            },
        )
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
        create_resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
            },
        )
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
        create_resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
            },
        )
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
        create_resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                },
            },
        )
        job_id = create_resp.json()["id"]
        response = await client.get(f"/api/v1/jobs/{job_id}/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["logs"] == []

    @pytest.mark.asyncio
    async def test_logs_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs/nonexistent/logs")
        assert response.status_code == 404
