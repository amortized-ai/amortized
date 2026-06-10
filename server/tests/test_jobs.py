"""Tests for job CRUD API endpoints."""

import json
import os
import tempfile

import httpx
import pytest
from conftest import requires_training_hub_functional

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    """Use a temporary database for each test."""
    import amortized.config as config_mod
    import amortized.db as db_mod
    import amortized.db.connection as db_conn_mod

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
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


class TestCreateTrainingJob:
    @pytest.mark.asyncio
    async def test_create_training_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "data_path": "./data.jsonl",
                "output_dir": "./outputs",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "training"
        assert data["status"] == "queued"
        assert data["config"]["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_training_job_with_hyperparams(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "meta-llama/Llama-3-8B",
                "data_path": "./data.jsonl",
                "output_dir": "./outputs",
                "learning_rate": 1e-4,
                "lora_r": 32,
                "load_in_4bit": True,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["config"]["lora_r"] == 32
        assert data["config"]["load_in_4bit"] is True

    @pytest.mark.asyncio
    async def test_create_training_job_with_compute(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "data_path": "./data.jsonl",
                "compute": {"backend": "ssh", "gpus": 2},
                "metadata": {"team": "ml"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["metadata"]["backend"] == "ssh"
        assert data["metadata"]["gpus"] == 2
        assert data["metadata"]["team"] == "ml"
        assert "compute" not in data["config"]
        assert "metadata" not in data["config"]

    @pytest.mark.asyncio
    async def test_create_training_job_validation_error(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={"model_name_or_path": "test"},  # missing required fields
        )
        assert response.status_code == 422


class TestCreateSDGJob:
    @pytest.mark.asyncio
    async def test_create_sdg_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "model": "openai/gpt-4o",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "sdg"
        assert data["status"] == "queued"
        assert data["config"]["model"] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_create_sdg_job_with_compute(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "model": "openai/gpt-4o",
                "compute": {"backend": "ssh", "gpus": 1, "gpu_type": "A100"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["metadata"]["backend"] == "ssh"
        assert data["metadata"]["gpus"] == 1
        assert data["metadata"]["gpu_type"] == "A100"
        assert "compute" not in data["config"]

    @pytest.mark.asyncio
    async def test_create_sdg_job_with_metadata(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "model": "openai/gpt-4o",
                "compute": {"backend": "ssh"},
                "metadata": {"project": "demo"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["metadata"]["backend"] == "ssh"
        assert data["metadata"]["project"] == "demo"


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test",
                "output_dir": "test",
            },
        )
        await client.post(
            "/api/v1/jobs/sdg",
            json={
                "pipeline": "conversation",
                "model": "test",
            },
        )
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_filter_by_type(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test",
                "output_dir": "test",
            },
        )
        await client.post(
            "/api/v1/jobs/sdg",
            json={
                "pipeline": "conversation",
                "model": "test",
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
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test",
                "output_dir": "test",
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
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test",
                "output_dir": "test",
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


class TestJobMetrics:
    @pytest.mark.asyncio
    async def test_metrics_no_file(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test",
                "output_dir": "/tmp/nonexistent",
            },
        )
        job_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/jobs/{job_id}/metrics")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_metrics_with_file(self, client: httpx.AsyncClient) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "training_metrics.jsonl")
            with open(metrics_path, "w") as f:
                f.write(json.dumps({"step": 1, "loss": 3.5, "epoch": 0.1}) + "\n")
                f.write(json.dumps({"step": 2, "loss": 3.2, "epoch": 0.2}) + "\n")

            create_resp = await client.post(
                "/api/v1/jobs/training",
                json={
                    "algorithm": "sft",
                    "model_name_or_path": "test",
                    "data_path": "test",
                    "output_dir": tmpdir,
                },
            )
            job_id = create_resp.json()["id"]

            response = await client.get(f"/api/v1/jobs/{job_id}/metrics")
            assert response.status_code == 200
            metrics = response.json()
            assert len(metrics) == 2
            assert metrics[0]["step"] == 1
            assert metrics[1]["loss"] == 3.2

    @pytest.mark.asyncio
    async def test_metrics_sdg_job_rejected(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "pipeline": "conversation",
                "model": "openai/gpt-4o",
            },
        )
        job_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/jobs/{job_id}/metrics")
        assert response.status_code == 400


class TestJobArtifacts:
    @pytest.mark.asyncio
    async def test_artifacts_empty(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test",
                "output_dir": "test",
            },
        )
        job_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert response.status_code == 200
        assert response.json() == []


class TestConfigRedaction:
    @pytest.mark.asyncio
    async def test_api_key_redacted_in_get(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/jobs/sdg",
            json={"model": "openai/gpt-4o", "api_key": "sk-secret-123"},
        )
        assert create_resp.status_code == 201
        job_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["config"]["api_key"] == "***redacted***"

    @pytest.mark.asyncio
    async def test_api_key_redacted_in_create_response(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={"model": "openai/gpt-4o", "api_key": "sk-secret-123"},
        )
        assert response.status_code == 201
        assert response.json()["config"]["api_key"] == "***redacted***"

    @pytest.mark.asyncio
    async def test_api_key_redacted_in_list(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/jobs/sdg",
            json={"model": "openai/gpt-4o", "api_key": "sk-secret-123"},
        )
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["config"]["api_key"] == "***redacted***"

    @pytest.mark.asyncio
    async def test_no_api_key_unchanged(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={"model": "openai/gpt-4o", "num_samples": 50},
        )
        assert response.status_code == 201
        config = response.json()["config"]
        assert config["model"] == "openai/gpt-4o"
        assert config["num_samples"] == 50
        assert "api_key" not in config

    @pytest.mark.asyncio
    async def test_other_fields_not_redacted(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={"model": "openai/gpt-4o", "api_key": "sk-secret-123", "num_samples": 100},
        )
        assert response.status_code == 201
        config = response.json()["config"]
        assert config["api_key"] == "***redacted***"
        assert config["model"] == "openai/gpt-4o"
        assert config["num_samples"] == 100


class TestFlows:
    @pytest.mark.asyncio
    async def test_list_flows(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/flows")
        assert response.status_code == 200
        pipelines = response.json()
        assert isinstance(pipelines, list)
        for pipeline in pipelines:
            assert "name" in pipeline
            assert "description" in pipeline
            assert "supports_multi_turn" in pipeline


@requires_training_hub_functional
class TestEstimate:
    @pytest.mark.asyncio
    async def test_estimate_memory(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/estimate",
            json={
                "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "lora_r": 16,
                "batch_size": 2,
                "max_length": 2048,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "estimated_vram_gb" in data
        assert data["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert data["estimated_vram_gb"] > 0

    @pytest.mark.asyncio
    async def test_estimate_qlora(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/estimate",
            json={
                "model_name_or_path": "meta-llama/Llama-3-8B",
                "load_in_4bit": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["load_in_4bit"] is True


class TestErrorFieldSerialization:
    @pytest.mark.asyncio
    async def test_error_is_null_not_string_none(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test",
                "output_dir": "test",
            },
        )
        assert create_resp.status_code == 201
        job_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert data["error"] != "None"

    @pytest.mark.asyncio
    async def test_error_null_in_list_response(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test",
                "output_dir": "test",
            },
        )
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["error"] is None
        assert jobs[0]["error"] != "None"


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_has_gpu_info(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "gpu" in data
        assert "timestamp" in data
