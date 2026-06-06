"""Tests for job CRUD API endpoints."""

import json
import os
import tempfile

import httpx
import pytest

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
                "algorithm": "lora_sft",
                "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "./outputs",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "training"
        assert data["status"] == "queued"
        assert data["config"]["model_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_training_job_with_hyperparams(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "lora_sft",
                "model_path": "meta-llama/Llama-3-8B",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "./outputs",
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
    async def test_create_training_job_validation_error(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={"model_path": "test"},  # missing required fields
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
                "algorithm": "lora_sft",
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
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
                "algorithm": "lora_sft",
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
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
                "algorithm": "lora_sft",
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
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
                "algorithm": "lora_sft",
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
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
                "algorithm": "lora_sft",
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "/tmp/nonexistent",
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
                    "algorithm": "lora_sft",
                    "model_path": "test",
                    "data_path": "test",
                    "ckpt_output_dir": tmpdir,
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
                "algorithm": "lora_sft",
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
            },
        )
        job_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert response.status_code == 200
        assert response.json() == []


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


class TestEstimate:
    @pytest.mark.asyncio
    async def test_estimate_memory(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/estimate",
            json={
                "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "lora_r": 16,
                "batch_size": 2,
                "max_seq_len": 2048,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "estimated_vram_gb" in data
        assert data["model_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert data["estimated_vram_gb"] > 0

    @pytest.mark.asyncio
    async def test_estimate_qlora(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/estimate",
            json={
                "model_path": "meta-llama/Llama-3-8B",
                "load_in_4bit": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["load_in_4bit"] is True


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_has_gpu_info(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "gpu" in data
        assert "timestamp" in data
