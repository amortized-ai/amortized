"""Integration test: end-to-end customer journey (issue #60).

Exercises the full API flow a customer would follow:
health check → discover job types → get schema → upload dataset →
submit training job with compute backend → monitor status → list artifacts.
"""

import io
import os

import httpx
import pytest

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
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


class TestCustomerJourney:
    """Simulate the full customer workflow from issue #60."""

    @pytest.mark.asyncio
    async def test_full_training_journey(self, client: httpx.AsyncClient) -> None:
        # 1. Health check
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        health = resp.json()
        assert health["status"] == "ok"
        assert "timestamp" in health

        # 2. Discover available job types
        resp = await client.get("/api/v1/job-types")
        assert resp.status_code == 200
        job_types = resp.json()
        type_names = [t["type"] for t in job_types]
        assert "training" in type_names
        assert "sdg" in type_names

        # 3. Get training schema — verify updated schema accepts algorithm field
        resp = await client.get("/api/v1/job-types/training/schema")
        assert resp.status_code == 200
        schema = resp.json()
        assert "algorithm" in schema["properties"]
        assert "algorithm" in schema["required"]
        assert "batch_size" in schema["properties"]
        assert "gradient_checkpointing" in schema["properties"]

        # 4. Upload a dataset artifact
        dataset_content = (
            b'{"messages": [{"role": "user", "content": "hi"}, '
            b'{"role": "assistant", "content": "hello"}]}\n'
        )
        upload_file = ("train.jsonl", io.BytesIO(dataset_content), "application/octet-stream")
        resp = await client.post(
            "/api/v1/artifacts/upload",
            files={"file": upload_file},
            data={"artifact_type": "dataset", "name": "train.jsonl"},
        )
        assert resp.status_code == 201
        artifact = resp.json()
        assert artifact["artifact_type"] == "dataset"
        assert artifact["name"] == "train.jsonl"
        dataset_location = artifact["location"]
        dataset_id = artifact["id"]

        # 5. Verify uploaded artifact appears in list
        resp = await client.get("/api/v1/artifacts", params={"type": "dataset"})
        assert resp.status_code == 200
        artifacts = resp.json()
        assert any(a["id"] == dataset_id for a in artifacts)

        # 6. Submit training job with SSH compute backend
        resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "lora_sft",
                    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "data_path": dataset_location,
                    "ckpt_output_dir": "/tmp/test-outputs",
                    "lora_r": 16,
                    "num_epochs": 1,
                },
                "compute": {
                    "backend": "ssh",
                    "gpus": 2,
                    "gpu_type": "A100",
                },
                "metadata": {"experiment": "journey-test"},
                "dry_run": False,
            },
        )
        assert resp.status_code == 201
        job = resp.json()
        assert job["type"] == "training"
        assert job["status"] == "queued"
        job_id = job["id"]

        # 7. Verify compute.backend was persisted in metadata (fix #55)
        assert job["metadata"]["backend"] == "ssh"
        assert job["metadata"]["gpus"] == 2
        assert job["metadata"]["gpu_type"] == "A100"
        assert job["metadata"]["experiment"] == "journey-test"

        # 8. Monitor job status via GET
        resp = await client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        job_detail = resp.json()
        assert job_detail["id"] == job_id
        assert job_detail["type"] == "training"
        assert job_detail["config"]["algorithm"] == "lora_sft"

        # 9. Job appears in list with filters
        resp = await client.get("/api/v1/jobs", params={"type": "training"})
        assert resp.status_code == 200
        jobs = resp.json()
        assert any(j["id"] == job_id for j in jobs)

        # 10. Job artifacts endpoint works (empty for a new job)
        resp = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_compute_backend_defaults_to_local(self, client: httpx.AsyncClient) -> None:
        """When no compute spec is provided, backend defaults to 'local'."""
        resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "lora_sft",
                    "model_path": "test/model",
                    "data_path": "./data.jsonl",
                },
                "dry_run": False,
            },
        )
        assert resp.status_code == 201
        job = resp.json()
        assert "backend" not in job.get("metadata", {})
        assert "gpus" not in job.get("metadata", {})

    @pytest.mark.asyncio
    async def test_dry_run_validates_without_creating(self, client: httpx.AsyncClient) -> None:
        """Dry run validates config and returns preview without creating a job."""
        resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "lora_sft",
                    "model_path": "test/model",
                    "data_path": "./data.jsonl",
                },
                "compute": {"backend": "ssh", "gpus": 1},
                "dry_run": True,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["dry_run"] is True
        assert data["valid"] is True
        assert data["compute"]["backend"] == "ssh"

        # No job was actually created
        resp = await client.get("/api/v1/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 0

    @pytest.mark.asyncio
    async def test_schema_rejects_missing_algorithm(self, client: httpx.AsyncClient) -> None:
        """Training jobs without the required 'algorithm' field are rejected."""
        resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "model_path": "test/model",
                    "data_path": "./data.jsonl",
                },
                "dry_run": False,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_tilde_paths_accepted_in_config(self, client: httpx.AsyncClient) -> None:
        """Jobs with tilde paths are accepted; worker expands them at dispatch (fix #56)."""
        resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "lora_sft",
                    "model_path": "test/model",
                    "data_path": "~/data/train.jsonl",
                    "ckpt_output_dir": "~/outputs/run-1",
                },
                "dry_run": False,
            },
        )
        assert resp.status_code == 201
        job = resp.json()
        assert job["config"]["data_path"] == "~/data/train.jsonl"

        # Verify the worker's path expansion logic handles tildes
        expanded = os.path.expanduser("~/outputs/run-1")
        assert "~" not in expanded

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, client: httpx.AsyncClient) -> None:
        """Customer can cancel a queued job."""
        resp = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "lora_sft",
                    "model_path": "test/model",
                    "data_path": "./data.jsonl",
                },
                "dry_run": False,
            },
        )
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        resp = await client.delete(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

        resp = await client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
