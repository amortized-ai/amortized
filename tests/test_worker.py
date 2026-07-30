"""Tests for the background worker job execution lifecycle."""

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml

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
        from amortized.backends.local import LocalBackend
        from amortized.core.compute import register_backend, reset
        from amortized.db import init_db

        await init_db()
        reset()
        register_backend(LocalBackend())
        yield c  # type: ignore[misc]


class TestWorkerJobExecution:
    @pytest.mark.asyncio
    async def test_worker_picks_oldest_job_first(self, client: httpx.AsyncClient) -> None:
        resp1 = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test/first",
                    "data_path": "./data.jsonl",
                },
            },
        )
        await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test/second",
                    "data_path": "./data.jsonl",
                },
            },
        )
        first_id = resp1.json()["id"]

        from amortized.worker import _pick_pending_job

        job = await _pick_pending_job()
        assert job is not None
        assert job["id"] == first_id

    @pytest.mark.asyncio
    async def test_no_pending_jobs_returns_none(self, client: httpx.AsyncClient) -> None:
        from amortized.worker import _pick_pending_job

        job = await _pick_pending_job()
        assert job is None


class TestOrphanedJobCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_orphaned_jobs(self, client: httpx.AsyncClient) -> None:
        import aiosqlite

        from amortized.config import settings
        from amortized.worker import cleanup_orphaned_jobs

        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test/model",
                    "data_path": "./data.jsonl",
                },
            },
        )
        job_id = response.json()["id"]

        async with aiosqlite.connect(str(settings.db_path)) as db:
            await db.execute(
                "UPDATE jobs SET status = ? WHERE id = ?",
                ("running", job_id),
            )
            await db.commit()

        await cleanup_orphaned_jobs()

        response = await client.get(f"/api/v1/jobs/{job_id}")
        data = response.json()
        assert data["status"] == "failed"
        assert "Orphaned" in (data.get("error") or "")


class TestCancelRunningJob:
    @pytest.mark.asyncio
    async def test_cancel_completed_job_rejected(self, client: httpx.AsyncClient) -> None:
        import aiosqlite

        from amortized.config import settings

        response = await client.post(
            "/api/v1/jobs",
            json={
                "type": "training",
                "config": {
                    "algorithm": "sft",
                    "model_name_or_path": "test/model",
                    "data_path": "./data.jsonl",
                },
            },
        )
        job_id = response.json()["id"]

        async with aiosqlite.connect(str(settings.db_path)) as db:
            await db.execute(
                "UPDATE jobs SET status = ? WHERE id = ?",
                ("succeeded", job_id),
            )
            await db.commit()

        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 400


class TestTrainingHubConfig:
    def test_thub_config_yaml_sft(self) -> None:
        from amortized.worker import _training_hub_config_yaml

        config = {
            "algorithm": "sft",
            "model_name_or_path": "Qwen/Qwen3-0.6B",
            "data_path": "/data/train.jsonl",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "learning_rate": 0.0002,
            "output_dir": "/output",
        }
        result = _training_hub_config_yaml("sft", config)
        parsed = yaml.safe_load(result)
        assert parsed["model_path"] == "Qwen/Qwen3-0.6B"
        assert parsed["data_path"] == "/data/train.jsonl"
        assert parsed["num_epochs"] == 3
        assert parsed["effective_batch_size"] == 8
        assert parsed["ckpt_output_dir"] == "/output"
        assert parsed["max_seq_len"] == 2048
        assert parsed["max_batch_len"] == 60000
        assert "algorithm" not in parsed
        assert "micro_batch_size" not in parsed

    def test_thub_config_yaml_gepa_output_dir(self) -> None:
        from amortized.worker import _training_hub_config_yaml

        config = {
            "algorithm": "gepa",
            "model_name_or_path": "Qwen/Qwen3-0.6B",
            "output_dir": "/output",
        }
        result = _training_hub_config_yaml("gepa", config)
        parsed = yaml.safe_load(result)
        assert parsed["output_dir"] == "/output"
        assert "ckpt_output_dir" not in parsed

    def test_thub_config_skips_keys(self) -> None:
        from amortized.worker import _training_hub_config_yaml

        config = {
            "algorithm": "sft",
            "model_name_or_path": "test",
            "engine": "vllm",
            "use_peft": True,
            "qlora": True,
        }
        result = _training_hub_config_yaml("sft", config)
        parsed = yaml.safe_load(result)
        assert "engine" not in parsed
        assert "use_peft" not in parsed
        assert "qlora" not in parsed

    def test_thub_config_handles_any_algorithm(self) -> None:
        from amortized.worker import _training_hub_config_yaml

        algos = ("sft", "lora_sft", "osft", "grpo", "lora_grpo", "gepa", "dpo", "kto")
        for algo in algos:
            cfg = {"model_name_or_path": "test", "algorithm": algo}
            result = _training_hub_config_yaml(algo, cfg)
            parsed = yaml.safe_load(result)
            assert parsed["model_path"] == "test"
            assert "algorithm" not in parsed


class TestResolveParentArtifacts:
    @pytest.mark.asyncio
    async def test_upload_parent_chains_data_path(self) -> None:
        from amortized.backends import S3Download
        from amortized.worker import _resolve_parent_artifacts

        parent_job = {
            "id": "parent-upload-1",
            "type": "upload",
            "status": "succeeded",
            "mlflow_run_id": "mlflow-run-abc",
        }
        training_job = {
            "id": "training-1",
            "type": "training",
            "parent_job_id": "parent-upload-1",
        }
        config: dict[str, object] = {
            "algorithm": "sft",
            "model_name_or_path": "test/model",
        }
        s3_downloads: list[S3Download] = []

        mock_repo = AsyncMock()
        mock_repo.get_job = AsyncMock(return_value=parent_job)

        with (
            patch("amortized.worker._get_repo", return_value=mock_repo),
            patch(
                "amortized.worker._resolve_mlflow_artifact_uri",
                return_value="s3://bucket/mlflow/abc",
            ),
        ):
            result = await _resolve_parent_artifacts(training_job, config, s3_downloads)

        assert result["data_path"] == "/amortized/work/data"
        assert len(s3_downloads) == 1
        assert s3_downloads[0].s3_uri == "s3://bucket/mlflow/abc/generated_data/"
        assert s3_downloads[0].local_path == "/amortized/work/data"
