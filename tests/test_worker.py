"""Tests for the background worker job execution lifecycle."""

import os
from typing import Any

import httpx
import pytest
import yaml

from amortized.core.config_translator import _build_synth_config, _generate_container_config
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


class TestBuildSynthConfig:
    def test_basic_config(self) -> None:
        config: dict[str, Any] = {"model": "openai/gpt-4o", "num_samples": 50}
        result = _build_synth_config(config)

        assert result["inference_config"]["model"] == "openai/gpt-4o"
        assert result["inference_config"]["temperature"] == 0.7
        assert result["inference_config"]["max_concurrency"] == 16
        assert result["num_samples"] == 50
        assert result["output_path"] == "output/generated_data.jsonl"

    def test_optional_inference_fields(self) -> None:
        config: dict[str, Any] = {
            "model": "openai/gpt-4o",
            "api_base": "http://localhost:8000/v1",
            "max_tokens": 1024,
            "top_p": 0.9,
            "seed": 42,
        }
        result = _build_synth_config(config)
        ic = result["inference_config"]

        assert ic["api_base"] == "http://localhost:8000/v1"
        assert "api_key" not in ic
        assert ic["max_tokens"] == 1024
        assert ic["top_p"] == 0.9
        assert ic["seed"] == 42

    def test_none_optionals_excluded(self) -> None:
        config: dict[str, Any] = {"model": "openai/gpt-4o", "max_tokens": None}
        result = _build_synth_config(config)
        assert "max_tokens" not in result["inference_config"]

    def test_strategy_params_passthrough(self) -> None:
        strategy = {"sampled_attributes": [{"name": "domain", "values": ["science"]}]}
        config: dict[str, Any] = {"model": "openai/gpt-4o", "strategy_params": strategy}
        result = _build_synth_config(config)
        assert result["strategy_params"]["sampled_attributes"] == strategy["sampled_attributes"]

    def test_input_data_merged_into_strategy(self) -> None:
        config: dict[str, Any] = {
            "model": "openai/gpt-4o",
            "strategy_params": {"sampled_attributes": []},
            "input_data": [{"text": "hello"}],
        }
        result = _build_synth_config(config)
        assert result["strategy_params"]["input_data"] == [{"text": "hello"}]

    def test_generate_container_config_sdg(self) -> None:
        config: dict[str, Any] = {"model": "openai/gpt-4o"}
        result = _generate_container_config("sdg", config)
        assert "inference_config" in result
        assert "num_samples" in result

    def test_generate_container_config_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="No container config"):
            _generate_container_config("unknown", {})


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
        assert parsed["micro_batch_size"] == 2
        assert parsed["ckpt_output_dir"] == "/output"
        assert "algorithm" not in parsed

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

    def test_thub_algos_use_thub_command(self) -> None:
        from amortized.worker import THUB_ALGOS, TRL_ONLY_ALGOS

        assert "sft" in THUB_ALGOS
        assert "lora_sft" in THUB_ALGOS
        assert "grpo" in THUB_ALGOS
        assert "gepa" in THUB_ALGOS
        assert "dpo" in TRL_ONLY_ALGOS
        assert "kto" in TRL_ONLY_ALGOS
