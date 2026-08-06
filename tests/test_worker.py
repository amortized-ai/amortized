"""Tests for the background worker job execution lifecycle."""

import os
import shlex
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

        mock_repo = AsyncMock()
        mock_repo.get_job = AsyncMock(return_value=parent_job)

        with (
            patch("amortized.db.connection._get_shared_db", new_callable=AsyncMock),
            patch("amortized.db.repository.Repository", return_value=mock_repo),
            patch("amortized.jobs.common.config_mod") as mock_config,
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            result_config, pre_commands = await _resolve_parent_artifacts(training_job, config)

        assert result_config["data_path"] == "/amortized/work/data/generated_data"
        assert len(pre_commands) == 1
        assert "mlflow.artifacts.download_artifacts" in pre_commands[0]
        assert "run_id='mlflow-run-abc'" in pre_commands[0]
        assert "artifact_path='generated_data'" in pre_commands[0]

    @pytest.mark.asyncio
    async def test_no_parent_returns_unchanged(self) -> None:
        from amortized.worker import _resolve_parent_artifacts

        config: dict[str, object] = {"algorithm": "sft"}
        job: dict[str, object] = {"id": "j1", "type": "training"}
        result_config, pre_commands = await _resolve_parent_artifacts(job, config)
        assert result_config == config
        assert pre_commands == []


class TestCommandWrapping:
    def _wrap(
        self,
        command: list[str],
        pre_commands: list[str] | None = None,
        post_commands: list[str] | None = None,
    ) -> list[str]:
        pre = pre_commands or []
        post = post_commands or []
        final = command
        if pre or post:
            if command[:2] == ["sh", "-c"] and len(command) == 3:
                main_cmd = command[2]
            else:
                main_cmd = shlex.join(command)
            pre_chain = " && ".join([*pre, main_cmd])
            if post:
                post_chain = " ; ".join(post)
                final = ["sh", "-c", f"{pre_chain} && {{ {post_chain} ; true; }}"]
            else:
                final = ["sh", "-c", pre_chain]
        return final

    def test_no_pre_post_passes_through(self) -> None:
        cmd = ["thub", "lora-sft", "--config", "/amortized/config.yaml"]
        assert self._wrap(cmd) == cmd

    def test_pre_commands_only(self) -> None:
        cmd = ["thub", "lora-sft", "--config", "/amortized/config.yaml"]
        result = self._wrap(cmd, pre_commands=["python3 -c 'download()'"])
        assert result[0:2] == ["sh", "-c"]
        assert "python3 -c 'download()' && thub" in result[2]

    def test_post_commands_only(self) -> None:
        cmd = ["thub", "lora-sft", "--config", "/amortized/config.yaml"]
        result = self._wrap(cmd, post_commands=["python3 -c 'upload()'"])
        assert result[0:2] == ["sh", "-c"]
        assert "thub" in result[2]
        assert "{ python3 -c 'upload()' ; true; }" in result[2]

    def test_pre_and_post(self) -> None:
        cmd = ["thub", "lora-sft", "--config", "/amortized/config.yaml"]
        result = self._wrap(
            cmd,
            pre_commands=["python3 -c 'download()'"],
            post_commands=["python3 -c 'upload()'"],
        )
        shell_cmd = result[2]
        pre_idx = shell_cmd.index("download()")
        main_idx = shell_cmd.index("thub")
        post_idx = shell_cmd.index("upload()")
        assert pre_idx < main_idx < post_idx

    def test_existing_sh_c_not_double_wrapped(self) -> None:
        cmd = ["sh", "-c", "data-designer create && upload.py"]
        result = self._wrap(cmd, pre_commands=["python3 -c 'download()'"])
        shell_cmd = result[2]
        assert "data-designer create && upload.py" in shell_cmd
        assert shell_cmd.count("sh -c") == 0

    def test_post_commands_dont_fail_job(self) -> None:
        cmd = ["thub", "train"]
        result = self._wrap(cmd, post_commands=["false"])
        assert "; true; }" in result[2]
