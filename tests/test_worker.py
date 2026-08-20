"""Tests for the background worker job execution lifecycle."""

import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import httpx
import pytest
import yaml
from conftest import TEST_DATABASE_URL

from amortized.main import app


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
        from amortized.backends.local import LocalBackend
        from amortized.core.compute import register_backend, reset
        from amortized.db import init_db

        await init_db()
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute("TRUNCATE jobs")
        reset()
        register_backend(LocalBackend())
        yield c  # type: ignore[misc]


class TestWorkerJobExecution:
    @pytest.mark.asyncio
    async def test_worker_picks_oldest_job_first(self, client: httpx.AsyncClient) -> None:
        resp1 = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test/first",
                "data_path": "./data.jsonl",
            },
        )
        await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test/second",
                "data_path": "./data.jsonl",
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
        from amortized.worker import cleanup_orphaned_jobs

        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test/model",
                "data_path": "./data.jsonl",
            },
        )
        job_id = response.json()["id"]

        async with asyncpg.create_pool(dsn=TEST_DATABASE_URL) as pool, pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status = $1 WHERE id = $2",
                "running",
                job_id,
            )

        await cleanup_orphaned_jobs()

        response = await client.get(f"/api/v1/jobs/{job_id}")
        data = response.json()
        assert data["status"] == "failed"
        assert "Orphaned" in (data.get("error") or "")


class TestCancelRunningJob:
    @pytest.mark.asyncio
    async def test_cancel_completed_job_rejected(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "algorithm": "sft",
                "model_name_or_path": "test/model",
                "data_path": "./data.jsonl",
            },
        )
        job_id = response.json()["id"]

        async with asyncpg.create_pool(dsn=TEST_DATABASE_URL) as pool, pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status = $1 WHERE id = $2",
                "succeeded",
                job_id,
            )

        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 400


class TestTrainingHubConfig:
    def test_thub_config_yaml_sft(self) -> None:
        from amortized.jobs.training import _training_hub_config_yaml

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
        from amortized.jobs.training import _training_hub_config_yaml

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
        from amortized.jobs.training import _training_hub_config_yaml

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
        from amortized.jobs.training import _training_hub_config_yaml

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

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = AsyncMock()

        with (
            patch("amortized.db.connection.get_pool", return_value=mock_pool),
            patch("amortized.db.repository.Repository", return_value=mock_repo),
            patch("amortized.jobs.common.config_mod") as mock_config,
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            result_config, pre_commands = await _resolve_parent_artifacts(training_job, config)

        assert result_config["data_path"] == "/amortized/work/data/generated_data"
        assert len(pre_commands) == 1
        assert pre_commands[0] == (
            "mlflow artifacts download -r mlflow-run-abc -a generated_data -d /amortized/work/data"
        )

    @pytest.mark.asyncio
    async def test_no_parent_returns_unchanged(self) -> None:
        from amortized.worker import _resolve_parent_artifacts

        config: dict[str, object] = {"algorithm": "sft"}
        job: dict[str, object] = {"id": "j1", "type": "training"}
        result_config, pre_commands = await _resolve_parent_artifacts(job, config)
        assert result_config == config
        assert pre_commands == []


class TestCommandWrapping:
    def test_no_pre_post_passes_through(self) -> None:
        from amortized.worker import _wrap_command

        cmd = ["thub", "lora-sft", "--config", "/amortized/config.yaml"]
        assert _wrap_command(cmd, [], []) == cmd

    def test_pre_commands_only(self) -> None:
        from amortized.worker import _wrap_command

        cmd = ["thub", "lora-sft", "--config", "/amortized/config.yaml"]
        result = _wrap_command(cmd, ["mlflow artifacts download -r abc"], [])
        assert result[0:2] == ["sh", "-c"]
        assert "mlflow artifacts download -r abc && thub" in result[2]

    def test_post_commands_only(self) -> None:
        from amortized.worker import _wrap_command

        cmd = ["thub", "lora-sft", "--config", "/amortized/config.yaml"]
        result = _wrap_command(cmd, [], ["python3 -c 'upload()'"])
        assert result[0:2] == ["sh", "-c"]
        assert "thub" in result[2]
        assert "python3 -c 'upload()'" in result[2]

    def test_pre_and_post(self) -> None:
        from amortized.worker import _wrap_command

        cmd = ["thub", "lora-sft", "--config", "/amortized/config.yaml"]
        result = _wrap_command(
            cmd,
            ["mlflow artifacts download -r abc"],
            ["python3 -c 'upload()'"],
        )
        shell_cmd = result[2]
        pre_idx = shell_cmd.index("mlflow artifacts download")
        main_idx = shell_cmd.index("thub")
        post_idx = shell_cmd.index("upload()")
        assert pre_idx < main_idx < post_idx

    def test_existing_sh_c_not_double_wrapped(self) -> None:
        from amortized.worker import _wrap_command

        cmd = ["sh", "-c", "data-designer create && upload.py"]
        result = _wrap_command(cmd, ["mlflow artifacts download -r abc"], [])
        shell_cmd = result[2]
        assert "data-designer create && upload.py" in shell_cmd
        assert shell_cmd.count("sh -c") == 0

    def test_post_commands_fail_fast(self) -> None:
        from amortized.worker import _wrap_command

        cmd = ["thub", "train"]
        result = _wrap_command(cmd, [], ["cmd1", "cmd2"])
        assert "cmd1 && cmd2" in result[2]


class TestUploadBuilder:
    @pytest.mark.asyncio
    async def test_build_generates_pre_command(self) -> None:
        from amortized.jobs.upload import build

        job = {"id": "doc-1", "type": "upload"}
        config = {
            "mlflow_upload_run_id": "abc123",
            "artifact_path": "source",
            "filename": "test.pdf",
        }
        result = await build(job, config, {})
        assert len(result.pre_commands) == 1
        assert "mlflow artifacts download" in result.pre_commands[0]
        assert "abc123" in result.pre_commands[0]
        assert "-a source" in result.pre_commands[0]

    @pytest.mark.asyncio
    async def test_build_missing_run_id_raises(self) -> None:
        from amortized.jobs.base import JobBuildError
        from amortized.jobs.upload import build

        job = {"id": "doc-1", "type": "upload"}
        config = {"filename": "test.pdf"}
        with pytest.raises(JobBuildError, match="mlflow_upload_run_id"):
            await build(job, config, {})


class TestCreateMlflowRun:
    @pytest.mark.asyncio
    async def test_returns_run_id_on_success(self) -> None:
        from amortized.worker import _create_mlflow_run

        mock_client = AsyncMock()
        mock_client.ensure_experiment = AsyncMock(return_value="exp-1")
        mock_client.create_run = AsyncMock(return_value="abc123def456abc123def456abc123de")

        with (
            patch("amortized.worker.config_mod") as mock_config,
            patch("amortized.core.mlflow_client.MLflowClient", return_value=mock_client),
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            result = await _create_mlflow_run("test/exp", "job-123", "training")

        assert result == "abc123def456abc123def456abc123de"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_tracking_uri(self) -> None:
        from amortized.worker import _create_mlflow_run

        with patch("amortized.worker.config_mod") as mock_config:
            mock_config.settings.mlflow_tracking_uri = ""
            result = await _create_mlflow_run("test/exp", "job-123", "training")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self) -> None:
        from amortized.worker import _create_mlflow_run

        mock_client = AsyncMock()
        mock_client.ensure_experiment = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with (
            patch("amortized.worker.config_mod") as mock_config,
            patch("amortized.core.mlflow_client.MLflowClient", return_value=mock_client),
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            result = await _create_mlflow_run("test/exp", "job-123", "training")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_value_error(self) -> None:
        from amortized.worker import _create_mlflow_run

        mock_client = AsyncMock()
        mock_client.ensure_experiment = AsyncMock(return_value="exp-1")
        mock_client.create_run = AsyncMock(side_effect=ValueError("bad artifact URI format"))

        with (
            patch("amortized.worker.config_mod") as mock_config,
            patch("amortized.core.mlflow_client.MLflowClient", return_value=mock_client),
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            result = await _create_mlflow_run("test/exp", "job-123", "training")

        assert result is None

    @pytest.mark.asyncio
    async def test_propagates_programming_errors(self) -> None:
        from amortized.worker import _create_mlflow_run

        mock_client = AsyncMock()
        mock_client.ensure_experiment = AsyncMock(
            side_effect=TypeError("unexpected keyword argument")
        )

        with (
            patch("amortized.worker.config_mod") as mock_config,
            patch("amortized.core.mlflow_client.MLflowClient", return_value=mock_client),
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            with pytest.raises(TypeError):
                await _create_mlflow_run("test/exp", "job-123", "training")


class TestFinishMlflowRun:
    @pytest.mark.asyncio
    async def test_logs_warning_when_not_critical(self, caplog: pytest.LogCaptureFixture) -> None:
        from amortized.worker import _finish_mlflow_run

        mock_client = AsyncMock()
        mock_client.finish_run = AsyncMock(side_effect=OSError("connection lost"))

        with (
            patch("amortized.worker.config_mod") as mock_config,
            patch("amortized.core.mlflow_client.MLflowClient", return_value=mock_client),
            caplog.at_level(logging.WARNING, logger="amortized.worker"),
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            await _finish_mlflow_run("run-abc")

        assert "Failed to finish MLflow run" in caplog.text
        assert caplog.records[-1].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_logs_error_when_critical(self, caplog: pytest.LogCaptureFixture) -> None:
        from amortized.worker import _finish_mlflow_run

        mock_client = AsyncMock()
        mock_client.finish_run = AsyncMock(side_effect=OSError("connection lost"))

        with (
            patch("amortized.worker.config_mod") as mock_config,
            patch("amortized.core.mlflow_client.MLflowClient", return_value=mock_client),
            caplog.at_level(logging.WARNING, logger="amortized.worker"),
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            await _finish_mlflow_run("run-abc", critical=True)

        assert "Failed to finish MLflow run" in caplog.text
        assert caplog.records[-1].levelno == logging.ERROR

    @pytest.mark.asyncio
    async def test_noop_when_no_run_id(self) -> None:
        from amortized.worker import _finish_mlflow_run

        with patch("amortized.worker.config_mod") as mock_config:
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            await _finish_mlflow_run("")


class TestExtractMlflowRunId:
    @pytest.mark.asyncio
    async def test_logs_warning_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        from amortized.backends import BackendHandle
        from amortized.worker import _extract_mlflow_run_id

        backend = AsyncMock()
        backend.logs = AsyncMock(side_effect=OSError("no logs available"))
        handle = BackendHandle(backend_name="test", job_id="j1")

        with caplog.at_level(logging.WARNING, logger="amortized.worker"):
            result = await _extract_mlflow_run_id(backend, handle)

        assert result == ""
        assert "Failed to extract MLflow run ID from logs" in caplog.text


class TestPostCommandGuard:
    @pytest.mark.asyncio
    async def test_post_commands_stripped_when_mlflow_run_not_created(self) -> None:
        """When _create_mlflow_run fails, post_commands from the builder should
        not be passed to _wrap_command, preventing silent artifact upload failures."""
        from amortized.worker import _wrap_command

        cmd = ["thub", "train"]
        post_commands = ["mlflow artifacts log-artifacts -l /output -r $MLFLOW_RUN_ID -a model"]

        mlflow_run_created = False
        guarded = post_commands if mlflow_run_created else []
        result = _wrap_command(cmd, [], guarded)

        assert result == cmd

    @pytest.mark.asyncio
    async def test_post_commands_included_when_mlflow_run_created(self) -> None:
        from amortized.worker import _wrap_command

        cmd = ["thub", "train"]
        post_commands = ["mlflow artifacts log-artifacts -l /output -r $MLFLOW_RUN_ID -a model"]

        mlflow_run_created = True
        guarded = post_commands if mlflow_run_created else []
        result = _wrap_command(cmd, [], guarded)

        assert "mlflow artifacts log-artifacts" in result[2]
