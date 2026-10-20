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


class TestResourceAwareAdmission:
    @pytest.mark.asyncio
    async def test_cpu_job_runs_past_unfittable_gpu_job(
        self, client: httpx.AsyncClient
    ) -> None:
        """With no GPU budget free, a queued CPU job (SDG) is claimed ahead of an
        older GPU job (training) that does not fit — the core unblocker. When a GPU
        frees, the older training job is then admitted."""
        import amortized.config as config_mod
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _claim_runnable_job

        ns = config_mod.settings.compute_namespace
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            await repo.create_job(
                job_id="train-old", job_type=JobType.training, config={"algorithm": "sft"},
                created_at="2026-01-01T00:00:00+00:00", k8s_namespace=ns,
            )
            await repo.create_job(
                job_id="sdg-new", job_type=JobType.sdg, config={},
                created_at="2026-01-01T00:00:01+00:00", k8s_namespace=ns,
            )

        # Budget exhausted: skip the training job (needs 1 GPU), claim the CPU job.
        claimed = await _claim_runnable_job(ns, gpu_free=0)
        assert claimed is not None
        assert claimed["id"] == "sdg-new"
        assert claimed["status"] == "provisioning"

        # A GPU is free now: the training job is the oldest that fits, so it wins.
        claimed = await _claim_runnable_job(ns, gpu_free=1)
        assert claimed is not None
        assert claimed["id"] == "train-old"

        # Both claimed — nothing runnable remains.
        assert await _claim_runnable_job(ns, gpu_free=1) is None

    @pytest.mark.asyncio
    async def test_cpu_job_found_past_a_full_page_of_unfittable_gpu_jobs(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CPU job must not be stranded behind a full page of unfittable GPU jobs —
        _claim_runnable_job pages through the whole queue (keyset cursor)."""
        import amortized.config as config_mod
        import amortized.worker as worker_mod
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _claim_runnable_job

        monkeypatch.setattr(worker_mod, "_CANDIDATE_BATCH", 2)  # force multi-page scan
        ns = config_mod.settings.compute_namespace
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            for i in range(3):  # three unfittable GPU jobs span the first page(s)
                await repo.create_job(
                    job_id=f"gpu{i}", job_type=JobType.training, config={"algorithm": "sft"},
                    created_at=f"2026-01-01T00:00:0{i}+00:00", k8s_namespace=ns,
                )
            await repo.create_job(
                job_id="cpu-last", job_type=JobType.sdg, config={},
                created_at="2026-01-01T00:00:09+00:00", k8s_namespace=ns,
            )

        # Budget exhausted: all 3 GPU jobs skipped across pages, the CPU job still claimed.
        claimed = await _claim_runnable_job(ns, gpu_free=0)
        assert claimed is not None
        assert claimed["id"] == "cpu-last"


class TestRecovery:
    @pytest.mark.asyncio
    async def test_resume_leaves_provisioning_job_on_indeterminate_status(
        self, client: httpx.AsyncClient
    ) -> None:
        """A transient status failure must NOT requeue a provisioning job (the resource
        may be live → double-submit). Leave it untouched for a later retry."""
        import amortized.config as config_mod
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _resume_inflight_jobs

        ns = config_mod.settings.compute_namespace
        handle = '{"backend_name": "kubernetes", "job_id": "prov1"}'
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            await repo.create_job(
                job_id="prov1", job_type=JobType.training, config={"algorithm": "sft"},
                created_at="2026-01-01T00:00:00+00:00", k8s_namespace=ns,
            )
            await repo.update_job("prov1", status="provisioning", backend_handle=handle)

        fake = MagicMock()
        fake.status = AsyncMock(side_effect=RuntimeError("transient API blip"))
        with patch("amortized.worker.get_backend", return_value=fake):
            resumable = await _resume_inflight_jobs()

        assert "prov1" not in [j["id"] for j in resumable]
        async with get_pool().acquire() as conn:
            row = await Repository(conn).get_job("prov1")
        assert row is not None
        assert row["status"] == "provisioning"  # left for retry, NOT requeued

    @pytest.mark.asyncio
    async def test_resume_requeues_provisioning_job_when_resource_confirmed_gone(
        self, client: httpx.AsyncClient
    ) -> None:
        """A 404 (not_found) is a confirmed-gone signal → requeue the provisioning job."""
        import amortized.config as config_mod
        from amortized.backends import BackendStatus
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _resume_inflight_jobs

        ns = config_mod.settings.compute_namespace
        handle = '{"backend_name": "kubernetes", "job_id": "gone1"}'
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            await repo.create_job(
                job_id="gone1", job_type=JobType.training, config={"algorithm": "sft"},
                created_at="2026-01-01T00:00:00+00:00", k8s_namespace=ns,
            )
            await repo.update_job("gone1", status="provisioning", backend_handle=handle)

        fake = MagicMock()
        fake.status = AsyncMock(return_value=BackendStatus(running=False, not_found=True))
        with patch("amortized.worker.get_backend", return_value=fake):
            await _resume_inflight_jobs()

        async with get_pool().acquire() as conn:
            row = await Repository(conn).get_job("gone1")
        assert row is not None
        assert row["status"] == "queued"  # confirmed gone → requeued

    @pytest.mark.asyncio
    async def test_resume_adopts_submitted_job_with_no_saved_handle(
        self, client: httpx.AsyncClient
    ) -> None:
        """A crash between submit() and persisting the handle leaves a provisioning row
        with no handle though the backend resource exists. Recovery must reconstruct the
        deterministic handle and adopt the live resource, NOT requeue it into a collision."""
        import amortized.config as config_mod
        from amortized.backends import BackendHandle, BackendStatus
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _resume_inflight_jobs

        ns = config_mod.settings.compute_namespace
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            await repo.create_job(
                job_id="nohandle1", job_type=JobType.training, config={"algorithm": "sft"},
                created_at="2026-01-01T00:00:00+00:00", k8s_namespace=ns,
            )
            await repo.update_job("nohandle1", status="provisioning")  # no backend_handle

        fake = MagicMock()
        fake.reconstruct_handle = MagicMock(
            return_value=BackendHandle(
                backend_name="kubernetes", job_id="nohandle1", scheduler_id="amortized-nohandle1"
            )
        )
        fake.status = AsyncMock(return_value=BackendStatus(running=True))
        with patch("amortized.worker.get_backend", return_value=fake):
            resumable = await _resume_inflight_jobs()

        assert "nohandle1" in [j["id"] for j in resumable]  # adopted, not requeued
        fake.reconstruct_handle.assert_called_once_with("nohandle1")
        async with get_pool().acquire() as conn:
            row = await Repository(conn).get_job("nohandle1")
        assert row is not None
        assert row["status"] == "provisioning"  # left for the poll loop to finalize
        # The reconstructed handle must be persisted, or _resume_job bails on the missing
        # handle and the live Job leaks while its GPU budget is released.
        assert row["backend_handle"]  # non-empty → the reconstructed handle was written
        assert row["k8s_job_name"] == "amortized-nohandle1"  # record matches a normal submit

    @pytest.mark.asyncio
    async def test_resume_requeues_no_handle_job_when_resource_absent(
        self, client: httpx.AsyncClient
    ) -> None:
        """No saved handle AND the reconstructed resource is confirmed gone → requeue."""
        import amortized.config as config_mod
        from amortized.backends import BackendHandle, BackendStatus
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _resume_inflight_jobs

        ns = config_mod.settings.compute_namespace
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            await repo.create_job(
                job_id="nohandle2", job_type=JobType.training, config={"algorithm": "sft"},
                created_at="2026-01-01T00:00:00+00:00", k8s_namespace=ns,
            )
            await repo.update_job("nohandle2", status="provisioning")

        fake = MagicMock()
        fake.reconstruct_handle = MagicMock(
            return_value=BackendHandle(
                backend_name="kubernetes", job_id="nohandle2", scheduler_id="amortized-nohandle2"
            )
        )
        fake.status = AsyncMock(return_value=BackendStatus(running=False, not_found=True))
        with patch("amortized.worker.get_backend", return_value=fake):
            await _resume_inflight_jobs()

        async with get_pool().acquire() as conn:
            row = await Repository(conn).get_job("nohandle2")
        assert row is not None
        assert row["status"] == "queued"  # confirmed gone → requeued

    @pytest.mark.asyncio
    async def test_resume_finalizes_running_job_that_finished_during_downtime(
        self, client: httpx.AsyncClient
    ) -> None:
        """A 'running' job that completed while the worker was down must be resumed so the
        poll loop records its real outcome — not marked failed as an orphan."""
        import amortized.config as config_mod
        from amortized.backends import BackendStatus
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _resume_inflight_jobs

        ns = config_mod.settings.compute_namespace
        handle = (
            '{"backend_name": "kubernetes", "job_id": "fin1", "scheduler_id": "amortized-fin1"}'
        )
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            await repo.create_job(
                job_id="fin1", job_type=JobType.training, config={"algorithm": "sft"},
                created_at="2026-01-01T00:00:00+00:00", k8s_namespace=ns,
            )
            await repo.update_job("fin1", status="running", backend_handle=handle)

        fake = MagicMock()
        fake.status = AsyncMock(return_value=BackendStatus(running=False, exit_code=0))
        with patch("amortized.worker.get_backend", return_value=fake):
            resumable = await _resume_inflight_jobs()

        assert "fin1" in [j["id"] for j in resumable]  # resumed to finalize, not failed
        async with get_pool().acquire() as conn:
            row = await Repository(conn).get_job("fin1")
        assert row is not None
        assert row["status"] == "running"  # untouched; the poll loop will record success

    @pytest.mark.asyncio
    async def test_resume_does_not_clobber_a_job_cancelled_mid_recovery(
        self, client: httpx.AsyncClient
    ) -> None:
        """If the API cancels a provisioning job while recovery is checking its backend
        status, the status-guarded update must not requeue the now-cancelled job."""
        import amortized.config as config_mod
        from amortized.backends import BackendStatus
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _resume_inflight_jobs

        ns = config_mod.settings.compute_namespace
        handle = (
            '{"backend_name": "kubernetes", "job_id": "race1",'
            ' "scheduler_id": "amortized-race1"}'
        )
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            await repo.create_job(
                job_id="race1", job_type=JobType.training, config={"algorithm": "sft"},
                created_at="2026-01-01T00:00:00+00:00", k8s_namespace=ns,
            )
            await repo.update_job("race1", status="provisioning", backend_handle=handle)

        async def _cancel_then_report_gone(_handle: object) -> BackendStatus:
            # The API cancels the job during the status check (the TOCTOU window).
            async with get_pool().acquire() as conn:
                await Repository(conn).update_job("race1", status="cancelled")
            return BackendStatus(running=False, not_found=True)

        fake = MagicMock()
        fake.status = AsyncMock(side_effect=_cancel_then_report_gone)
        with patch("amortized.worker.get_backend", return_value=fake):
            await _resume_inflight_jobs()

        async with get_pool().acquire() as conn:
            row = await Repository(conn).get_job("race1")
        assert row is not None
        assert row["status"] == "cancelled"  # guard prevented the requeue

    @pytest.mark.asyncio
    async def test_resume_skips_reconstructed_job_if_status_changed_before_persist(
        self, client: httpx.AsyncClient
    ) -> None:
        """A no-handle provisioning job cancelled during its status check must not be resumed:
        persisting the reconstructed handle is status-guarded, so it is skipped (not appended)."""
        import amortized.config as config_mod
        from amortized.backends import BackendHandle, BackendStatus
        from amortized.db.connection import get_pool
        from amortized.db.repository import Repository
        from amortized.models import JobType
        from amortized.worker import _resume_inflight_jobs

        ns = config_mod.settings.compute_namespace
        async with get_pool().acquire() as conn:
            repo = Repository(conn)
            await repo.create_job(
                job_id="race2", job_type=JobType.training, config={"algorithm": "sft"},
                created_at="2026-01-01T00:00:00+00:00", k8s_namespace=ns,
            )
            await repo.update_job("race2", status="provisioning")  # no handle

        async def _cancel_then_running(_handle: object) -> BackendStatus:
            async with get_pool().acquire() as conn:
                await Repository(conn).update_job("race2", status="cancelled")
            return BackendStatus(running=True)

        fake = MagicMock()
        fake.reconstruct_handle = MagicMock(
            return_value=BackendHandle(
                backend_name="kubernetes", job_id="race2", scheduler_id="amortized-race2"
            )
        )
        fake.status = AsyncMock(side_effect=_cancel_then_running)
        with patch("amortized.worker.get_backend", return_value=fake):
            resumable = await _resume_inflight_jobs()

        assert "race2" not in [j["id"] for j in resumable]  # not resumed
        async with get_pool().acquire() as conn:
            row = await Repository(conn).get_job("race2")
        assert row is not None
        assert row["status"] == "cancelled"
        assert not row["backend_handle"]  # guarded-out persist left no handle


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
        # thub sft takes max_tokens_per_gpu (like osft), not max_batch_len, and requires a
        # learning_rate; the caller-provided learning_rate is preserved.
        assert parsed["max_tokens_per_gpu"] == 4096
        assert parsed["learning_rate"] == 0.0002
        assert "max_batch_len" not in parsed
        assert "algorithm" not in parsed
        assert "micro_batch_size" not in parsed

    def test_thub_config_yaml_sft_defaults_match_osft(self) -> None:
        # sft previously failed on thub because it sent max_batch_len and omitted learning_rate.
        # It must now produce the same instructlab-training knobs as osft (max_tokens_per_gpu +
        # a default learning_rate), so a plain sft job (no explicit lr) runs on the thub sft path.
        from amortized.jobs.training import _training_hub_config_yaml

        base = {"model_name_or_path": "Qwen/Qwen3-0.6B", "data_path": "/d.jsonl"}
        sft = yaml.safe_load(_training_hub_config_yaml("sft", {**base, "algorithm": "sft"}))
        osft = yaml.safe_load(_training_hub_config_yaml("osft", {**base, "algorithm": "osft"}))
        assert sft["max_tokens_per_gpu"] == osft["max_tokens_per_gpu"] == 4096
        assert sft["learning_rate"] == osft["learning_rate"] == 2e-5
        assert "max_batch_len" not in sft

    def test_thub_config_yaml_lora_sft_no_packing_knobs(self) -> None:
        # thub lora-sft has no max_tokens_per_gpu/max_batch_len knobs and its own lr default.
        from amortized.jobs.training import _training_hub_config_yaml

        parsed = yaml.safe_load(_training_hub_config_yaml(
            "lora_sft", {"algorithm": "lora_sft", "model_name_or_path": "m", "data_path": "/d"}
        ))
        assert parsed["effective_batch_size"] == 8
        assert parsed["max_seq_len"] == 2048
        assert "max_batch_len" not in parsed
        assert "max_tokens_per_gpu" not in parsed

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

    def test_thub_config_yaml_embedding_sft(self) -> None:
        from amortized.jobs.training import _training_hub_config_yaml

        config = {
            "algorithm": "embedding_sft",
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "data_path": "/data/train.parquet",
            "num_train_epochs": 40,
            "per_device_train_batch_size": 32,
            "learning_rate": 2e-5,
            "loss_type": "batch_all_triplet",
            "batch_sampler": "group_by_label",
            "text_column": "text",
            "label_column": "label",
            "output_dir": "/output",
        }
        parsed = yaml.safe_load(_training_hub_config_yaml("embedding_sft", config))
        assert parsed["model_path"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert parsed["num_epochs"] == 40
        assert parsed["batch_size"] == 32  # per_device_train_batch_size → batch_size
        assert parsed["loss_type"] == "batch_all_triplet"
        assert parsed["batch_sampler"] == "group_by_label"
        assert parsed["ckpt_output_dir"] == "/output"
        # SFT/OSFT-only knobs must not leak into an embedding_sft config
        assert "micro_batch_size" not in parsed
        assert "max_seq_len" not in parsed
        assert "effective_batch_size" not in parsed
        assert "data_output_dir" not in parsed

    async def test_embedding_sft_aliases_map_to_subcommand(self) -> None:
        from amortized.jobs.training import build

        for alias in ("embedding_sft", "classifier", "embedding"):
            cfg = {
                "algorithm": alias,
                "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
                "data_path": "/data/train.parquet",
            }
            result = await build({"id": "job1234abcd", "config": cfg}, dict(cfg), {})
            assert result.command == ["thub", "embedding-sft", "--config", "/amortized/config.yaml"]


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
    async def test_data_run_id_chains_data_path(self) -> None:
        from amortized.worker import _resolve_parent_artifacts

        # No parent job — training references a dataset run directly
        # (uploaded dataset or materialized split).
        training_job = {"id": "training-1", "type": "training"}
        config: dict[str, object] = {
            "algorithm": "sft",
            "model_name_or_path": "test/model",
            "data_run_id": "dataset-run-xyz",
        }

        with patch("amortized.db.connection.get_pool") as mock_pool:
            mock_pool.assert_not_called()
            result_config, pre_commands = await _resolve_parent_artifacts(training_job, config)

        assert result_config["data_path"] == "/amortized/work/data/generated_data"
        assert len(pre_commands) == 1
        assert pre_commands[0] == (
            "mlflow artifacts download -r dataset-run-xyz -a generated_data -d /amortized/work/data"
        )

    @pytest.mark.asyncio
    async def test_explicit_data_path_wins_over_data_run_id(self) -> None:
        from amortized.worker import _resolve_parent_artifacts

        training_job = {"id": "training-1", "type": "training"}
        config: dict[str, object] = {
            "algorithm": "sft",
            "data_run_id": "dataset-run-xyz",
            "data_path": "/custom/data.jsonl",
        }

        result_config, pre_commands = await _resolve_parent_artifacts(training_job, config)
        assert result_config["data_path"] == "/custom/data.jsonl"
        assert pre_commands == []

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


class TestJobLoggingWrap:
    def test_noop_without_mlflow_run(self) -> None:
        from amortized.worker import _wrap_job_logging

        cmd = ["sh", "-c", "thub osft --config /amortized/config.yaml"]
        assert _wrap_job_logging(cmd, mlflow_run_created=False) == cmd

    def test_tees_and_uploads_always_preserving_exit(self) -> None:
        from amortized.worker import _wrap_job_logging

        cmd = ["sh", "-c", "thub osft --config /c.yaml && mlflow artifacts log-artifacts -a model"]
        result = _wrap_job_logging(cmd, mlflow_run_created=True)
        assert result[0:2] == ["sh", "-c"]
        shell = result[2]
        # inner chain preserved
        assert "thub osft --config /c.yaml" in shell
        # a per-job temp dir is used instead of a fixed /tmp path (no collisions,
        # not predictable)
        assert "mktemp -d" in shell
        assert "/tmp/amortized-job.log" not in shell
        # tee to the per-job file for persistence while keeping stdout for live logs
        assert 'tee "$_amz_log"' in shell
        # the log is uploaded to the run's logs/ artifact path
        assert 'mlflow artifacts log-artifact --local-file "$_amz_log"' in shell
        assert "--artifact-path logs" in shell
        # the temp dir is cleaned up after the upload
        assert 'rm -rf "$_amz_dir"' in shell
        # the real exit code is captured + re-raised (not tee's)
        assert 'exit "$rc"' in shell

    def test_reapplies_auth_pre_command_to_upload(self) -> None:
        from amortized.worker import _wrap_job_logging

        result = _wrap_job_logging(
            ["thub", "osft"], mlflow_run_created=True, auth_pre_command="export PYTHONPATH=/x"
        )
        # the auth export precedes the upload; the piped subshell drops the in-chain export
        assert "export PYTHONPATH=/x && mlflow artifacts log-artifact" in result[2]

    def test_upload_guarded_on_run_id(self) -> None:
        from amortized.worker import _wrap_job_logging

        result = _wrap_job_logging(["thub", "x"], mlflow_run_created=True)
        assert 'if [ -n "$MLFLOW_RUN_ID" ]; then' in result[2]


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
