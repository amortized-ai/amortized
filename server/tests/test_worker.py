"""Tests for the background worker job execution lifecycle."""

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


class TestWorkerJobExecution:
    """Test the worker picks up and executes jobs."""

    @pytest.mark.asyncio
    async def test_training_job_lifecycle(self, client: httpx.AsyncClient) -> None:
        """Create a training job, run it through the worker, verify status transitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a training job
            response = await client.post(
                "/api/v1/jobs/training",
                json={
                    "model_path": "test/model",
                    "data_path": "./data.jsonl",
                    "ckpt_output_dir": tmpdir,
                },
            )
            assert response.status_code == 201
            job_id = response.json()["id"]

            # Verify initial status is pending
            response = await client.get(f"/api/v1/jobs/{job_id}")
            assert response.json()["status"] == "pending"

            # Run the job through the worker
            from amortized.worker import _pick_pending_job, _run_job

            job = await _pick_pending_job()
            assert job is not None
            assert job["id"] == job_id

            await _run_job(job)

            # Verify job completed
            response = await client.get(f"/api/v1/jobs/{job_id}")
            data = response.json()
            assert data["status"] == "completed"
            assert data["started_at"] is not None
            assert data["completed_at"] is not None

            # Verify artifacts were registered
            response = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
            artifacts = response.json()
            assert len(artifacts) > 0
            artifact_types = [a["artifact_type"] for a in artifacts]
            assert "training_metrics" in artifact_types
            assert "adapter_config" in artifact_types

            # Verify metrics file was created (under job_id subdirectory)
            metrics_path = os.path.join(tmpdir, job_id, "training_metrics.jsonl")
            assert os.path.exists(metrics_path)
            with open(metrics_path) as f:
                lines = f.readlines()
            assert len(lines) > 0
            first_metric = json.loads(lines[0])
            assert "step" in first_metric
            assert "loss" in first_metric

    @pytest.mark.asyncio
    async def test_sdg_job_lifecycle(self, client: httpx.AsyncClient) -> None:
        """Create an SDG job, run it through the worker, verify completion."""
        with tempfile.TemporaryDirectory():
            # Create an SDG job
            response = await client.post(
                "/api/v1/jobs/sdg",
                json={
                    "flow_id": "knowledge-qa",
                    "dataset_path": "./docs.jsonl",
                    "model": "openai/gpt-4o",
                },
            )
            assert response.status_code == 201
            job_id = response.json()["id"]

            # Run through worker
            from amortized.worker import _pick_pending_job, _run_job

            job = await _pick_pending_job()
            assert job is not None
            await _run_job(job)

            # Verify completed
            response = await client.get(f"/api/v1/jobs/{job_id}")
            data = response.json()
            assert data["status"] == "completed"

            # SDG jobs should have artifacts too
            response = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
            artifacts = response.json()
            assert len(artifacts) > 0

    @pytest.mark.asyncio
    async def test_worker_picks_oldest_job_first(self, client: httpx.AsyncClient) -> None:
        """Worker should pick the oldest pending job first (FIFO)."""
        # Create two jobs
        resp1 = await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test/first",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "/tmp/test-first",
            },
        )
        await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test/second",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "/tmp/test-second",
            },
        )
        first_id = resp1.json()["id"]

        from amortized.worker import _pick_pending_job

        job = await _pick_pending_job()
        assert job is not None
        assert job["id"] == first_id

    @pytest.mark.asyncio
    async def test_no_pending_jobs_returns_none(self, client: httpx.AsyncClient) -> None:
        """Worker returns None when no jobs are pending."""
        from amortized.worker import _pick_pending_job

        job = await _pick_pending_job()
        assert job is None


class TestOrphanedJobCleanup:
    """Test orphaned job detection on startup."""

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_jobs(self, client: httpx.AsyncClient) -> None:
        """Jobs with status=running but dead PID should be marked failed."""
        import aiosqlite

        from amortized.config import settings
        from amortized.worker import cleanup_orphaned_jobs

        # Create a job and manually set it to running with a dead PID
        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test/model",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "/tmp/test-orphan",
            },
        )
        job_id = response.json()["id"]

        # Manually update to running with a PID that doesn't exist
        async with aiosqlite.connect(str(settings.db_path)) as db:
            await db.execute(
                "UPDATE jobs SET status = ?, pid = ? WHERE id = ?",
                ("running", 999999, job_id),
            )
            await db.commit()

        # Run cleanup
        await cleanup_orphaned_jobs()

        # Verify it was marked as failed
        response = await client.get(f"/api/v1/jobs/{job_id}")
        data = response.json()
        assert data["status"] == "failed"
        assert "Orphaned" in (data.get("error") or "")

    @pytest.mark.asyncio
    async def test_cleanup_readopts_live_pid(self, client: httpx.AsyncClient) -> None:
        """Jobs with status=running and a live PID should be re-adopted, not failed."""
        import subprocess

        import aiosqlite

        from amortized.config import settings
        from amortized.worker import cleanup_orphaned_jobs

        # Create a job
        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test/model",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "/tmp/test-readopt",
            },
        )
        job_id = response.json()["id"]

        # Start a real long-running process we can use as a live PID
        proc = subprocess.Popen(
            ["sleep", "300"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        live_pid = proc.pid

        try:
            # Set job to running with the live PID
            async with aiosqlite.connect(str(settings.db_path)) as db:
                await db.execute(
                    "UPDATE jobs SET status = ?, pid = ?, output_dir = ? WHERE id = ?",
                    ("running", live_pid, "/tmp/test-readopt", job_id),
                )
                await db.commit()

            # Run cleanup — should re-adopt, not mark as failed
            await cleanup_orphaned_jobs()

            # Verify job is still running (not failed)
            response = await client.get(f"/api/v1/jobs/{job_id}")
            data = response.json()
            assert data["status"] == "running", (
                f"Expected 'running' but got '{data['status']}' — "
                "live process should be re-adopted, not marked failed"
            )
        finally:
            proc.terminate()
            proc.wait()

    @pytest.mark.asyncio
    async def test_cleanup_finds_pid_none_dead(self, client: httpx.AsyncClient) -> None:
        """Jobs with pid=None and no matching /proc entry should be marked failed."""
        import aiosqlite

        from amortized.config import settings
        from amortized.worker import cleanup_orphaned_jobs

        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test/model",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "/tmp/test-no-pid",
            },
        )
        job_id = response.json()["id"]

        # Set to running with no PID
        async with aiosqlite.connect(str(settings.db_path)) as db:
            await db.execute(
                "UPDATE jobs SET status = ?, pid = NULL WHERE id = ?",
                ("running", job_id),
            )
            await db.commit()

        await cleanup_orphaned_jobs()

        response = await client.get(f"/api/v1/jobs/{job_id}")
        data = response.json()
        assert data["status"] == "failed"
        assert "Orphaned" in (data.get("error") or "")


class TestCancelRunningJob:
    """Test enhanced cancel behavior with subprocess kill."""

    @pytest.mark.asyncio
    async def test_cancel_running_job_with_pid(self, client: httpx.AsyncClient) -> None:
        """Cancel should attempt to kill the subprocess."""
        import aiosqlite

        from amortized.config import settings

        # Create and start a job
        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test/model",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "/tmp/test-cancel",
            },
        )
        job_id = response.json()["id"]

        # Manually set to running with a dead PID (so kill won't error dangerously)
        async with aiosqlite.connect(str(settings.db_path)) as db:
            await db.execute(
                "UPDATE jobs SET status = ?, pid = ? WHERE id = ?",
                ("running", 999999, job_id),
            )
            await db.commit()

        # Cancel the job
        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_completed_job_rejected(self, client: httpx.AsyncClient) -> None:
        """Cannot cancel a completed job."""
        import aiosqlite

        from amortized.config import settings

        response = await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test/model",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "/tmp/test-done",
            },
        )
        job_id = response.json()["id"]

        # Manually set to completed
        async with aiosqlite.connect(str(settings.db_path)) as db:
            await db.execute(
                "UPDATE jobs SET status = ? WHERE id = ?",
                ("completed", job_id),
            )
            await db.commit()

        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 400
