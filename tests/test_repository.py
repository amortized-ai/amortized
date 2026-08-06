"""Tests for the Repository CRUD class — no HTTP server required."""

from pathlib import Path

import asyncpg
import pytest
from conftest import TEST_DATABASE_URL

from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType


@pytest.fixture
async def repo():
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    schema_path = Path(__file__).parent.parent / "src" / "amortized" / "db" / "schema.sql"
    schema_sql = schema_path.read_text()
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.execute(schema_sql)
    yield Repository(conn)
    await conn.close()


class TestJobCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get_job(self, repo: Repository) -> None:
        row = await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={"model_name_or_path": "test"},
            created_at="2026-01-01T00:00:00",
        )
        assert row["id"] == "j1"
        assert row["type"] == "training"
        assert row["status"] == "queued"
        assert row["config"]["model_name_or_path"] == "test"

        fetched = await repo.get_job("j1")
        assert fetched is not None
        assert fetched["id"] == "j1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, repo: Repository) -> None:
        assert await repo.get_job("nope") is None

    @pytest.mark.asyncio
    async def test_list_jobs_with_filters(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        await repo.create_job(
            job_id="j2",
            job_type=JobType.sdg,
            config={},
            created_at="2026-01-01T00:00:01",
        )

        all_jobs = await repo.list_jobs()
        assert len(all_jobs) == 2

        training_only = await repo.list_jobs(job_type=JobType.training)
        assert len(training_only) == 1
        assert training_only[0]["type"] == "training"

    @pytest.mark.asyncio
    async def test_update_job(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        updated = await repo.update_job(
            "j1",
            status=JobStatus.running.value,
            started_at="2026-01-01T00:01:00",
        )
        assert updated is not None
        assert updated["status"] == "running"
        assert updated["started_at"] == "2026-01-01T00:01:00"

    @pytest.mark.asyncio
    async def test_error_field_null_not_string_none(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        job = await repo.get_job("j1")
        assert job is not None
        assert job["error"] is None

    @pytest.mark.asyncio
    async def test_error_string_none_sanitized(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        await repo.conn.execute("UPDATE jobs SET error = 'None' WHERE id = 'j1'")
        job = await repo.get_job("j1")
        assert job is not None
        assert job["error"] is None

    @pytest.mark.asyncio
    async def test_create_with_recipe_and_parent(self, repo: Repository) -> None:
        row = await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
            recipe="models/qwen-1.5b",
            parent_job_id="parent-1",
        )
        assert row["recipe"] == "models/qwen-1.5b"
        assert row["parent_job_id"] == "parent-1"

    @pytest.mark.asyncio
    async def test_pick_pending_job(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        await repo.create_job(
            job_id="j2",
            job_type=JobType.sdg,
            config={},
            created_at="2026-01-01T00:00:01",
        )
        job = await repo.pick_pending_job()
        assert job is not None
        assert job["id"] == "j1"

    @pytest.mark.asyncio
    async def test_pick_pending_no_jobs(self, repo: Repository) -> None:
        assert await repo.pick_pending_job() is None

    @pytest.mark.asyncio
    async def test_update_mlflow_run_id(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="j1",
            job_type=JobType.training,
            config={},
            created_at="2026-01-01T00:00:00",
        )
        updated = await repo.update_job("j1", mlflow_run_id="abc123def456")
        assert updated is not None
        assert updated["mlflow_run_id"] == "abc123def456"

    @pytest.mark.asyncio
    async def test_delete_existing_job(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="del1",
            job_type=JobType.training,
            config={"algorithm": "sft"},
            created_at="2024-01-01T00:00:00",
        )
        assert await repo.delete_job("del1") is True
        assert await repo.get_job("del1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_job(self, repo: Repository) -> None:
        assert await repo.delete_job("nonexistent") is False

    @pytest.mark.asyncio
    async def test_list_jobs_with_both_filters(self, repo: Repository) -> None:
        await repo.create_job(
            job_id="f1",
            job_type=JobType.training,
            config={},
            created_at="2024-01-01T00:00:00",
        )
        await repo.create_job(
            job_id="f2",
            job_type=JobType.sdg,
            config={},
            created_at="2024-01-02T00:00:00",
        )
        jobs = await repo.list_jobs(status=JobStatus.queued, job_type=JobType.training)
        assert len(jobs) == 1
        assert jobs[0]["id"] == "f1"
