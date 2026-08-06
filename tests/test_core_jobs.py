"""Tests for core/jobs.py — no HTTP server required."""

import os
import subprocess

import asyncpg
import pytest
from conftest import TEST_DATABASE_URL

from amortized.core.jobs import (
    InvalidJobStateError,
    JobNotFoundError,
    cancel_job,
    create_job,
    get_job,
    list_jobs,
)
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType


@pytest.fixture
async def repo():
    env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    await conn.execute("DROP TABLE IF EXISTS alembic_version")
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.close()
    subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    yield Repository(conn)
    await conn.close()


class TestCreateJob:
    @pytest.mark.asyncio
    async def test_create_training_job(self, repo: Repository) -> None:
        row = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test.jsonl",
            },
        )
        assert row["type"] == "training"
        assert row["status"] == "queued"
        assert row["config"]["model_name_or_path"] == "test"
        assert row["id"]

    @pytest.mark.asyncio
    async def test_create_sdg_job(self, repo: Repository) -> None:
        row = await create_job(
            repo,
            job_type=JobType.sdg,
            config={"model": "openai/gpt-4o"},
        )
        assert row["type"] == "sdg"

    @pytest.mark.asyncio
    async def test_create_with_recipe(self, repo: Repository) -> None:
        row = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test.jsonl",
            },
            recipe="models/qwen-1.5b-lora",
        )
        assert row["recipe"] == "models/qwen-1.5b-lora"

    @pytest.mark.asyncio
    async def test_create_with_parent_job_id(self, repo: Repository) -> None:
        row = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "test.jsonl",
            },
            parent_job_id="parent-123",
        )
        assert row["parent_job_id"] == "parent-123"


class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_existing(self, repo: Repository) -> None:
        created = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "t",
                "data_path": "t.jsonl",
            },
        )
        fetched = await get_job(repo, created["id"])
        assert fetched is not None
        assert fetched["id"] == created["id"]

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, repo: Repository) -> None:
        assert await get_job(repo, "nope") is None


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_empty(self, repo: Repository) -> None:
        assert await list_jobs(repo) == []

    @pytest.mark.asyncio
    async def test_list_with_type_filter(self, repo: Repository) -> None:
        await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "t",
                "data_path": "t.jsonl",
            },
        )
        await create_job(
            repo,
            job_type=JobType.sdg,
            config={"model": "openai/gpt-4o"},
        )

        training = await list_jobs(repo, job_type=JobType.training)
        assert len(training) == 1
        assert training[0]["type"] == "training"


class TestCancelJob:
    @pytest.mark.asyncio
    async def test_cancel_queued(self, repo: Repository) -> None:
        created = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "t",
                "data_path": "t.jsonl",
            },
        )
        cancelled = await cancel_job(repo, created["id"])
        assert cancelled["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_raises(self, repo: Repository) -> None:
        with pytest.raises(JobNotFoundError):
            await cancel_job(repo, "nope")

    @pytest.mark.asyncio
    async def test_cancel_succeeded_raises(self, repo: Repository) -> None:
        created = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "t",
                "data_path": "t.jsonl",
            },
        )
        await repo.update_job(created["id"], status=JobStatus.succeeded.value)
        with pytest.raises(InvalidJobStateError):
            await cancel_job(repo, created["id"])

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_is_idempotent(self, repo: Repository) -> None:
        created = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "t",
                "data_path": "t.jsonl",
            },
        )
        await cancel_job(repo, created["id"])
        result = await cancel_job(repo, created["id"])
        assert result["status"] == "cancelled"
