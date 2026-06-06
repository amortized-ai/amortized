"""Tests for core/jobs.py — no HTTP server required."""

import aiosqlite
import pytest

from amortized.core.jobs import (
    InvalidJobStateError,
    JobNotFoundError,
    cancel_job,
    create_job,
    get_job,
    list_jobs,
)
from amortized.db.connection import _SCHEMA_PATH
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType


@pytest.fixture
async def repo(tmp_path):
    db_path = tmp_path / "test.db"
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    await db.executescript(_SCHEMA_PATH.read_text())
    await db.commit()
    repo = Repository(db)
    yield repo
    await db.close()


class TestCreateJob:
    @pytest.mark.asyncio
    async def test_create_training_job(self, repo: Repository) -> None:
        row = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "lora_sft",
                "model_path": "test",
                "data_path": "test.jsonl",
                "ckpt_output_dir": "/tmp/out",
            },
            output_dir="/tmp/out",
        )
        assert row["type"] == "training"
        assert row["status"] == "queued"
        assert row["config"]["model_path"] == "test"
        assert row["output_dir"] == "/tmp/out"
        assert row["id"]

    @pytest.mark.asyncio
    async def test_create_sdg_job(self, repo: Repository) -> None:
        row = await create_job(
            repo,
            job_type=JobType.sdg,
            config={
                "pipeline": "conversation",
                "model": "openai/gpt-4o",
            },
        )
        assert row["type"] == "sdg"

    @pytest.mark.asyncio
    async def test_create_emits_event(self, repo: Repository) -> None:
        row = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "lora_sft",
                "model_path": "test",
                "data_path": "test.jsonl",
                "ckpt_output_dir": "/tmp/out",
            },
        )
        events = await repo.list_events(row["id"])
        statuses = [e["data"]["status"] for e in events]
        assert "queued" in statuses


class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_existing(self, repo: Repository) -> None:
        created = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "lora_sft",
                "model_path": "t",
                "data_path": "t.jsonl",
                "ckpt_output_dir": "/tmp/t",
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
                "algorithm": "lora_sft",
                "model_path": "t",
                "data_path": "t.jsonl",
                "ckpt_output_dir": "/tmp/t",
            },
        )
        await create_job(
            repo,
            job_type=JobType.sdg,
            config={"pipeline": "conversation", "model": "openai/gpt-4o"},
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
                "algorithm": "lora_sft",
                "model_path": "t",
                "data_path": "t.jsonl",
                "ckpt_output_dir": "/tmp/t",
            },
        )
        cancelled = await cancel_job(repo, created["id"])
        assert cancelled["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_emits_event(self, repo: Repository) -> None:
        created = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "lora_sft",
                "model_path": "t",
                "data_path": "t.jsonl",
                "ckpt_output_dir": "/tmp/t",
            },
        )
        await cancel_job(repo, created["id"])
        events = await repo.list_events(created["id"])
        types = [e["data"]["status"] for e in events]
        assert "cancelled" in types

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
                "algorithm": "lora_sft",
                "model_path": "t",
                "data_path": "t.jsonl",
                "ckpt_output_dir": "/tmp/t",
            },
        )
        await repo.update_job_status(
            created["id"],
            status=JobStatus.succeeded,
            updated_at="2026-01-01T00:01:00",
        )
        with pytest.raises(InvalidJobStateError):
            await cancel_job(repo, created["id"])

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_is_idempotent(self, repo: Repository) -> None:
        created = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "lora_sft",
                "model_path": "t",
                "data_path": "t.jsonl",
                "ckpt_output_dir": "/tmp/t",
            },
        )
        await cancel_job(repo, created["id"])
        result = await cancel_job(repo, created["id"])
        assert result["status"] == "cancelled"
