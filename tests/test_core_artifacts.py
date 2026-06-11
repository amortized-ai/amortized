"""Tests for core/artifacts.py — no HTTP server required."""

import aiosqlite
import pytest

from amortized.core.artifacts import (
    list_artifacts,
    register_artifacts_for_job,
    register_log_artifacts,
)
from amortized.db.connection import _SCHEMA_PATH
from amortized.db.repository import Repository
from amortized.models import JobType


@pytest.fixture
async def repo(tmp_path):
    db_path = tmp_path / "test.db"
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    await db.executescript(_SCHEMA_PATH.read_text())
    await db.commit()
    repo = Repository(db)
    await repo.create_job(
        job_id="j1",
        job_type=JobType.training,
        config={},
        created_at="2026-01-01T00:00:00",
    )
    yield repo
    await db.close()


class TestRegisterArtifacts:
    @pytest.mark.asyncio
    async def test_register_training_artifacts(self, repo: Repository, tmp_path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "adapter_config.json").write_text("{}")
        (output_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 100)
        (output_dir / "training_metrics.jsonl").write_text('{"step":1}\n')
        (output_dir / "tokenizer.json").write_text("{}")

        registered = await register_artifacts_for_job(
            repo, "j1", str(output_dir), job_type="training"
        )
        assert len(registered) == 1
        assert registered[0]["artifact_type"] == "model"
        assert registered[0]["name"] == "model"
        assert registered[0]["path"] == str(output_dir)

    @pytest.mark.asyncio
    async def test_register_nonexistent_dir(self, repo: Repository) -> None:
        result = await register_artifacts_for_job(repo, "j1", "/nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_register_checkpoint_subdirs_non_training(
        self, repo: Repository, tmp_path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "generated_data.jsonl").write_text('{"a":1}\n')
        ckpt = output_dir / "checkpoints"
        ckpt.mkdir()
        (ckpt / "batch_0.jsonl").write_text('{"b":1}\n')

        registered = await register_artifacts_for_job(repo, "j1", str(output_dir))
        types = [a["artifact_type"] for a in registered]
        assert "generated_data" in types
        assert "checkpoint" in types

    @pytest.mark.asyncio
    async def test_register_sdg_artifacts(self, repo: Repository, tmp_path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "generated_data.jsonl").write_text('{"messages":[]}\n')
        (output_dir / "stats.json").write_text('{"num_samples":10}\n')

        registered = await register_artifacts_for_job(repo, "j1", str(output_dir))
        types = {a["artifact_type"] for a in registered}
        names = {a["name"] for a in registered}
        assert "generated_data" in types
        assert "sdg_stats" in types
        assert "generated_data.jsonl" in names
        assert "stats.json" in names
        assert len(registered) == 2

    @pytest.mark.asyncio
    async def test_register_sdg_checkpoints(self, repo: Repository, tmp_path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_0100.jsonl").write_text('{"data": "test"}\n')

        registered = await register_artifacts_for_job(repo, "j1", str(output_dir))
        types = [a["artifact_type"] for a in registered]
        assert "checkpoint" in types


class TestRegisterLogArtifacts:
    @pytest.mark.asyncio
    async def test_register_logs(self, repo: Repository, tmp_path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "stdout.log").write_text("some output\n")
        (output_dir / "stderr.log").write_text("some error\n")

        registered = await register_log_artifacts(repo, "j1", str(output_dir))
        assert len(registered) == 2
        assert all(a["artifact_type"] == "log" for a in registered)

    @pytest.mark.asyncio
    async def test_skip_empty_logs(self, repo: Repository, tmp_path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "stdout.log").write_text("")

        registered = await register_log_artifacts(repo, "j1", str(output_dir))
        assert registered == []


class TestListArtifacts:
    @pytest.mark.asyncio
    async def test_list_empty(self, repo: Repository) -> None:
        result = await list_artifacts(repo, "j1")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_after_register(self, repo: Repository, tmp_path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "adapter_config.json").write_text("{}")

        await register_artifacts_for_job(repo, "j1", str(output_dir))
        result = await list_artifacts(repo, "j1")
        assert len(result) == 1
