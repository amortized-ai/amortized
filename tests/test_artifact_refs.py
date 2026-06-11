"""Tests for artifact reference resolution."""

import aiosqlite
import pytest

from amortized.core.artifacts import resolve_artifact_refs
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
        job_id="job-abc",
        job_type=JobType.training,
        config={},
        created_at="2026-01-01T00:00:00",
    )
    await repo.create_artifact(
        artifact_id="a1",
        job_id="job-abc",
        artifact_type="adapter_weights",
        path="/outputs/job-abc/adapter_model.safetensors",
        size=1024,
        created_at="2026-01-01T00:00:00",
        name="adapter_model.safetensors",
        location="/outputs/job-abc/adapter_model.safetensors",
    )
    yield repo
    await db.close()


class TestResolveArtifactRefs:
    @pytest.mark.asyncio
    async def test_resolves_simple_ref(self, repo: Repository) -> None:
        config = {"model_name_or_path": "artifact:job-abc/adapter_model.safetensors"}
        resolved = await resolve_artifact_refs(repo, config)
        assert resolved["model_name_or_path"] == "/outputs/job-abc/adapter_model.safetensors"

    @pytest.mark.asyncio
    async def test_passes_through_non_refs(self, repo: Repository) -> None:
        config = {"model_name_or_path": "Qwen/Qwen2.5-1.5B", "num_train_epochs": 3}
        resolved = await resolve_artifact_refs(repo, config)
        assert resolved == config

    @pytest.mark.asyncio
    async def test_resolves_nested_dicts(self, repo: Repository) -> None:
        config = {
            "outer": {
                "inner": "artifact:job-abc/adapter_model.safetensors",
                "plain": "value",
            }
        }
        resolved = await resolve_artifact_refs(repo, config)
        assert resolved["outer"]["inner"] == "/outputs/job-abc/adapter_model.safetensors"
        assert resolved["outer"]["plain"] == "value"

    @pytest.mark.asyncio
    async def test_missing_artifact_raises(self, repo: Repository) -> None:
        config = {"path": "artifact:job-abc/nonexistent.bin"}
        with pytest.raises(ValueError, match="Artifact not found"):
            await resolve_artifact_refs(repo, config)

    @pytest.mark.asyncio
    async def test_invalid_ref_format_raises(self, repo: Repository) -> None:
        config = {"path": "artifact:no-slash-here"}
        with pytest.raises(ValueError, match="Invalid artifact reference"):
            await resolve_artifact_refs(repo, config)

    @pytest.mark.asyncio
    async def test_missing_job_raises(self, repo: Repository) -> None:
        config = {"path": "artifact:nonexistent-job/file.bin"}
        with pytest.raises(ValueError, match="Artifact not found"):
            await resolve_artifact_refs(repo, config)
