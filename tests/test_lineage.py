"""Tests for job lineage graph traversal."""

import os
import subprocess

import asyncpg
import pytest
from conftest import TEST_DATABASE_URL

from amortized.core.jobs import create_job
from amortized.core.lineage import get_job_lineage
from amortized.db.repository import Repository
from amortized.models import JobType


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


class TestGetJobLineage:
    @pytest.mark.asyncio
    async def test_single_job_no_parents_no_children(self, repo: Repository) -> None:
        job = await create_job(
            repo,
            job_type=JobType.sdg,
            config={"topic": "testing", "num_records": 50},
        )
        result = await get_job_lineage(repo, job["id"])
        assert result is not None
        assert len(result.nodes) == 1
        assert result.nodes[0].id == job["id"]
        assert result.edges == []
        assert result.root_id == job["id"]
        assert result.target_id == job["id"]

    @pytest.mark.asyncio
    async def test_nonexistent_job_returns_none(self, repo: Repository) -> None:
        result = await get_job_lineage(repo, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_parent_child_chain(self, repo: Repository) -> None:
        upload = await create_job(
            repo,
            job_type=JobType.upload,
            config={"original_filename": "doc.pdf"},
        )
        sdg = await create_job(
            repo,
            job_type=JobType.sdg,
            config={"topic": "support", "num_records": 100},
            parent_job_id=upload["id"],
        )
        training = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "Qwen/Qwen3.5-1.5B",
                "data_path": "data.jsonl",
            },
            parent_job_id=sdg["id"],
        )

        result = await get_job_lineage(repo, sdg["id"])
        assert result is not None
        assert len(result.nodes) == 3
        assert result.root_id == upload["id"]
        assert result.target_id == sdg["id"]

        node_ids = {n.id for n in result.nodes}
        assert node_ids == {upload["id"], sdg["id"], training["id"]}

        edge_pairs = {(e.source, e.target) for e in result.edges}
        assert (upload["id"], sdg["id"]) in edge_pairs
        assert (sdg["id"], training["id"]) in edge_pairs

    @pytest.mark.asyncio
    async def test_walks_up_from_leaf(self, repo: Repository) -> None:
        parent = await create_job(
            repo,
            job_type=JobType.sdg,
            config={"topic": "qa"},
        )
        child = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "lora_sft",
                "model_name_or_path": "test-model",
                "data_path": "data.jsonl",
            },
            parent_job_id=parent["id"],
        )

        result = await get_job_lineage(repo, child["id"])
        assert result is not None
        assert result.root_id == parent["id"]
        assert result.target_id == child["id"]
        assert len(result.nodes) == 2

    @pytest.mark.asyncio
    async def test_walks_down_from_root(self, repo: Repository) -> None:
        root = await create_job(
            repo,
            job_type=JobType.upload,
            config={"original_filename": "data.pdf"},
        )
        child = await create_job(
            repo,
            job_type=JobType.sdg,
            config={"topic": "extract"},
            parent_job_id=root["id"],
        )

        result = await get_job_lineage(repo, root["id"])
        assert result is not None
        assert result.root_id == root["id"]
        assert len(result.nodes) == 2
        node_ids = {n.id for n in result.nodes}
        assert child["id"] in node_ids

    @pytest.mark.asyncio
    async def test_meta_extraction_training(self, repo: Repository) -> None:
        job = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "grpo",
                "model_name_or_path": "Qwen/Qwen3.5-1.5B",
                "data_path": "data.jsonl",
            },
        )
        result = await get_job_lineage(repo, job["id"])
        assert result is not None
        node = result.nodes[0]
        assert node.meta["model"] == "Qwen/Qwen3.5-1.5B"
        assert node.meta["algorithm"] == "grpo"

    @pytest.mark.asyncio
    async def test_meta_extraction_sdg(self, repo: Repository) -> None:
        job = await create_job(
            repo,
            job_type=JobType.sdg,
            config={
                "topic": "customer support",
                "num_records": 500,
                "model_configs": [{"model": "gpt-4o", "alias": "default"}],
            },
        )
        result = await get_job_lineage(repo, job["id"])
        assert result is not None
        node = result.nodes[0]
        assert node.meta["topic"] == "customer support"
        assert node.meta["num_records"] == 500
        assert node.meta["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_meta_extraction_upload(self, repo: Repository) -> None:
        job = await create_job(
            repo,
            job_type=JobType.upload,
            config={"original_filename": "guide.pdf", "source": "s3://bucket/key"},
        )
        result = await get_job_lineage(repo, job["id"])
        assert result is not None
        node = result.nodes[0]
        assert node.meta["filename"] == "guide.pdf"
        assert node.meta["source"] == "s3://bucket/key"

    @pytest.mark.asyncio
    async def test_meta_extraction_eval(self, repo: Repository) -> None:
        job = await create_job(
            repo,
            job_type=JobType.eval,
            config={"model_name_or_path": "Qwen/Qwen3.5-1.5B"},
        )
        result = await get_job_lineage(repo, job["id"])
        assert result is not None
        node = result.nodes[0]
        assert node.meta["model"] == "Qwen/Qwen3.5-1.5B"

    @pytest.mark.asyncio
    async def test_multiple_children(self, repo: Repository) -> None:
        parent = await create_job(
            repo,
            job_type=JobType.sdg,
            config={"topic": "qa"},
        )
        child1 = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "model-a",
                "data_path": "data.jsonl",
            },
            parent_job_id=parent["id"],
        )
        child2 = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "lora_sft",
                "model_name_or_path": "model-b",
                "data_path": "data.jsonl",
            },
            parent_job_id=parent["id"],
        )

        result = await get_job_lineage(repo, parent["id"])
        assert result is not None
        assert len(result.nodes) == 3
        node_ids = {n.id for n in result.nodes}
        assert node_ids == {parent["id"], child1["id"], child2["id"]}
        assert len(result.edges) == 2


class TestGetChildrenByParentId:
    @pytest.mark.asyncio
    async def test_returns_children(self, repo: Repository) -> None:
        parent = await create_job(
            repo,
            job_type=JobType.sdg,
            config={},
        )
        child = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "t",
                "data_path": "t.jsonl",
            },
            parent_job_id=parent["id"],
        )

        children = await repo.get_children_by_parent_id(parent["id"])
        assert len(children) == 1
        assert children[0]["id"] == child["id"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_children(self, repo: Repository) -> None:
        children = await repo.get_children_by_parent_id("nonexistent")
        assert children == []
