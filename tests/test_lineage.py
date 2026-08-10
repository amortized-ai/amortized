"""Tests for job lineage graph traversal."""

import os
import subprocess

import asyncpg
import pytest
from conftest import TEST_DATABASE_URL

from amortized.core.jobs import create_job
from amortized.core.lineage import get_job_lineage, list_lineage_chains
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


class TestListLineageChains:
    @pytest.mark.asyncio
    async def test_empty_when_no_jobs(self, repo: Repository) -> None:
        chains = await list_lineage_chains(repo)
        assert chains == []

    @pytest.mark.asyncio
    async def test_excludes_single_orphan_jobs(self, repo: Repository) -> None:
        await create_job(repo, job_type=JobType.sdg, config={"topic": "solo"})
        chains = await list_lineage_chains(repo)
        assert chains == []

    @pytest.mark.asyncio
    async def test_returns_chain_with_two_jobs(self, repo: Repository) -> None:
        parent = await create_job(repo, job_type=JobType.sdg, config={"topic": "chained"})
        await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "test-model",
                "data_path": "d.jsonl",
            },
            parent_job_id=parent["id"],
        )

        chains = await list_lineage_chains(repo)
        assert len(chains) == 1
        assert chains[0].chain_id == parent["id"]
        assert chains[0].job_count == 2
        assert len(chains[0].lineage.nodes) == 2
        assert len(chains[0].lineage.edges) == 1

    @pytest.mark.asyncio
    async def test_chain_name_from_recipe(self, repo: Repository) -> None:
        parent = await create_job(repo, job_type=JobType.sdg, config={}, recipe="customer-support")
        await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "m",
                "data_path": "d.jsonl",
            },
            parent_job_id=parent["id"],
        )

        chains = await list_lineage_chains(repo)
        assert chains[0].name == "customer-support"

    @pytest.mark.asyncio
    async def test_chain_name_fallback_to_type_and_id(self, repo: Repository) -> None:
        parent = await create_job(repo, job_type=JobType.sdg, config={"topic": "t"})
        await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "m",
                "data_path": "d.jsonl",
            },
            parent_job_id=parent["id"],
        )

        chains = await list_lineage_chains(repo)
        assert chains[0].name == f"sdg {parent['id'][:8]}"

    @pytest.mark.asyncio
    async def test_filter_by_type(self, repo: Repository) -> None:
        p1 = await create_job(repo, job_type=JobType.sdg, config={})
        await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "m",
                "data_path": "d.jsonl",
            },
            parent_job_id=p1["id"],
        )

        p2 = await create_job(repo, job_type=JobType.upload, config={})
        await create_job(
            repo,
            job_type=JobType.sdg,
            config={"topic": "t"},
            parent_job_id=p2["id"],
        )

        chains = await list_lineage_chains(repo, job_type="training")
        assert len(chains) == 1
        assert chains[0].chain_id == p1["id"]

    @pytest.mark.asyncio
    async def test_filter_by_status(self, repo: Repository) -> None:
        p = await create_job(repo, job_type=JobType.sdg, config={})
        child = await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "m",
                "data_path": "d.jsonl",
            },
            parent_job_id=p["id"],
        )
        await repo.update_job(child["id"], status=JobStatus.succeeded.value)

        chains_succeeded = await list_lineage_chains(repo, status="succeeded")
        assert len(chains_succeeded) == 1

        chains_failed = await list_lineage_chains(repo, status="failed")
        assert len(chains_failed) == 0

    @pytest.mark.asyncio
    async def test_sorted_newest_first(self, repo: Repository) -> None:
        import asyncio

        p1 = await create_job(repo, job_type=JobType.sdg, config={})
        await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "m",
                "data_path": "d.jsonl",
            },
            parent_job_id=p1["id"],
        )

        await asyncio.sleep(0.01)

        p2 = await create_job(repo, job_type=JobType.upload, config={})
        await create_job(
            repo,
            job_type=JobType.sdg,
            config={"topic": "t"},
            parent_job_id=p2["id"],
        )

        chains = await list_lineage_chains(repo)
        assert len(chains) == 2
        assert chains[0].chain_id == p2["id"]
        assert chains[1].chain_id == p1["id"]

    @pytest.mark.asyncio
    async def test_multiple_independent_chains(self, repo: Repository) -> None:
        p1 = await create_job(repo, job_type=JobType.sdg, config={})
        await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "m",
                "data_path": "d.jsonl",
            },
            parent_job_id=p1["id"],
        )

        p2 = await create_job(repo, job_type=JobType.upload, config={})
        await create_job(
            repo,
            job_type=JobType.sdg,
            config={},
            parent_job_id=p2["id"],
        )

        # orphan — should NOT appear
        await create_job(repo, job_type=JobType.sdg, config={})

        chains = await list_lineage_chains(repo)
        assert len(chains) == 2

    @pytest.mark.asyncio
    async def test_three_job_chain(self, repo: Repository) -> None:
        upload = await create_job(
            repo, job_type=JobType.upload, config={"original_filename": "doc.pdf"}
        )
        sdg = await create_job(
            repo,
            job_type=JobType.sdg,
            config={"topic": "support"},
            parent_job_id=upload["id"],
        )
        await create_job(
            repo,
            job_type=JobType.training,
            config={
                "algorithm": "sft",
                "model_name_or_path": "m",
                "data_path": "d.jsonl",
            },
            parent_job_id=sdg["id"],
        )

        chains = await list_lineage_chains(repo)
        assert len(chains) == 1
        assert chains[0].chain_id == upload["id"]
        assert chains[0].job_count == 3
        assert len(chains[0].lineage.edges) == 2
