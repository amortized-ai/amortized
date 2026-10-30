"""Tests for the prompt store (#541) — Repository CRUD, content_sha256 dedup, and
ownership stamping. No HTTP server required."""

import os
import subprocess

import asyncpg
import pytest
from conftest import TEST_DATABASE_URL

from amortized.db.repository import Repository


@pytest.fixture
async def repo():
    env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    await conn.execute("DROP TABLE IF EXISTS alembic_version")
    await conn.execute("DROP TABLE IF EXISTS prompts")
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.close()
    subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    yield Repository(conn)
    await conn.close()


async def _create(repo: Repository, prompt_id: str, content: str, **kw) -> dict:
    import hashlib

    return await repo.create_prompt(
        prompt_id=prompt_id,
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        created_at="2026-01-01T00:00:00+00:00",
        **kw,
    )


@pytest.mark.asyncio
async def test_create_and_get_prompt(repo: Repository) -> None:
    row = await _create(repo, "prm_1", "hello world", label="Assessor", role="system_prompt")
    assert row["id"] == "prm_1"
    assert row["content"] == "hello world"
    assert row["label"] == "Assessor"
    assert await repo.get_prompt_content("prm_1") == "hello world"
    fetched = await repo.get_prompt("prm_1")
    assert fetched is not None and fetched["content"] == "hello world"


@pytest.mark.asyncio
async def test_missing_prompt_is_none(repo: Repository) -> None:
    assert await repo.get_prompt_content("prm_nope") is None
    assert await repo.get_prompt("prm_nope") is None


@pytest.mark.asyncio
async def test_content_sha256_dedup_reuses_row(repo: Repository) -> None:
    first = await _create(repo, "prm_a", "identical text")
    # A second create of the SAME text gets a different id but dedups to the first row.
    second = await _create(repo, "prm_b", "identical text")
    assert second["id"] == first["id"] == "prm_a"
    # Distinct text creates a distinct row.
    other = await _create(repo, "prm_c", "different text")
    assert other["id"] == "prm_c"


@pytest.mark.asyncio
async def test_stamp_and_list_by_user(repo: Repository) -> None:
    await _create(repo, "prm_x", "x text")  # user_id='' (ownerless, agent path)
    await _create(repo, "prm_y", "y text", user_id="alice")
    # Stamp the ownerless row for alice; already-owned rows are untouched.
    await repo.stamp_prompts_user(["prm_x", "prm_y"], "alice")
    owned = {r["id"] for r in await repo.list_prompts("alice")}
    assert owned == {"prm_x", "prm_y"}
    # Stamp is a claim: it does not steal a row already owned by someone else.
    await _create(repo, "prm_z", "z text", user_id="bob")
    await repo.stamp_prompts_user(["prm_z"], "alice")
    assert (await repo.get_prompt("prm_z"))["user_id"] == "bob"
