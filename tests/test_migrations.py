"""Tests for alembic migrations."""

import os
import subprocess

import asyncpg
import pytest
from conftest import TEST_DATABASE_URL


@pytest.fixture(autouse=True)
async def _clean_db():
    """Drop all tables before each test for a clean slate."""
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    await conn.execute("DROP TABLE IF EXISTS alembic_version")
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.close()
    yield
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    await conn.execute("DROP TABLE IF EXISTS alembic_version")
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.close()


class TestAlembicMigrations:
    def test_upgrade_creates_schema(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"alembic upgrade failed: {result.stderr}"

    @pytest.mark.asyncio
    async def test_upgrade_creates_jobs_table(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)

        conn = await asyncpg.connect(TEST_DATABASE_URL)
        row = await conn.fetchrow(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'jobs'"
        )
        await conn.close()
        assert row is not None

    @pytest.mark.asyncio
    async def test_downgrade_drops_jobs_table(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)
        subprocess.run(["alembic", "downgrade", "base"], capture_output=True, env=env, check=True)

        conn = await asyncpg.connect(TEST_DATABASE_URL)
        row = await conn.fetchrow(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'jobs'"
        )
        await conn.close()
        assert row is None

    def test_current_shows_head_after_upgrade(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)
        result = subprocess.run(["alembic", "current"], capture_output=True, text=True, env=env)
        assert "0001" in result.stdout

    def test_upgrade_is_idempotent(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        result1 = subprocess.run(
            ["alembic", "upgrade", "head"], capture_output=True, text=True, env=env
        )
        assert result1.returncode == 0
        result2 = subprocess.run(
            ["alembic", "upgrade", "head"], capture_output=True, text=True, env=env
        )
        assert result2.returncode == 0
