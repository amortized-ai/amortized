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
        # Dynamically find the latest migration revision
        versions_dir = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "src",
            "amortized",
            "db",
            "migrations",
            "versions",
        )
        revisions = sorted(
            f for f in os.listdir(versions_dir) if f.endswith(".py") and not f.startswith("_")
        )
        latest_rev = revisions[-1].split("_")[0]  # e.g. "0002" from "0002_proper_pg_types.py"
        assert latest_rev in result.stdout

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
