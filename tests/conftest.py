"""Shared test fixtures and markers."""

import os
import subprocess

import pytest

import amortized.db.connection as db_conn_mod

TEST_DATABASE_URL = os.environ.get(
    "AMORTIZED_TEST_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized_test",
)


@pytest.fixture(autouse=True, scope="session")
def _run_migrations() -> None:
    """Run alembic migrations once per session."""
    env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
    subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        check=True,
        env=env,
    )


@pytest.fixture(autouse=True)
async def _reset_db() -> None:
    """Truncate jobs table and close the pool after each test."""
    try:
        import asyncpg

        conn = await asyncpg.connect(TEST_DATABASE_URL)
        await conn.execute("TRUNCATE jobs")
        await conn.close()
    except OSError:
        pass

    yield

    if db_conn_mod._pool is not None:
        await db_conn_mod._pool.close()
        db_conn_mod._pool = None
