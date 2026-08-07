"""Shared test fixtures and markers."""

import os
import subprocess

import asyncpg
import pytest

import amortized.db.connection as db_conn_mod

TEST_DATABASE_URL = os.environ.get(
    "AMORTIZED_TEST_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized_test",
)

_ALEMBIC_ENV = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}


@pytest.fixture(autouse=True, scope="session")
def _run_migrations() -> None:
    """Run alembic migrations once per session."""
    subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        check=True,
        env=_ALEMBIC_ENV,
    )


@pytest.fixture(autouse=True)
async def _reset_db() -> None:
    """Truncate jobs table and close the pool after each test."""
    try:
        conn = await asyncpg.connect(TEST_DATABASE_URL)
        try:
            await conn.execute("TRUNCATE jobs")
        finally:
            await conn.close()
    except asyncpg.UndefinedTableError:
        # Migration tests may have dropped the schema; restore it
        subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            check=True,
            env=_ALEMBIC_ENV,
        )
    except OSError:
        warnings.warn(
            "Could not connect to test database — TRUNCATE skipped, tests may lack isolation",
            stacklevel=1,
        )

    yield

    if db_conn_mod._pool is not None:
        await db_conn_mod._pool.close()
        db_conn_mod._pool = None
