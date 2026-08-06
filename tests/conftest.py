"""Shared test fixtures and markers."""

import os

import pytest

import amortized.db.connection as db_conn_mod

TEST_DATABASE_URL = os.environ.get(
    "AMORTIZED_TEST_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized_test",
)


@pytest.fixture(autouse=True)
async def _reset_db() -> None:
    """Close the connection pool between tests so each test gets a fresh one."""
    yield
    if db_conn_mod._pool is not None:
        await db_conn_mod._pool.close()
        db_conn_mod._pool = None
