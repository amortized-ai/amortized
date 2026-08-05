"""Shared test fixtures and markers."""

import pytest

import amortized.db.connection as db_conn_mod


@pytest.fixture(autouse=True)
async def _reset_db() -> None:
    """Close the connection pool between tests so each test gets a fresh one."""
    yield
    if db_conn_mod._pool is not None:
        await db_conn_mod._pool.close()
        db_conn_mod._pool = None
