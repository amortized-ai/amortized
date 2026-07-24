"""Shared test fixtures and markers."""

import pytest


@pytest.fixture(autouse=True)
async def _reset_shared_db() -> None:
    """Close the shared DB connection between tests so each test gets a fresh one."""
    import amortized.db.connection as db_conn_mod

    yield
    if db_conn_mod._shared_db is not None:
        await db_conn_mod._shared_db.close()
        db_conn_mod._shared_db = None
