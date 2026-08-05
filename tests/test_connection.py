"""Tests for database connection resilience."""

import os
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

import amortized.db.connection as db_conn
from amortized.config import Settings

TEST_DATABASE_URL = os.environ.get(
    "AMORTIZED_TEST_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized_test",
)


@pytest.fixture(autouse=True)
async def _cleanup():
    """Ensure pool is cleaned up between tests."""
    yield
    if db_conn._pool is not None:
        await db_conn._pool.close()
        db_conn._pool = None


class TestCheckDbHealth:
    @pytest.mark.asyncio
    async def test_returns_true_when_pool_alive(self) -> None:
        db_conn._pool = await asyncpg.create_pool(dsn=TEST_DATABASE_URL)
        assert await db_conn.check_db_health() is True

    @pytest.mark.asyncio
    async def test_returns_false_when_pool_is_none(self) -> None:
        db_conn._pool = None
        assert await db_conn.check_db_health() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_closed_pool(self) -> None:
        pool = await asyncpg.create_pool(dsn=TEST_DATABASE_URL)
        await pool.close()
        db_conn._pool = pool
        assert await db_conn.check_db_health() is False
        db_conn._pool = None


class TestInitDbResilience:
    @pytest.mark.asyncio
    async def test_retries_on_connection_failure(self) -> None:
        real_pool = await asyncpg.create_pool(dsn=TEST_DATABASE_URL)

        call_count = 0

        async def flaky_create_pool(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Connection refused")
            return real_pool

        os.environ["AMORTIZED_DATABASE_URL"] = TEST_DATABASE_URL
        db_conn.settings = Settings()

        with (
            patch("amortized.db.connection.asyncpg.create_pool", side_effect=flaky_create_pool),
            patch("amortized.db.connection.asyncio.sleep", new_callable=AsyncMock),
        ):
            await db_conn.init_db()

        assert db_conn._pool is real_pool
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        os.environ["AMORTIZED_DATABASE_URL"] = TEST_DATABASE_URL
        db_conn.settings = Settings()

        with (
            patch(
                "amortized.db.connection.asyncpg.create_pool",
                side_effect=OSError("Connection refused"),
            ),
            patch("amortized.db.connection.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="Failed to connect"),
        ):
            await db_conn.init_db()
