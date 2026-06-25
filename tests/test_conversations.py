"""Tests for conversation list/detail endpoints."""

import os

import httpx
import pytest

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    """Use a temporary database for each test."""
    import amortized.config as config_mod
    import amortized.db as db_mod
    import amortized.db.connection as db_conn_mod

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


class TestConversationsList:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/agent/conversations")
        assert resp.status_code == 200
        assert resp.json() == []


class TestConversationDetail:
    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/agent/conversations/nonexistent-id")
        assert resp.status_code == 404
