"""Tests for conversation CRUD via the agent chat API."""

import os

import httpx
import pytest

from amortized_runtime.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    """Use a temporary database for each test."""
    import amortized_runtime.config as config_mod
    import amortized_runtime.db as db_mod

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized_runtime.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_creates_conversation(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/agent/chat",
            json={"message": "Hello!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert "message" in data
        assert len(data["message"]) > 0

    @pytest.mark.asyncio
    async def test_chat_with_existing_conversation(
        self, client: httpx.AsyncClient
    ) -> None:
        # Create conversation
        resp1 = await client.post(
            "/api/v1/agent/chat",
            json={"message": "Hello!"},
        )
        conv_id = resp1.json()["conversation_id"]

        # Continue conversation
        resp2 = await client.post(
            "/api/v1/agent/chat",
            json={"message": "I want to fine-tune a model", "conversation_id": conv_id},
        )
        assert resp2.status_code == 200
        assert resp2.json()["conversation_id"] == conv_id

    @pytest.mark.asyncio
    async def test_chat_returns_suggested_action(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post(
            "/api/v1/agent/chat",
            json={"message": "I want to train a model"},
        )
        data = resp.json()
        assert data["suggested_action"] is not None
        assert data["suggested_action"]["type"] == "create_training_job"

    @pytest.mark.asyncio
    async def test_chat_empty_message_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post(
            "/api/v1/agent/chat",
            json={"message": ""},
        )
        assert resp.status_code == 422


class TestConversationsList:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/agent/conversations")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_after_chat(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/agent/chat",
            json={"message": "Hello!"},
        )
        resp = await client.get("/api/v1/agent/conversations")
        assert resp.status_code == 200
        convs = resp.json()
        assert len(convs) == 1
        assert convs[0]["title"] == "Hello!"

    @pytest.mark.asyncio
    async def test_list_multiple_conversations(
        self, client: httpx.AsyncClient
    ) -> None:
        await client.post(
            "/api/v1/agent/chat", json={"message": "First conversation"}
        )
        await client.post(
            "/api/v1/agent/chat", json={"message": "Second conversation"}
        )
        resp = await client.get("/api/v1/agent/conversations")
        convs = resp.json()
        assert len(convs) == 2


class TestConversationDetail:
    @pytest.mark.asyncio
    async def test_get_conversation_with_messages(
        self, client: httpx.AsyncClient
    ) -> None:
        # Create a conversation with messages
        chat_resp = await client.post(
            "/api/v1/agent/chat",
            json={"message": "Hello!"},
        )
        conv_id = chat_resp.json()["conversation_id"]

        resp = await client.get(f"/api/v1/agent/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == conv_id
        # Should have user message + assistant response
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/agent/conversations/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_conversation_accumulates_messages(
        self, client: httpx.AsyncClient
    ) -> None:
        resp1 = await client.post(
            "/api/v1/agent/chat", json={"message": "Hello!"}
        )
        conv_id = resp1.json()["conversation_id"]

        await client.post(
            "/api/v1/agent/chat",
            json={"message": "What can you do?", "conversation_id": conv_id},
        )

        resp = await client.get(f"/api/v1/agent/conversations/{conv_id}")
        data = resp.json()
        # 2 user messages + 2 assistant responses = 4
        assert len(data["messages"]) == 4
