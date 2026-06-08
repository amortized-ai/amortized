"""Tests for Settings API endpoints — API keys and backends."""

import os
from collections.abc import Iterator

import httpx
import pytest

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> Iterator[None]:
    import amortized.config as config_mod
    import amortized.db as db_mod
    import amortized.db.connection as db_conn_mod
    from amortized.backends.local import LocalBackend
    from amortized.core.compute import register_backend, reset

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_mod.settings = new_settings
    db_conn_mod.settings = new_settings

    reset()
    register_backend(LocalBackend())
    yield
    reset()


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


class TestApiKeys:
    @pytest.mark.asyncio
    async def test_add_api_key(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/settings/api-keys",
            json={"name": "My OpenAI", "provider": "openai", "key": "sk-abc123xyz"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My OpenAI"
        assert data["provider"] == "openai"
        assert data["key_preview"] == "...3xyz"
        assert "key" not in data
        assert "key_value" not in data
        assert data["id"]

    @pytest.mark.asyncio
    async def test_list_api_keys(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/settings/api-keys",
            json={"name": "Key A", "provider": "openai", "key": "sk-aaaa1111"},
        )
        await client.post(
            "/api/v1/settings/api-keys",
            json={"name": "Key B", "provider": "anthropic", "key": "sk-bbbb2222"},
        )
        resp = await client.get("/api/v1/settings/api-keys")
        assert resp.status_code == 200
        keys = resp.json()
        assert len(keys) == 2
        for k in keys:
            assert "key" not in k
            assert "key_value" not in k
            assert k["key_preview"].startswith("...")

    @pytest.mark.asyncio
    async def test_delete_api_key(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/settings/api-keys",
            json={"name": "To Delete", "provider": "google", "key": "goog-key-1234"},
        )
        key_id = resp.json()["id"]

        del_resp = await client.delete(f"/api/v1/settings/api-keys/{key_id}")
        assert del_resp.status_code == 204

        list_resp = await client.get("/api/v1/settings/api-keys")
        assert len(list_resp.json()) == 0

    @pytest.mark.asyncio
    async def test_delete_api_key_not_found(self, client: httpx.AsyncClient) -> None:
        resp = await client.delete("/api/v1/settings/api-keys/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_api_key_duplicate_provider(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/settings/api-keys",
            json={"name": "OpenAI v1", "provider": "openai", "key": "sk-first1111"},
        )
        resp = await client.post(
            "/api/v1/settings/api-keys",
            json={"name": "OpenAI v2", "provider": "openai", "key": "sk-second22"},
        )
        assert resp.status_code == 201
        list_resp = await client.get("/api/v1/settings/api-keys")
        keys = list_resp.json()
        assert len(keys) == 2
        providers = [k["provider"] for k in keys]
        assert providers.count("openai") == 2

    @pytest.mark.asyncio
    async def test_get_api_key_for_provider(self, client: httpx.AsyncClient) -> None:
        """Repository method returns the full key for internal provider lookup."""
        from amortized.db import get_db
        from amortized.db.repository import Repository

        await client.post(
            "/api/v1/settings/api-keys",
            json={"name": "Anthropic Key", "provider": "anthropic", "key": "sk-ant-secret"},
        )
        async for db in get_db():
            repo = Repository(db)
            row = await repo.get_api_key_for_provider("anthropic")
            assert row is not None
            assert row["key_value"] == "sk-ant-secret"
            break


class TestBackends:
    @pytest.mark.asyncio
    async def test_list_backends(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/settings/backends")
        assert resp.status_code == 200
        backends = resp.json()
        names = [b["name"] for b in backends]
        assert "local" in names

    @pytest.mark.asyncio
    async def test_add_backend(self, client: httpx.AsyncClient, tmp_path: object) -> None:
        resp = await client.post(
            "/api/v1/settings/backends",
            json={
                "name": "my-gpu-box",
                "type": "ssh",
                "host": "gpu.example.com",
                "user": "trainer",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my-gpu-box"
        assert "gpu" in data["capabilities"]

    @pytest.mark.asyncio
    async def test_delete_backend(self, client: httpx.AsyncClient) -> None:
        await client.post(
            "/api/v1/settings/backends",
            json={"name": "to-remove", "type": "ssh", "host": "rm.example.com"},
        )
        del_resp = await client.delete("/api/v1/settings/backends/to-remove")
        assert del_resp.status_code == 204

        list_resp = await client.get("/api/v1/settings/backends")
        names = [b["name"] for b in list_resp.json()]
        assert "to-remove" not in names

    @pytest.mark.asyncio
    async def test_delete_local_backend_rejected(self, client: httpx.AsyncClient) -> None:
        resp = await client.delete("/api/v1/settings/backends/local")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_delete_backend_not_found(self, client: httpx.AsyncClient) -> None:
        resp = await client.delete("/api/v1/settings/backends/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_test_backend(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/api/v1/settings/backends/local/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "local"

    @pytest.mark.asyncio
    async def test_test_backend_not_found(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/api/v1/settings/backends/nonexistent/test")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_unsupported_backend_type(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/settings/backends",
            json={"name": "bad", "type": "kubernetes", "host": "k8s.example.com"},
        )
        assert resp.status_code == 400
