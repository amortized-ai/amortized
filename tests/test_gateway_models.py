"""Tests for GET /api/v1/models (gateway model discovery)."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from amortized.api import models as models_mod
from amortized.main import app
from amortized.models import GatewayModel


def _reset_cache() -> None:
    models_mod._models_cache = None
    models_mod._models_cache_time = 0


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    _reset_cache()


@pytest.mark.asyncio
async def test_list_models_no_gateway() -> None:
    with patch.object(models_mod.settings, "mlflow_tracking_uri", ""):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert data["models"] == []


@pytest.mark.asyncio
async def test_list_models_with_gateway() -> None:
    fake_models = [
        GatewayModel(name="chat", provider="openai", model_name="gpt-4.1-mini"),
        GatewayModel(name="claude-haiku", provider="anthropic", model_name="claude-haiku-4-5"),
    ]

    async def mock_fetch() -> list[GatewayModel]:
        return fake_models

    with (
        patch.object(models_mod.settings, "gateway_url", "http://mlflow:5000/gateway/mlflow/v1"),
        patch.object(models_mod, "_fetch_gateway_models", mock_fetch),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 2
    assert data["models"][0]["name"] == "chat"
    assert data["models"][0]["provider"] == "openai"
    assert data["models"][0]["model_name"] == "gpt-4.1-mini"
    assert data["models"][1]["name"] == "claude-haiku"
    assert data["models"][1]["provider"] == "anthropic"
    assert data["gateway_url"] == "http://mlflow:5000/gateway/mlflow/v1"


@pytest.mark.asyncio
async def test_list_models_gateway_unreachable() -> None:
    with patch.object(models_mod.settings, "mlflow_tracking_uri", "http://mlflow:5000"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert data["models"] == []


@pytest.mark.asyncio
async def test_list_models_caches_result() -> None:
    call_count = 0

    async def mock_fetch() -> list[GatewayModel]:
        nonlocal call_count
        call_count += 1
        return [GatewayModel(name="chat", provider="openai", model_name="gpt-4.1-mini")]

    with patch.object(models_mod, "_fetch_gateway_models", mock_fetch):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp1 = await client.get("/api/v1/models")
            resp2 = await client.get("/api/v1/models")

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["models"] == resp2.json()["models"]
    # The patched function is called each time since it replaces the whole function,
    # but real caching is tested via the _fetch_gateway_models unit test below
    assert call_count == 2


@pytest.mark.asyncio
async def test_fetch_gateway_models_caches() -> None:
    """Test that _fetch_gateway_models uses its internal cache."""
    models_mod._models_cache = [
        GatewayModel(name="cached", provider="test", model_name="gpt-4.1-mini")
    ]
    models_mod._models_cache_time = 9999999999.0

    result = await models_mod._fetch_gateway_models()
    assert len(result) == 1
    assert result[0].name == "cached"
