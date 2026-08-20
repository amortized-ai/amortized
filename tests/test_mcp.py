"""Tests for MCP auto-generation from OpenAPI."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from amortized.main import app


@pytest.mark.asyncio
async def test_mcp_endpoint_mounted() -> None:
    """The /mcp endpoint is mounted and reachable (406 = needs SSE negotiation)."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/mcp")
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_mcp_does_not_break_health() -> None:
    with patch("amortized.db.check_db_health", new_callable=AsyncMock, return_value=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
