"""Tests for the fastmcp MCP server scaffold."""

import httpx
import pytest

from amortized.main import app
from amortized.mcp.server import _call, init_mcp_client, mcp


@pytest.mark.asyncio
async def test_mcp_endpoint_mounted() -> None:
    """The /mcp endpoint is mounted and reachable."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/mcp")
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_mcp_does_not_break_health() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_mcp_server_name() -> None:
    assert mcp.name == "amortized"


@pytest.mark.asyncio
async def test_mcp_no_tools_registered() -> None:
    """Scaffold exposes zero tools — tools are added in later issues."""
    tools = await mcp.list_tools()
    assert tools == []


@pytest.mark.asyncio
async def test_call_helper_success() -> None:
    """_call can reach FastAPI endpoints via ASGI transport."""
    init_mcp_client(app)
    try:
        result = await _call("GET", "/api/v1/health")
        assert result["status"] == "ok"
    finally:
        from amortized.mcp import server

        server._client = None


@pytest.mark.asyncio
async def test_call_helper_404_raises() -> None:
    """_call raises ValueError on 404."""
    init_mcp_client(app)
    try:
        with pytest.raises(ValueError, match="http_404"):
            await _call("GET", "/api/v1/nonexistent")
    finally:
        from amortized.mcp import server

        server._client = None


@pytest.mark.asyncio
async def test_call_helper_uninitialised() -> None:
    """_call raises RuntimeError when client is not initialised."""
    from amortized.mcp import server

    saved = server._client
    server._client = None
    try:
        with pytest.raises(RuntimeError, match="not initialised"):
            await _call("GET", "/api/v1/health")
    finally:
        server._client = saved
