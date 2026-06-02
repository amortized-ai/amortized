"""Smoke test for the health endpoint."""

import httpx
import pytest

from amortized_runtime.main import app


@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
