"""Smoke test for the health endpoint."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    with patch("amortized.db.check_db_health", new_callable=AsyncMock, return_value=True):
        from amortized.main import health

        result = await health()
    assert result["status"] == "ok"
    assert result["db"] == "ok"
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_health_degraded_when_db_unreachable() -> None:
    with patch("amortized.db.check_db_health", new_callable=AsyncMock, return_value=False):
        from amortized.main import health

        result = await health()
    assert result["status"] == "degraded"
    assert result["db"] == "unreachable"
