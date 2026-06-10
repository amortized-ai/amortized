"""Tests for the Judge API endpoints."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from amortized.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_list_templates_returns_list() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/judge/templates")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.anyio
async def test_judge_without_asynth_returns_501() -> None:
    with patch.dict(sys.modules, {"asynth": None}):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/judge",
                json={
                    "template": "safety",
                    "data": [{"response": "hello"}],
                    "model": "openai/gpt-4o-mini",
                },
            )
    assert resp.status_code == 501


@pytest.mark.anyio
async def test_judge_with_mocked_asynth() -> None:
    mock_judge_instance = MagicMock()
    mock_judge_instance.judge = MagicMock(return_value=[{"passed": True, "score": 0.9}])

    mock_create_judge = MagicMock(return_value=mock_judge_instance)
    mock_judge_config = MagicMock()
    mock_inference_config = MagicMock()

    mock_asynth = MagicMock()
    mock_asynth.JudgeConfig = mock_judge_config
    mock_asynth.LiteLLMInferenceConfig = mock_inference_config
    mock_asynth.create_judge = mock_create_judge

    with patch.dict(sys.modules, {"asynth": mock_asynth}):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/judge",
                json={
                    "template": "safety",
                    "data": [{"response": "hello"}],
                    "model": "openai/gpt-4o-mini",
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["total"] == 1
    assert body["summary"]["passed"] == 1
    assert body["summary"]["pass_rate"] == 1.0
