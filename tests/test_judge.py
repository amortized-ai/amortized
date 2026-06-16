"""Tests for the Judge API endpoints."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from amortized.core.judge_templates import translate_template_to_judge_config
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
    mock_judge_config_cls = MagicMock()
    mock_judge_config_cls.from_dict = MagicMock(return_value=MagicMock())
    mock_inference_config = MagicMock()

    mock_asynth = MagicMock()
    mock_asynth.JudgeConfig = mock_judge_config_cls
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
    mock_judge_config_cls.from_dict.assert_called_once()
    call_arg = mock_judge_config_cls.from_dict.call_args[0][0]
    assert "judge_params" in call_arg
    assert call_arg["judge_params"]["judgment_type"] == "bool"


def test_translate_llm_template() -> None:
    template = {
        "type": "eval",
        "description": "test",
        "config": {
            "judge": {"model": "openai/gpt-5.4", "prompt": "Evaluate: {response}"},
            "system_instruction": "You are a judge.",
            "judgment_type": "bool",
            "response_format": "json",
            "include_explanation": True,
            "temperature": 0.0,
        },
    }
    config_dict, defaults = translate_template_to_judge_config(template)
    assert config_dict == {
        "judge_params": {
            "prompt_template": "Evaluate: {response}",
            "system_instruction": "You are a judge.",
            "judgment_type": "bool",
            "response_format": "json",
            "include_explanation": True,
        },
    }
    assert defaults == {"model": "openai/gpt-5.4", "temperature": 0.0}


def test_translate_rule_based_template() -> None:
    template = {
        "judge_params": {"prompt_template": "{response}"},
        "rule_judge_params": {
            "rule_type": "regex",
            "input_fields": ["response"],
            "rule_config": {"pattern": "\\d+"},
            "response_format": "xml",
            "judgment_type": "bool",
        },
    }
    config_dict, defaults = translate_template_to_judge_config(template)
    assert config_dict is template
    assert defaults == {}
