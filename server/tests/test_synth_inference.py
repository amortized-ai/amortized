"""Tests for the inference client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from amortized_synth.inference import InferenceClient, ModelConfig


class TestInferenceClient:
    def test_init(self) -> None:
        config = ModelConfig(model="test/model", max_concurrent=4)
        client = InferenceClient(config)
        assert client.config.model == "test/model"
        assert client.config.max_concurrent == 4
        assert client.total_tokens == 0

    @pytest.mark.asyncio
    async def test_complete_batch_with_failures(self) -> None:
        config = ModelConfig(model="test/model")
        client = InferenceClient(config)

        async def mock_complete(msgs: list[dict[str, str]]) -> str:
            if "fail" in msgs[0].get("content", ""):
                raise Exception("API error")
            return "success"

        client.complete = mock_complete  # type: ignore[assignment]
        results = await client.complete_batch([
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "fail"}],
            [{"role": "user", "content": "world"}],
        ])

        assert results[0] == "success"
        assert results[1] is None
        assert results[2] == "success"

    def test_model_config_defaults(self) -> None:
        config = ModelConfig(model="openai/gpt-4o")
        assert config.api_base is None
        assert config.api_key is None
        assert config.max_concurrent == 16
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    @pytest.mark.asyncio
    async def test_complete_calls_litellm(self) -> None:
        config = ModelConfig(model="test/model", api_base="http://localhost:8000")
        client = InferenceClient(config)

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Hello world"
        mock_response.usage = AsyncMock()
        mock_response.usage.total_tokens = 42

        with patch("amortized_synth.inference.litellm.acompletion", return_value=mock_response):
            result = await client.complete([{"role": "user", "content": "Hi"}])

        assert result == "Hello world"
        assert client.total_tokens == 42
