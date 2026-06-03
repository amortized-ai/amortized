"""Tests for the SynthEngine batched generation loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from amortized_synth.engine import SynthEngine
from amortized_synth.inference import InferenceClient, ModelConfig
from amortized_synth.pipelines.conversation import ConversationPipeline


@pytest.fixture
def mock_client() -> InferenceClient:
    config = ModelConfig(model="test/model")
    client = InferenceClient(config)
    client.complete = AsyncMock(return_value="Test response")  # type: ignore[method-assign]
    client.complete_batch = AsyncMock(  # type: ignore[method-assign]
        return_value=["Test response", "Test response 2"]
    )
    return client


class TestSynthEngine:
    @pytest.mark.asyncio
    async def test_basic_run(self, mock_client: InferenceClient) -> None:
        pipeline = ConversationPipeline()
        engine = SynthEngine(mock_client, pipeline)
        seeds = [{"topic": "python"}, {"topic": "rust"}]

        result = await engine.run(seeds, max_turns=3)

        assert result.stats.total_requested == 2
        assert result.stats.total_completed == 2
        assert result.stats.total_failed == 0
        assert len(result.conversations) == 2
        for conv in result.conversations:
            assert conv.status == "completed"
            assert len(conv.turns) > 0

    @pytest.mark.asyncio
    async def test_handles_failures(self, mock_client: InferenceClient) -> None:
        call_count = 0

        async def batch_with_one_failure(
            batches: list[list[dict[str, Any]]],
        ) -> list[str | None]:
            nonlocal call_count
            call_count += 1
            results: list[str | None] = []
            for i, _ in enumerate(batches):
                if call_count == 1 and i == 0:
                    results.append(None)
                else:
                    results.append("Success")
            return results

        mock_client.complete_batch = batch_with_one_failure  # type: ignore[assignment]
        pipeline = ConversationPipeline()
        engine = SynthEngine(mock_client, pipeline)
        seeds = [{"topic": "fail"}, {"topic": "succeed"}]

        result = await engine.run(seeds, max_turns=2, max_retries=1)

        assert result.stats.total_failed == 1
        assert result.stats.total_completed == 1

    @pytest.mark.asyncio
    async def test_progress_callback(self, mock_client: InferenceClient) -> None:
        pipeline = ConversationPipeline()
        engine = SynthEngine(mock_client, pipeline)
        seeds = [{"topic": "test"}]
        progress_calls: list[tuple[int, int]] = []

        def on_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        mock_client.complete_batch = AsyncMock(  # type: ignore[method-assign]
            return_value=["response"]
        )
        await engine.run(seeds, max_turns=2, on_progress=on_progress)

        assert len(progress_calls) > 0
        for _completed, total in progress_calls:
            assert total == 1

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_load(
        self, mock_client: InferenceClient, tmp_path: Any
    ) -> None:
        pipeline = ConversationPipeline()
        engine = SynthEngine(mock_client, pipeline)
        seeds = [{"topic": "checkpoint_test"}]

        mock_client.complete_batch = AsyncMock(  # type: ignore[method-assign]
            return_value=["response"]
        )

        result = await engine.run(
            seeds, max_turns=2, checkpoint_dir=tmp_path, checkpoint_interval=1
        )
        assert (tmp_path / "state.json").exists()
        assert result.stats.total_completed == 1

    @pytest.mark.asyncio
    async def test_empty_seeds(self, mock_client: InferenceClient) -> None:
        pipeline = ConversationPipeline()
        engine = SynthEngine(mock_client, pipeline)

        result = await engine.run([], max_turns=3)

        assert result.stats.total_requested == 0
        assert result.stats.total_completed == 0
        assert len(result.conversations) == 0

    @pytest.mark.asyncio
    async def test_max_turns_respected(self, mock_client: InferenceClient) -> None:
        pipeline = ConversationPipeline()
        engine = SynthEngine(mock_client, pipeline)
        seeds = [{"topic": "test"}]

        mock_client.complete_batch = AsyncMock(  # type: ignore[method-assign]
            return_value=["response"]
        )
        result = await engine.run(seeds, max_turns=4)

        for conv in result.conversations:
            assert len(conv.turns) <= 4
