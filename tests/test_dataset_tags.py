"""Tests for dataset source tags and topic generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from amortized.api.datasets import _topic_from_filename


class TestTopicFromFilename:
    def test_simple_jsonl(self) -> None:
        assert _topic_from_filename("training_data.jsonl") == "training data"

    def test_underscores_replaced(self) -> None:
        assert _topic_from_filename("my_dataset_v2.parquet") == "my dataset v2"

    def test_hyphens_replaced(self) -> None:
        assert _topic_from_filename("test-upload.jsonl") == "test upload"

    def test_mixed_separators(self) -> None:
        assert _topic_from_filename("customer_support-tickets.jsonl") == "customer support tickets"

    def test_no_extension(self) -> None:
        assert _topic_from_filename("README") == "README"

    def test_hidden_file(self) -> None:
        assert _topic_from_filename(".gitignore") == ""

    def test_empty_string(self) -> None:
        assert _topic_from_filename("") == ""

    def test_whitespace_collapsed(self) -> None:
        assert _topic_from_filename("a__b--c.jsonl") == "a b c"

    def test_multi_dot_filename(self) -> None:
        assert _topic_from_filename("data.v2.jsonl") == "data.v2"


class TestSdgSourceTag:
    @pytest.mark.asyncio
    @patch("amortized.jobs.sdg.set_mlflow_run_tag", new_callable=AsyncMock)
    async def test_on_success_sets_source_sdg(self, mock_tag: AsyncMock) -> None:
        from amortized.jobs.sdg import on_success

        job = {"config": {"num_records": 50, "topic": "test topic"}, "id": "abc12345"}
        await on_success(job, "run-xyz")

        mock_tag.assert_any_call("run-xyz", "source", "sdg")

    @pytest.mark.asyncio
    @patch("amortized.jobs.sdg.set_mlflow_run_tag", new_callable=AsyncMock)
    async def test_on_success_sets_topic(self, mock_tag: AsyncMock) -> None:
        from amortized.jobs.sdg import on_success

        job = {"config": {"topic": "sentiment analysis"}, "id": "abc12345"}
        await on_success(job, "run-xyz")

        mock_tag.assert_any_call("run-xyz", "dataset_topic", "sentiment analysis")

    @pytest.mark.asyncio
    @patch("amortized.jobs.sdg.set_mlflow_run_tag", new_callable=AsyncMock)
    async def test_on_success_skips_empty_topic(self, mock_tag: AsyncMock) -> None:
        from amortized.jobs.sdg import on_success

        job = {"config": {"topic": ""}, "id": "abc12345"}
        await on_success(job, "run-xyz")

        tag_calls = [c.args for c in mock_tag.call_args_list]
        assert ("run-xyz", "dataset_topic", "") not in tag_calls
