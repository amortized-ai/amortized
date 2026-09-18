"""Metric-set persistence must restore soft-deleted dataset runs."""
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_restores_deleted_run_before_tagging():
    from amortized.api.jobs import _persist_metric_set

    config = {
        "eval_data_run_id": "run-1",
        "metrics": [],
        "rubric": [{"name": "a", "description": "d"}],
    }
    run = {"info": {"run_id": "run-1", "lifecycle_stage": "deleted"}}

    client = AsyncMock()
    client.get_run = AsyncMock(return_value=run)
    client.restore_run = AsyncMock()

    with patch("amortized.core.mlflow_client.MLflowClient", return_value=client), \
         patch("amortized.jobs.common.set_mlflow_run_tag", new_callable=AsyncMock) as tag, \
         patch("amortized.config.settings.mlflow_tracking_uri", "http://mlflow"):
        await _persist_metric_set(config, "parent-1", db=None)

    client.restore_run.assert_awaited_once_with("run-1")
    tag.assert_awaited_once()
    args = tag.await_args.args
    assert args[0] == "run-1" and args[1] == "eval_metric_set"
    assert "a" in args[2]


@pytest.mark.asyncio
async def test_skips_restore_for_active_run():
    from amortized.api.jobs import _persist_metric_set

    config = {"eval_data_run_id": "run-2", "metrics": ["m"], "rubric": []}
    run = {"info": {"run_id": "run-2", "lifecycle_stage": "active"}}
    client = AsyncMock()
    client.get_run = AsyncMock(return_value=run)

    async def _fail_restore(run_id):
        raise AssertionError("restore should not be called for active runs")

    client.restore_run = _fail_restore

    with patch("amortized.core.mlflow_client.MLflowClient", return_value=client), \
         patch("amortized.jobs.common.set_mlflow_run_tag", new_callable=AsyncMock) as tag, \
         patch("amortized.config.settings.mlflow_tracking_uri", "http://mlflow"):
        await _persist_metric_set(config, "", db=None)

    tag.assert_awaited_once()
