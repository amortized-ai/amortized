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
    client._url = lambda path: "http://mlflow" + path

    restore_resp = AsyncMock()
    restore_resp.raise_for_status = lambda: None

    class FakeHttp:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            assert url.endswith("/api/2.0/mlflow/runs/restore")
            assert json == {"run_id": "run-1"}
            return restore_resp

    with patch("amortized.core.mlflow_client.MLflowClient", return_value=client), \
         patch("amortized.api.jobs.httpx.AsyncClient", return_value=FakeHttp()), \
         patch("amortized.jobs.common.set_mlflow_run_tag", new_callable=AsyncMock) as tag, \
         patch("amortized.config.settings.mlflow_tracking_uri", "http://mlflow"):
        await _persist_metric_set(config, "parent-1", db=None)

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

    class FailHttp:
        def __init__(self, *a, **k):
            raise AssertionError("restore should not be called for active runs")

    with patch("amortized.core.mlflow_client.MLflowClient", return_value=client), \
         patch("amortized.api.jobs.httpx.AsyncClient", side_effect=FailHttp), \
         patch("amortized.jobs.common.set_mlflow_run_tag", new_callable=AsyncMock) as tag, \
         patch("amortized.config.settings.mlflow_tracking_uri", "http://mlflow"):
        await _persist_metric_set(config, "", db=None)

    tag.assert_awaited_once()
