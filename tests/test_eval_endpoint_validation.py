"""Eval job creation must reject endpoints pointing at dead serve jobs."""
import pytest
from unittest.mock import AsyncMock, patch

from amortized.api.jobs import _validate_eval_endpoint


def _config(url):
    return {"endpoint": {"base_url": url, "model": "m"}}


@pytest.mark.asyncio
async def test_rejects_cancelled_serve_job():
    job = {"id": "9f226235-e8bd-433c-aca0-4a42181210f4", "type": "serve", "status": "cancelled"}
    repo = AsyncMock()
    repo.get_job = AsyncMock(return_value=job)
    with patch("amortized.api.jobs.Repository", return_value=repo):
        errors = await _validate_eval_endpoint(
            _config("http://amortized-9f226235-e8bd-433c-aca0-4a42181210f4.amortized-xyang-jobs.svc.cluster.local:8000/v1"),
            db=None,
        )
    assert errors and "not running" in errors[0]


@pytest.mark.asyncio
async def test_accepts_running_serve_job():
    job = {"id": "9f226235-e8bd-433c-aca0-4a42181210f4", "type": "serve", "status": "running"}
    repo = AsyncMock()
    repo.get_job = AsyncMock(return_value=job)
    with patch("amortized.api.jobs.Repository", return_value=repo):
        errors = await _validate_eval_endpoint(
            _config("http://amortized-9f226235-e8bd-433c-aca0-4a42181210f4.amortized-xyang-jobs.svc.cluster.local:8000/v1"),
            db=None,
        )
    assert errors == []


@pytest.mark.asyncio
async def test_ignores_gateway_urls():
    errors = await _validate_eval_endpoint(
        _config("http://mlflow.amortized.svc.cluster.local:5000/gateway/mlflow/v1"),
        db=None,
    )
    assert errors == []


@pytest.mark.asyncio
async def test_ignores_unknown_host_job():
    repo = AsyncMock()
    repo.get_job = AsyncMock(return_value=None)
    with patch("amortized.api.jobs.Repository", return_value=repo):
        errors = await _validate_eval_endpoint(
            _config("http://amortized-00000000-1111-2222-3333-444444444444.amortized-xyang-jobs.svc.cluster.local:8000/v1"),
            db=None,
        )
    assert errors == []
