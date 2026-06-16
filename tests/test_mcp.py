"""Tests for the fastmcp MCP server and job management tools."""

import os

import httpx
import pytest

from amortized.main import app
from amortized.mcp import server as mcp_server


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db as db_mod
    import amortized.db.connection as db_conn_mod

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_mod.settings = new_settings
    db_conn_mod.settings = new_settings

    mcp_server._fastapi_app = app


@pytest.fixture
async def db_ready() -> None:  # type: ignore[misc]
    from amortized.db import init_db

    await init_db()
    yield  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Scaffold tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_endpoint_mounted() -> None:
    """The /mcp endpoint is mounted and reachable."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/mcp")
    assert response.status_code != 404


@pytest.mark.asyncio
async def test_mcp_does_not_break_health() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# _call helper tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_success(db_ready: None) -> None:
    result = await mcp_server._call("GET", "/api/v1/health")
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_call_404(db_ready: None) -> None:
    with pytest.raises(ValueError, match="not found"):
        await mcp_server._call("GET", "/api/v1/jobs/nonexistent-id")


# ---------------------------------------------------------------------------
# Tool: list_jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_jobs_empty(db_ready: None) -> None:
    result = await mcp_server.list_jobs()
    assert result == []


@pytest.mark.asyncio
async def test_list_jobs_with_filter(db_ready: None) -> None:
    result = await mcp_server.list_jobs(status="running")
    assert result == []


# ---------------------------------------------------------------------------
# Tool: submit_job + get_job + list_jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_and_get_job(db_ready: None) -> None:
    job = await mcp_server.submit_job(
        type="sdg",
        config={
            "model": "openai/gpt-4o-mini",
            "num_samples": 10,
        },
    )
    assert job["type"] == "sdg"
    assert job["status"] == "queued"
    assert job["id"]

    fetched = await mcp_server.get_job(job["id"])
    assert fetched["id"] == job["id"]
    assert fetched["type"] == "sdg"


@pytest.mark.asyncio
async def test_submit_training_job(db_ready: None) -> None:
    job = await mcp_server.submit_job(
        type="training",
        config={
            "algorithm": "sft",
            "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
            "data_path": "./data.jsonl",
        },
    )
    assert job["type"] == "training"
    assert job["status"] == "queued"


@pytest.mark.asyncio
async def test_submit_job_invalid_type(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.submit_job(type="invalid", config={})


@pytest.mark.asyncio
async def test_list_jobs_after_submit(db_ready: None) -> None:
    await mcp_server.submit_job(
        type="sdg",
        config={"model": "openai/gpt-4o-mini", "num_samples": 5},
    )
    jobs = await mcp_server.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["type"] == "sdg"


@pytest.mark.asyncio
async def test_list_jobs_type_filter(db_ready: None) -> None:
    await mcp_server.submit_job(
        type="sdg",
        config={"model": "openai/gpt-4o-mini", "num_samples": 5},
    )
    await mcp_server.submit_job(
        type="training",
        config={
            "algorithm": "sft",
            "model_name_or_path": "test/model",
            "data_path": "./data.jsonl",
        },
    )
    sdg_jobs = await mcp_server.list_jobs(type="sdg")
    assert len(sdg_jobs) == 1
    assert sdg_jobs[0]["type"] == "sdg"


# ---------------------------------------------------------------------------
# Tool: cancel_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_job(db_ready: None) -> None:
    job = await mcp_server.submit_job(
        type="sdg",
        config={"model": "openai/gpt-4o-mini", "num_samples": 5},
    )
    cancelled = await mcp_server.cancel_job(job["id"])
    assert cancelled["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_nonexistent_job(db_ready: None) -> None:
    with pytest.raises(ValueError, match="not found"):
        await mcp_server.cancel_job("nonexistent-id")


# ---------------------------------------------------------------------------
# Tool: get_job_logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_logs_empty(db_ready: None) -> None:
    job = await mcp_server.submit_job(
        type="sdg",
        config={"model": "openai/gpt-4o-mini", "num_samples": 5},
    )
    logs = await mcp_server.get_job_logs(job["id"])
    assert isinstance(logs, list)


@pytest.mark.asyncio
async def test_get_job_logs_nonexistent(db_ready: None) -> None:
    with pytest.raises(ValueError, match="not found"):
        await mcp_server.get_job_logs("nonexistent-id")


# ---------------------------------------------------------------------------
# Tool: get_job_metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_metrics_non_training(db_ready: None) -> None:
    job = await mcp_server.submit_job(
        type="sdg",
        config={"model": "openai/gpt-4o-mini", "num_samples": 5},
    )
    with pytest.raises(ValueError, match="training"):
        await mcp_server.get_job_metrics(job["id"])


@pytest.mark.asyncio
async def test_get_job_metrics_no_data(db_ready: None) -> None:
    job = await mcp_server.submit_job(
        type="training",
        config={
            "algorithm": "sft",
            "model_name_or_path": "test/model",
            "data_path": "./data.jsonl",
        },
    )
    result = await mcp_server.get_job_metrics(job["id"])
    assert result["total_steps"] == 0
    assert result["trend"] == "no metrics recorded yet"


# ---------------------------------------------------------------------------
# Tool: get_job_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_results_non_eval(db_ready: None) -> None:
    job = await mcp_server.submit_job(
        type="sdg",
        config={"model": "openai/gpt-4o-mini", "num_samples": 5},
    )
    with pytest.raises(ValueError, match="eval"):
        await mcp_server.get_job_results(job["id"])


# ---------------------------------------------------------------------------
# Tool: get_job (not found)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError, match="not found"):
        await mcp_server.get_job("nonexistent-id")


# ---------------------------------------------------------------------------
# _summarise_metrics unit tests
# ---------------------------------------------------------------------------


def test_summarise_metrics_empty() -> None:
    result = mcp_server._summarise_metrics([])
    assert result["total_steps"] == 0
    assert result["trend"] == "no metrics recorded yet"


def test_summarise_metrics_single_step() -> None:
    result = mcp_server._summarise_metrics([{"step": 1, "loss": 2.5, "epoch": 0.1}])
    assert result["total_steps"] == 1
    assert result["latest"]["loss"] == 2.5
    assert "single step" in result["trend"]


def test_summarise_metrics_decreasing_loss() -> None:
    metrics = [
        {"step": 1, "loss": 3.0, "epoch": 0.0},
        {"step": 2, "loss": 2.5, "epoch": 0.5},
        {"step": 3, "loss": 1.5, "epoch": 1.0},
    ]
    result = mcp_server._summarise_metrics(metrics)
    assert result["total_steps"] == 3
    assert "decreasing" in result["trend"]
    assert result["latest"]["step"] == 3
