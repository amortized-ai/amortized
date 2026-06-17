"""Tests for the fastmcp MCP server and job management tools."""

import contextlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from amortized.main import app
from amortized.mcp import server as mcp_server


def _parse_sse_json(text: str) -> Any:
    """Extract the first JSON payload from an SSE response body."""
    for line in text.strip().split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise ValueError(f"No data line in SSE response: {text[:200]}")


def _parse_response(response: httpx.Response) -> Any:
    """Parse a response that may be JSON or SSE-wrapped JSON."""
    ct = response.headers.get("content-type", "")
    if "event-stream" in ct:
        return _parse_sse_json(response.text)
    return response.json()


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
    mcp_server._transport = httpx.ASGITransport(app=app)


@pytest.fixture
async def db_ready() -> None:  # type: ignore[misc]
    from amortized.db import init_db

    await init_db()
    yield  # type: ignore[misc]


@pytest.fixture
async def mcp_lifespan() -> AsyncIterator[None]:
    """Start the MCP session manager lifespan for protocol-level tests."""
    ctx = mcp_server.mcp_http_app.router.lifespan_context(mcp_server.mcp_http_app)
    await ctx.__aenter__()
    yield
    with contextlib.suppress(RuntimeError):
        await ctx.__aexit__(None, None, None)


_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


# ---------------------------------------------------------------------------
# Scaffold tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_endpoint_mounted(mcp_lifespan: None) -> None:
    """The /mcp endpoint is mounted and responds to MCP initialize."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1.0"},
            }},
            headers=_MCP_HEADERS,
        )
    assert response.status_code == 200
    body = _parse_response(response)
    assert body["result"]["serverInfo"]["name"] == "amortized"


@pytest.mark.asyncio
async def test_mcp_tools_list(mcp_lifespan: None) -> None:
    """MCP tools/list returns all registered tools."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        init_resp = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1.0"},
            }},
            headers=_MCP_HEADERS,
        )
        assert init_resp.status_code == 200
        session_id = init_resp.headers.get("mcp-session-id", "")

        headers = {**_MCP_HEADERS}
        if session_id:
            headers["mcp-session-id"] = session_id

        list_resp = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers=headers,
        )
    assert list_resp.status_code == 200
    body = _parse_response(list_resp)
    tool_names = [t["name"] for t in body["result"]["tools"]]
    assert "submit_job" in tool_names
    assert "list_jobs" in tool_names
    assert "get_job" in tool_names


@pytest.mark.asyncio
async def test_mcp_tools_call_list_jobs(mcp_lifespan: None, db_ready: None) -> None:
    """Calling list_jobs via MCP protocol returns an empty list."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        init_resp = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1.0"},
            }},
            headers=_MCP_HEADERS,
        )
        assert init_resp.status_code == 200
        session_id = init_resp.headers.get("mcp-session-id", "")

        headers = {**_MCP_HEADERS}
        if session_id:
            headers["mcp-session-id"] = session_id

        call_resp = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "list_jobs",
                "arguments": {},
            }},
            headers=headers,
        )
    assert call_resp.status_code == 200
    body = _parse_response(call_resp)
    assert "result" in body
    content = body["result"]["content"]
    assert isinstance(content, list)


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


# ---------------------------------------------------------------------------
# Auth + MCP integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_with_api_key_set(db_ready: None) -> None:
    """_call() works when AMORTIZED_API_KEY is set because it forwards auth."""
    import amortized.config as config_mod
    import amortized.main as main_mod

    original_config = config_mod.settings.api_key
    original_main = main_mod._settings.api_key
    try:
        config_mod.settings.api_key = "test-secret-key"
        main_mod._settings.api_key = "test-secret-key"
        result = await mcp_server._call("GET", "/api/v1/jobs")
        assert isinstance(result, list)
    finally:
        config_mod.settings.api_key = original_config
        main_mod._settings.api_key = original_main


@pytest.mark.asyncio
async def test_call_with_api_key_rejects_without_header(db_ready: None) -> None:
    """Requests without auth header are rejected when API key is set."""
    import amortized.config as config_mod
    import amortized.main as main_mod

    original_config = config_mod.settings.api_key
    original_main = main_mod._settings.api_key
    try:
        config_mod.settings.api_key = "test-secret-key"
        main_mod._settings.api_key = "test-secret-key"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/jobs")
        assert response.status_code == 401
    finally:
        config_mod.settings.api_key = original_config
        main_mod._settings.api_key = original_main
