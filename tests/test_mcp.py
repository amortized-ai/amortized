"""Tests for the fastmcp MCP server — all 31 tools and 3 resources."""

import json
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


@pytest.fixture
async def db_with_evaluators() -> None:  # type: ignore[misc]
    from amortized.api import evaluators
    from amortized.db import get_db, init_db

    await init_db()
    async for db in get_db():
        await evaluators.seed_default_evaluators(db)
    yield  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Snapshot test — exact set of 31 tool names
# ---------------------------------------------------------------------------

EXPECTED_TOOL_NAMES = sorted(
    [
        # Job management (8)
        "submit_job",
        "list_jobs",
        "get_job",
        "cancel_job",
        "resume_job",
        "get_job_logs",
        "get_job_metrics",
        "get_job_results",
        # Artifacts (4)
        "list_artifacts",
        "get_artifact",
        "preview_artifact",
        "upload_artifact",
        # Recipes & discovery (6)
        "list_recipes",
        "get_recipe",
        "submit_recipe_job",
        "list_sdg_capabilities",
        "validate_config",
        "get_job_type_schema",
        # Judge & evaluators (4)
        "judge_data",
        "list_judge_templates",
        "list_evaluators",
        "run_evaluation",
        # Admin & infrastructure (9)
        "list_compute_backends",
        "register_backend",
        "test_backend",
        "remove_backend",
        "add_api_key",
        "list_api_keys",
        "delete_api_key",
        "estimate_vram",
        "health_check",
    ]
)


@pytest.mark.asyncio
async def test_tool_names_snapshot() -> None:
    """All 31 tools are registered with the expected names."""
    from amortized.mcp.server import mcp

    tools = await mcp.list_tools()
    registered = sorted(t.name for t in tools)
    assert registered == EXPECTED_TOOL_NAMES
    assert len(registered) == 31


@pytest.mark.asyncio
async def test_resource_uris_snapshot() -> None:
    """All 3 MCP resources are registered (2 static + 1 template)."""
    from amortized.mcp.server import mcp

    resources = await mcp.list_resources()
    uris = sorted(str(r.uri) for r in resources)
    assert "amortized://capabilities" in uris
    assert "amortized://recipes" in uris

    templates = await mcp.list_resource_templates()
    template_uris = [t.uri_template for t in templates]
    assert any("recipes" in u for u in template_uris)


# ---------------------------------------------------------------------------
# Scaffold tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_endpoint_mounted() -> None:
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
# Tool: list_jobs / submit_job / get_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_jobs_empty(db_ready: None) -> None:
    result = await mcp_server.list_jobs()
    assert result == []


@pytest.mark.asyncio
async def test_list_jobs_with_filter(db_ready: None) -> None:
    result = await mcp_server.list_jobs(status="running")
    assert result == []


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
# Tool: list_artifacts / get_artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_artifacts_empty(db_ready: None) -> None:
    result = await mcp_server.list_artifacts()
    assert result == []


@pytest.mark.asyncio
async def test_list_artifacts_with_type_filter(db_ready: None) -> None:
    result = await mcp_server.list_artifacts(type="model")
    assert result == []


@pytest.mark.asyncio
async def test_get_artifact_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError, match="not found"):
        await mcp_server.get_artifact("nonexistent-id")


# ---------------------------------------------------------------------------
# Tool: upload_artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_artifact(db_ready: None, tmp_path: object) -> None:
    file_path = str(tmp_path) + "/test_data.jsonl"
    with open(file_path, "w") as f:
        f.write('{"question": "What is AI?"}\n')

    result = await mcp_server.upload_artifact(
        file_path=file_path,
        artifact_type="dataset",
        name="test_data.jsonl",
    )
    assert result["id"]
    assert result["name"] == "test_data.jsonl"

    artifacts = await mcp_server.list_artifacts()
    assert len(artifacts) == 1
    assert artifacts[0]["id"] == result["id"]


@pytest.mark.asyncio
async def test_upload_artifact_file_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError, match="File not found"):
        await mcp_server.upload_artifact(file_path="/nonexistent/file.jsonl")


# ---------------------------------------------------------------------------
# Tool: preview_artifact (error path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_artifact_job_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError, match="not found"):
        await mcp_server.preview_artifact(job_id="nonexistent", artifact_id="nonexistent")


# ---------------------------------------------------------------------------
# Tool: list_recipes / get_recipe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_recipes(db_ready: None) -> None:
    result = await mcp_server.list_recipes()
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_recipe_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.get_recipe("nonexistent/recipe")


# ---------------------------------------------------------------------------
# Tool: submit_recipe_job (error path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_recipe_job_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.submit_recipe_job(recipe="nonexistent/recipe")


# ---------------------------------------------------------------------------
# Tool: list_sdg_capabilities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_sdg_capabilities(db_ready: None) -> None:
    result = await mcp_server.list_sdg_capabilities()
    assert "available" in result
    assert "strategies" in result
    assert "attribute_types" in result


# ---------------------------------------------------------------------------
# Tool: validate_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_config_valid_sdg(db_ready: None) -> None:
    result = await mcp_server.validate_config(
        type="sdg",
        config={"model": "openai/gpt-4o-mini", "num_samples": 10},
    )
    assert "valid" in result or "type" in result


@pytest.mark.asyncio
async def test_validate_config_invalid_type(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.validate_config(type="invalid", config={})


# ---------------------------------------------------------------------------
# Tool: get_job_type_schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_type_schema_training(db_ready: None) -> None:
    result = await mcp_server.get_job_type_schema("training")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_job_type_schema_sdg(db_ready: None) -> None:
    result = await mcp_server.get_job_type_schema("sdg")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_job_type_schema_invalid(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.get_job_type_schema("invalid")


# ---------------------------------------------------------------------------
# Tool: list_judge_templates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_judge_templates(db_ready: None) -> None:
    result = await mcp_server.list_judge_templates()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tool: judge_data (error path — needs LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_data_invalid_template(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.judge_data(
            template="nonexistent/template",
            data=[{"text": "hello"}],
            model="openai/gpt-4o-mini",
        )


# ---------------------------------------------------------------------------
# Tool: list_evaluators
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_evaluators(db_with_evaluators: None) -> None:
    result = await mcp_server.list_evaluators()
    assert isinstance(result, list)
    assert len(result) >= 5


@pytest.mark.asyncio
async def test_list_evaluators_empty(db_ready: None) -> None:
    result = await mcp_server.list_evaluators()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tool: run_evaluation (error path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_evaluation_invalid_evaluator(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.run_evaluation(
            evaluator_id="nonexistent-id",
            dataset=[{"text": "hello"}],
        )


# ---------------------------------------------------------------------------
# Tool: list_compute_backends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_compute_backends(db_ready: None) -> None:
    result = await mcp_server.list_compute_backends()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tool: register_backend / remove_backend (error paths)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_backend_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.remove_backend("nonexistent-backend")


# ---------------------------------------------------------------------------
# Tool: test_backend (error path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_backend_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.test_backend("nonexistent-backend")


# ---------------------------------------------------------------------------
# Tool: add_api_key / list_api_keys / delete_api_key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_api_keys_empty(db_ready: None) -> None:
    result = await mcp_server.list_api_keys()
    assert result == []


@pytest.mark.asyncio
async def test_add_and_list_api_key(db_ready: None) -> None:
    added = await mcp_server.add_api_key(
        name="openai-test",
        provider="openai",
        key="sk-test-key-12345678",
    )
    assert added["id"]
    assert added["provider"] == "openai"
    assert "5678" in added["key_preview"]

    keys = await mcp_server.list_api_keys()
    assert len(keys) == 1
    assert keys[0]["provider"] == "openai"


@pytest.mark.asyncio
async def test_add_and_delete_api_key(db_ready: None) -> None:
    added = await mcp_server.add_api_key(
        name="temp-key",
        provider="anthropic",
        key="sk-ant-test-key-abcd",
    )
    result = await mcp_server.delete_api_key(added["id"])
    assert result == {}

    keys = await mcp_server.list_api_keys()
    assert len(keys) == 0


@pytest.mark.asyncio
async def test_delete_api_key_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError, match="not found"):
        await mcp_server.delete_api_key("nonexistent-id")


# ---------------------------------------------------------------------------
# Tool: estimate_vram
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_vram(db_ready: None) -> None:
    result = await mcp_server.estimate_vram(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    )
    assert "estimated_vram_gb" in result
    assert result["estimated_vram_gb"] > 0


@pytest.mark.asyncio
async def test_estimate_vram_4bit(db_ready: None) -> None:
    result = await mcp_server.estimate_vram(
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        load_in_4bit=True,
    )
    assert result["estimated_vram_gb"] > 0
    assert result["load_in_4bit"] is True


# ---------------------------------------------------------------------------
# Tool: health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check(db_ready: None) -> None:
    result = await mcp_server.health_check()
    assert result["status"] == "ok"
    assert "timestamp" in result
    assert "gpu" in result


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_resource(db_ready: None) -> None:
    result = await mcp_server.capabilities_resource()
    data = json.loads(result)
    assert "job_types" in data
    assert "algorithms" in data
    assert "training" in data["job_types"]
    assert "sft" in data["algorithms"]


@pytest.mark.asyncio
async def test_recipes_resource(db_ready: None) -> None:
    result = await mcp_server.recipes_resource()
    data = json.loads(result)
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_recipe_detail_resource_not_found(db_ready: None) -> None:
    with pytest.raises(ValueError):
        await mcp_server.recipe_detail_resource("nonexistent/recipe")
