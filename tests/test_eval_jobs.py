"""Tests for eval job API endpoints and builder."""

import os

import httpx
import pytest
from conftest import TEST_DATABASE_URL

from amortized.jobs import eval as eval_builder
from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db.connection as db_conn_mod

    os.environ["AMORTIZED_DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    import subprocess

    import asyncpg

    conn = await asyncpg.connect(TEST_DATABASE_URL)
    await conn.execute("DROP TABLE IF EXISTS alembic_version")
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.close()
    env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
    subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute("TRUNCATE jobs")
        yield c  # type: ignore[misc]


EVAL_BODY = {
    "endpoint_base": {
        "base_url": "http://base:8000/v1",
        "model": "qwen-base",
    },
    "endpoint_tuned": {
        "base_url": "http://tuned:8000/v1",
        "model": "qwen-tuned",
    },
    "eval_data_run_id": "a" * 32,
}


async def _create_eval(client: httpx.AsyncClient, **overrides: object) -> httpx.Response:
    body = {**EVAL_BODY, **overrides}
    return await client.post("/api/v1/jobs/eval", json=body)


class TestCreateEvalJob:
    @pytest.mark.asyncio
    async def test_create_eval_job_with_run_id(self, client: httpx.AsyncClient) -> None:
        response = await _create_eval(client)
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "eval"
        assert data["status"] == "queued"
        assert data["config"]["endpoint_tuned"]["model"] == "qwen-tuned"
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_eval_job_requires_data_ref(self, client: httpx.AsyncClient) -> None:
        response = await _create_eval(client, eval_data_run_id="")
        assert response.status_code == 422
        assert "parent_job_id" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_create_eval_job_requires_endpoints(self, client: httpx.AsyncClient) -> None:
        response = await _create_eval(client, endpoint_tuned={"base_url": "", "model": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_eval_job_validates_parent_status(self, client: httpx.AsyncClient) -> None:
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO jobs (id, type, status, config, created_at)
                   VALUES ('parent-1', 'sdg', 'failed', '{}', now())"""
            )

        response = await _create_eval(client, eval_data_run_id="", parent_job_id="parent-1")
        assert response.status_code == 422
        assert "succeeded" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_api_key_not_echoed_in_response(self, client: httpx.AsyncClient) -> None:
        response = await _create_eval(
            client,
            endpoint_base={
                "base_url": "http://base:8000/v1",
                "model": "qwen-base",
                "api_key": "sk-secret",
            },
        )
        assert response.status_code == 201
        config = response.json()["config"]
        assert "api_key" not in config["endpoint_base"]


class TestEvalBuilder:
    @pytest.mark.asyncio
    async def test_build_generates_runner_config(self) -> None:
        config = {
            **EVAL_BODY,
            "max_samples": 50,
            "temperature": 0.0,
        }
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        assert result.image == "ghcr.io/amortized-ai/eval:latest"
        assert result.command[:2] == ["python3", "/app/run_eval.py"]
        assert result.resources.gpus == 0

        import json

        runner = json.loads(result.config_files["config.json"])
        assert runner["eval_data_path"] == "/amortized/work/eval_data/generated_data"
        assert runner["endpoints"]["base"]["model"] == "qwen-base"
        assert "api_key" not in runner["endpoints"]["base"]
        assert runner["max_samples"] == 50
        assert any("mlflow artifacts download" in c for c in result.pre_commands)
        assert any("eval_results" in c and "log-artifacts" in c for c in result.post_commands)

    @pytest.mark.asyncio
    async def test_build_scrubs_api_keys_from_resolved_config(self) -> None:
        config = {
            **EVAL_BODY,
            "endpoint_base": {
                "base_url": "http://base:8000/v1",
                "model": "qwen-base",
                "api_key": "sk-secret",
            },
        }
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        assert result.env["EVAL_BASE_API_KEY"] == "sk-secret"
        assert "api_key" not in result.resolved_config["endpoint_base"]

    @pytest.mark.asyncio
    async def test_build_requires_judge_for_win_rate(self) -> None:
        config = {**EVAL_BODY, "metrics": ["judge_win_rate"]}
        with pytest.raises(eval_builder.JobBuildError, match="judge"):
            await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

    @pytest.mark.asyncio
    async def test_build_judge_implies_win_rate_metric(self) -> None:
        config = {
            **EVAL_BODY,
            "judge": {"base_url": "http://judge:8000/v1", "model": "gpt-judge"},
        }
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        import json

        runner = json.loads(result.config_files["config.json"])
        assert "judge_win_rate" in runner["metrics"]
        assert "judge" in runner["endpoints"]

    @pytest.mark.asyncio
    async def test_build_requires_data_source(self) -> None:
        config = {k: v for k, v in EVAL_BODY.items() if k != "eval_data_run_id"}
        with pytest.raises(eval_builder.JobBuildError, match="parent_job_id"):
            await eval_builder.build({"id": "j1", "type": "eval"}, config, {})
