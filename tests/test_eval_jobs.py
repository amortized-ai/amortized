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
    async def test_build_requires_judge_for_win_rate_without_lineage(self) -> None:
        config = {**EVAL_BODY, "metrics": ["judge_win_rate"]}
        with pytest.raises(eval_builder.JobBuildError, match="judge"):
            await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

    @pytest.mark.asyncio
    async def test_build_judge_auto_filled_from_teacher(self, monkeypatch) -> None:
        config = {**EVAL_BODY, "metrics": ["judge_win_rate"]}

        async def fake_resolve(job):
            return "gpt-teacher"

        monkeypatch.setattr(eval_builder, "_resolve_teacher_model", fake_resolve)
        monkeypatch.setattr(
            eval_builder.config_mod.settings, "gateway_url", "http://gateway:5000/gw/v1"
        )

        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        import json

        runner = json.loads(result.config_files["config.json"])
        assert "judge_win_rate" in runner["metrics"]
        assert runner["endpoints"]["judge"]["model"] == "gpt-teacher"
        assert runner["endpoints"]["judge"]["base_url"] == "http://gateway:5000/gw/v1"
        assert result.env["EVAL_JUDGE_API_KEY"] == "not-needed"

    @pytest.mark.asyncio
    async def test_build_judge_explicit_overrides_default(self, monkeypatch) -> None:
        config = {
            **EVAL_BODY,
            "metrics": ["judge_win_rate"],
            "judge": {"base_url": "http://judge:8000/v1", "model": "gpt-judge"},
        }

        async def fake_resolve(job):
            return "gpt-teacher"

        monkeypatch.setattr(eval_builder, "_resolve_teacher_model", fake_resolve)

        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        import json

        runner = json.loads(result.config_files["config.json"])
        assert runner["endpoints"]["judge"]["model"] == "gpt-judge"

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


class TestEvalResultsEndpoint:
    @pytest.mark.asyncio
    async def test_eval_results_not_found(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs/no-such-job/eval-results")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_eval_results_rejects_non_eval_job(self, client: httpx.AsyncClient) -> None:
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO jobs (id, type, status, config, created_at)
                   VALUES ('tj-1', 'training', 'succeeded', '{}', now())"""
            )

        response = await client.get("/api/v1/jobs/tj-1/eval-results")
        assert response.status_code == 400
        assert "not an eval job" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_eval_results_incomplete_job(self, client: httpx.AsyncClient) -> None:
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO jobs (id, type, status, config, created_at)
                   VALUES ('ej-1', 'eval', 'running', '{}', now())"""
            )

        response = await client.get("/api/v1/jobs/ej-1/eval-results")
        assert response.status_code == 200
        assert response.json()["results"] is None
        assert "not available" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_eval_results_returns_metrics(self, client: httpx.AsyncClient) -> None:
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO jobs (id, type, status, config, created_at, mlflow_run_id)
                   VALUES ('ej-2', 'eval', 'succeeded', '{}', now(), 'run123')"""
            )

        metrics = {
            "results": {
                "num_records": 6,
                "base": {"exact_match": 0.5, "error_rate": 0.0},
                "tuned": {"exact_match": 0.8, "error_rate": 0.0},
                "judge": {"win_rate": 0.7, "num_judged": 5},
            }
        }

        class FakeClient:
            def __init__(self, tracking_uri: str, timeout: float = 30.0) -> None:
                pass

            async def get_artifact_text(self, run_id: str, path: str) -> str:
                import json

                return json.dumps(metrics)

        from unittest.mock import patch

        import amortized.config as config_mod

        with (
            patch("amortized.core.mlflow_client.MLflowClient", FakeClient),
            patch.object(config_mod.settings, "mlflow_tracking_uri", "http://mlflow:5000"),
        ):
            response = await client.get("/api/v1/jobs/ej-2/eval-results")

        assert response.status_code == 200
        results = response.json()["results"]
        assert results["tuned"]["exact_match"] == 0.8
        assert results["judge"]["win_rate"] == 0.7
