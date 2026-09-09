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


# One model per eval job — a single endpoint to evaluate.
EVAL_BODY = {
    "endpoint": {
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
        assert data["config"]["endpoint"]["model"] == "qwen-tuned"
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_eval_job_requires_data_ref(self, client: httpx.AsyncClient) -> None:
        response = await _create_eval(client, eval_data_run_id="")
        assert response.status_code == 422
        assert "parent_job_id" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_create_eval_job_requires_endpoint(self, client: httpx.AsyncClient) -> None:
        response = await _create_eval(client, endpoint={"base_url": "", "model": ""})
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
            endpoint={
                "base_url": "http://tuned:8000/v1",
                "model": "qwen-tuned",
                "api_key": "sk-secret",
            },
        )
        assert response.status_code == 201
        config = response.json()["config"]
        assert "api_key" not in config["endpoint"]


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
        # single model under the "model" key
        assert runner["endpoints"]["model"]["model"] == "qwen-tuned"
        assert "base" not in runner["endpoints"]
        assert "tuned" not in runner["endpoints"]
        assert "api_key" not in runner["endpoints"]["model"]
        assert runner["max_samples"] == 50
        assert any("mlflow artifacts download" in c for c in result.pre_commands)
        assert any("eval_results" in c and "log-artifacts" in c for c in result.post_commands)

    @pytest.mark.asyncio
    async def test_build_accepts_legacy_endpoint_tuned(self) -> None:
        # Backward compatibility: old configs used endpoint_tuned/endpoint_base.
        config = {
            "endpoint_tuned": {"base_url": "http://tuned:8000/v1", "model": "legacy-tuned"},
            "endpoint_base": {"base_url": "http://base:8000/v1", "model": "legacy-base"},
            "eval_data_run_id": "a" * 32,
        }
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        import json

        runner = json.loads(result.config_files["config.json"])
        # endpoint_tuned wins over endpoint_base as the model to evaluate
        assert runner["endpoints"]["model"]["model"] == "legacy-tuned"

    @pytest.mark.asyncio
    async def test_build_scrubs_api_keys_from_resolved_config(self) -> None:
        config = {
            **EVAL_BODY,
            "endpoint": {
                "base_url": "http://tuned:8000/v1",
                "model": "qwen-tuned",
                "api_key": "sk-secret",
            },
        }
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        assert result.env["EVAL_MODEL_API_KEY"] == "sk-secret"
        assert "api_key" not in result.resolved_config["endpoint"]

    @pytest.mark.asyncio
    async def test_build_no_judge_without_rubric(self) -> None:
        # No rubric => no judge required, none configured.
        result = await eval_builder.build({"id": "j1", "type": "eval"}, dict(EVAL_BODY), {})

        import json

        runner = json.loads(result.config_files["config.json"])
        assert "judge" not in runner["endpoints"]

    @pytest.mark.asyncio
    async def test_build_rubric_implies_judge_auto_fill(self, monkeypatch) -> None:
        rubric = [
            {"name": "factual_accuracy", "description": "Facts match the reference"},
            {"name": "reasoning_quality", "description": "Rationale is sound"},
        ]
        config = {**EVAL_BODY, "rubric": rubric}

        async def fake_resolve(job):
            return "gpt-teacher"

        monkeypatch.setattr(eval_builder, "_resolve_teacher_model", fake_resolve)
        monkeypatch.setattr(
            eval_builder.config_mod.settings, "gateway_url", "http://gateway:5000/gw/v1"
        )

        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        import json

        runner = json.loads(result.config_files["config.json"])
        assert runner["rubric"] == rubric
        assert "judge" in runner["endpoints"]
        assert runner["endpoints"]["judge"]["model"] == "gpt-teacher"

    @pytest.mark.asyncio
    async def test_build_rubric_without_resolvable_judge_errors(self) -> None:
        config = {**EVAL_BODY, "rubric": [{"name": "x", "description": "y"}]}
        with pytest.raises(eval_builder.JobBuildError, match="rubric"):
            await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

    @pytest.mark.asyncio
    async def test_build_judge_explicit_overrides_default(self, monkeypatch) -> None:
        config = {
            **EVAL_BODY,
            "rubric": [{"name": "accuracy", "description": "matches reference"}],
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
                "model": {"exact_match": 0.8, "error_rate": 0.0, "num_samples": 6},
                "scores": {"score_accuracy": 0.72, "reasoning_quality": 0.65},
                "scores_n": {"score_accuracy": 5, "reasoning_quality": 5},
                "num_scored": 5,
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
        assert results["model"]["exact_match"] == 0.8
        assert results["scores"]["score_accuracy"] == 0.72


class TestScoreParsing:
    """Unit tests for the runner's absolute-score parsing (no network)."""

    def _load_runner_module(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "containers" / "eval" / "run_eval.py"
        spec = importlib.util.spec_from_file_location("run_eval", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_parse_valid_scores(self) -> None:
        m = self._load_runner_module()
        raw = 'Scores: {"score_accuracy": 8, "reasoning_quality": 3}'
        scores = m.parse_scores(raw, ["score_accuracy", "reasoning_quality"])
        assert scores == {"score_accuracy": 8.0, "reasoning_quality": 3.0}

    def test_parse_missing_criteria_default_none(self) -> None:
        m = self._load_runner_module()
        raw = '{"score_accuracy": 8}'
        scores = m.parse_scores(raw, ["score_accuracy", "reasoning_quality"])
        assert scores == {"score_accuracy": 8.0, "reasoning_quality": None}

    def test_parse_garbage_defaults_all_none(self) -> None:
        m = self._load_runner_module()
        scores = m.parse_scores("not json at all", ["a", "b"])
        assert scores == {"a": None, "b": None}

    def test_parse_clamps_out_of_range(self) -> None:
        m = self._load_runner_module()
        raw = '{"a": 15, "b": -3}'
        scores = m.parse_scores(raw, ["a", "b"])
        assert scores == {"a": 10.0, "b": 0.0}


class TestEvalEndpointSuggestions:
    @pytest.mark.asyncio
    async def test_suggestions_without_training_job(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/eval/endpoint-suggestions")
        assert response.status_code == 200
        data = response.json()
        assert data["training_job_id"] == ""
        assert data["model"] == ""
        assert isinstance(data["known_endpoints"], list)

    @pytest.mark.asyncio
    async def test_suggestions_for_training_job(self, client: httpx.AsyncClient) -> None:
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO jobs (id, type, status, config, created_at, mlflow_run_id)
                   VALUES ('tj-9', 'training', 'succeeded',
                           '{"model_name_or_path": "Qwen/Qwen3.5-2B", "algorithm": "lora_sft"}',
                           now(), 'run9')"""
            )

        class FakeClient:
            def __init__(self, tracking_uri: str, timeout: float = 30.0) -> None:
                pass

            async def list_gateway_models(self) -> list[dict]:
                return [
                    {"name": "gpt-oss", "provider": "openai", "model_name": "openai/gpt-oss-120b"}
                ]

            async def get_run(self, run_id: str) -> dict:
                return {
                    "data": {"tags": [{"key": "model_display_name", "value": "mdl-rfe-scorer"}]}
                }

        from unittest.mock import patch

        import amortized.config as config_mod

        with (
            patch("amortized.api.eval.MLflowClient", FakeClient),
            patch.object(config_mod.settings, "mlflow_tracking_uri", "http://mlflow:5000"),
            patch.object(config_mod.settings, "gateway_url", "http://gw:5000/v1"),
        ):
            response = await client.get(
                "/api/v1/eval/endpoint-suggestions", params={"training_job_id": "tj-9"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "mdl-rfe-scorer"
        assert data["known_endpoints"][0]["model_name"] == "openai/gpt-oss-120b"
        assert data["known_endpoints"][0]["base_url"] == "http://gw:5000/v1"

    @pytest.mark.asyncio
    async def test_suggestions_unknown_training_job(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/v1/eval/endpoint-suggestions", params={"training_job_id": "nope"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == ""
        assert "not found" in data.get("message", "")
