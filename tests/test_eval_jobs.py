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
    "rubric": [{"name": "score_accuracy", "description": "Facts match the reference"}],
    # A rubric without a judge now fails at the validate/create boundary
    # (the test env has no gateway/SDG ancestor to auto-fill one).
    "judge": {"base_url": "http://judge:8000/v1", "model": "gpt-judge"},
}

# Every eval carries a rubric now (built-ins removed) — builder tests
# that aren't about judge resolution pass an explicit judge.
JUDGE = {"base_url": "http://judge:8000/v1", "model": "gpt-judge"}


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
    async def test_create_eval_job_requires_rubric(self, client: httpx.AsyncClient) -> None:
        # The built-in structural metrics were removed — every eval
        # needs at least one rubric criterion.
        response = await _create_eval(client, rubric=[])
        assert response.status_code == 422
        assert "rubric criterion" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_create_eval_job_requires_model_source(self, client: httpx.AsyncClient) -> None:
        # A config with no model to evaluate fails at the boundary
        # (Pydantic model_validator), not at dispatch.
        response = await _create_eval(client, endpoint=None)
        assert response.status_code == 422
        assert "model source" in str(response.json())

    @pytest.mark.asyncio
    async def test_create_eval_job_requires_judge_for_rubric(
        self, client: httpx.AsyncClient
    ) -> None:
        # A rubric with no judge (and no SDG ancestor to auto-fill one)
        # fails at the boundary, not at dispatch.
        response = await _create_eval(client, judge=None)
        assert response.status_code == 422
        assert "judge endpoint is required" in str(response.json())

    @pytest.mark.asyncio
    async def test_create_eval_job_with_model_name(self, client: httpx.AsyncClient) -> None:
        response = await _create_eval(
            client, endpoint=None, model_name_or_path="Qwen/Qwen3.5-4B"
        )
        assert response.status_code == 201
        data = response.json()
        assert data["config"]["model_name_or_path"] == "Qwen/Qwen3.5-4B"
        assert "endpoint" not in data["config"] or data["config"]["endpoint"] is None

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
            "judge": JUDGE,
            "max_samples": 50,
            "temperature": 0.0,
        }
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        assert result.image == "ghcr.io/amortized-ai/eval:latest"
        assert result.command[:2] == ["python3", "/app/run_eval.py"]
        assert result.resources.gpus == 0
        # eval image is non-root (USER 1000); sdg/training images are root
        assert result.run_as_non_root is True

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
    async def test_build_embeds_serving_for_model_name(self) -> None:
        config = {**EVAL_BODY, "model_name_or_path": "Qwen/Qwen3.5-4B", "judge": JUDGE}
        del config["endpoint"]
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        import json

        # command is a sh -c script that runs serve, waits for health, evals
        assert result.command[:2] == ["sh", "-c"]
        script = result.command[2]
        assert "=== EVAL-STAGE: serving ===" in script
        assert "=== EVAL-STAGE: waiting-for-endpoint ===" in script
        assert "=== EVAL-STAGE: evaluating ===" in script
        assert "serve_vllm.py serve" in script
        assert "run_eval.py" in script
        # fail fast when the server dies; never eval without an endpoint
        assert "wait_for_endpoint.py" in script
        assert "EVAL-STAGE: serve-failed ===" in script
        assert "EVAL_RC" in script
        # brace group: a failed pre-command aborts the whole script
        assert script.startswith("{")
        assert script.rstrip().endswith("}")
        # GPU budget like a training job: nvidia.com/gpu device request
        assert result.resources.gpus == 1
        assert result.resources.cpus == 4
        assert "NVIDIA_VISIBLE_DEVICES" not in result.env
        # serve assets ship alongside config.json
        assert "serve_vllm.py" in result.config_files
        assert "patch_model_config.py" in result.config_files
        assert "wait_for_endpoint.py" in result.config_files
        # model endpoint points at the in-pod server
        runner = json.loads(result.config_files["config.json"])
        assert runner["endpoints"]["model"]["base_url"] == "http://localhost:8000/v1"
        assert runner["endpoints"]["model"]["model"] == "Qwen/Qwen3.5-4B"
        # resolved config records the embedded serve parameters
        assert result.resolved_config["gpus"] == 1
        assert result.resolved_config["gpu_memory_utilization"] == 0.9
        assert result.resolved_config["served_model_name"] == "Qwen/Qwen3.5-4B"

    @pytest.mark.asyncio
    async def test_build_embeds_serving_respects_explicit_utilization(self) -> None:
        config = {
            **EVAL_BODY,
            "model_name_or_path": "Qwen/Qwen3.5-4B",
            "vllm_args": ["--gpu-memory-utilization=0.5"],
            "judge": JUDGE,
        }
        del config["endpoint"]
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        assert result.resolved_config["gpu_memory_utilization"] == 0.5
        assert "--gpu-memory-utilization 0.5" in result.command[2]

    @pytest.mark.asyncio
    async def test_build_endpoint_wins_over_model_source(self) -> None:
        # An explicit endpoint bypasses embedded serving entirely.
        config = {
            **EVAL_BODY,
            "model_name_or_path": "Qwen/Qwen3.5-4B",
            "judge": JUDGE,
        }
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        assert result.command[:2] == ["python3", "/app/run_eval.py"]
        assert "EVAL-STAGE" not in result.command[2]
        assert "NVIDIA_VISIBLE_DEVICES" not in result.env

    @pytest.mark.asyncio
    async def test_build_accepts_legacy_endpoint_tuned(self) -> None:
        # Backward compatibility: old configs used endpoint_tuned/endpoint_base.
        config = {
            "endpoint_tuned": {"base_url": "http://tuned:8000/v1", "model": "legacy-tuned"},
            "endpoint_base": {"base_url": "http://base:8000/v1", "model": "legacy-base"},
            "eval_data_run_id": "a" * 32,
            "judge": JUDGE,
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
            "judge": JUDGE,
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
        # No rubric => no judge required, none configured. (Rubrics are
        # mandatory at the API layer now, but the builder still accepts
        # rubric-free configs — e.g. retries of legacy jobs.)
        config = {**EVAL_BODY, "rubric": [], "judge": None}
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        import json

        runner = json.loads(result.config_files["config.json"])
        assert "judge" not in runner["endpoints"]

    @pytest.mark.asyncio
    async def test_build_embeds_serving_for_lora_training_job(self, monkeypatch) -> None:
        # lora_sft: the resolved model path is a merged dir produced by
        # merge_lora.py in a pre-command — not the $SERVE_MODEL_DIR var.
        async def fake_resolve(config):
            return (
                "mdl-lora",
                "/amortized/work/served_model/merged",
                [
                    "mlflow artifacts download -r run -a model"
                    " -d /amortized/work/served_model",
                    "python3 /amortized/merge_lora.py"
                    " /amortized/work/served_model/model"
                    " /amortized/work/served_model/merged",
                ],
            )

        monkeypatch.setattr(eval_builder, "_resolve_training_model", fake_resolve)

        config = {**EVAL_BODY, "training_job_id": "tj-1", "judge": JUDGE}
        del config["endpoint"]
        result = await eval_builder.build({"id": "j1", "type": "eval"}, config, {})

        script = result.command[2]
        # the merged (plain) path is quoted, not the $VAR passthrough
        assert "serve_vllm.py serve /amortized/work/served_model/merged" in script
        assert "merge_lora.py" in "\n".join(result.pre_commands)
        # the merge asset ships alongside the other serve assets
        assert "merge_lora.py" in result.config_files

    @pytest.mark.asyncio
    async def test_is_lora_export_detects_adapter(self, monkeypatch) -> None:
        class FakeClient:
            def __init__(self, files):
                self._files = files

            async def list_artifacts(self, run_id, path):
                return self._files

        import amortized.jobs.eval as eb

        monkeypatch.setattr(eb.config_mod.settings, "mlflow_tracking_uri", "http://ml")
        monkeypatch.setattr(
            "amortized.core.mlflow_client.MLflowClient",
            lambda uri: FakeClient(
                [{"path": "model/adapter_config.json"}, {"path": "model/tokenizer.json"}]
            ),
        )
        assert await eb._is_lora_export("run-1") is True

        monkeypatch.setattr(
            "amortized.core.mlflow_client.MLflowClient",
            lambda uri: FakeClient([{"path": "model/hf_format"}]),
        )
        assert await eb._is_lora_export("run-1") is False

    @pytest.mark.asyncio
    async def test_build_rubric_implies_judge_auto_fill(self, monkeypatch) -> None:
        rubric = [
            {"name": "factual_accuracy", "description": "Facts match the reference"},
            {"name": "reasoning_quality", "description": "Rationale is sound"},
        ]
        config = {**EVAL_BODY, "rubric": rubric, "judge": None}

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
        # the auto-filled judge is persisted into the stored config (so the
        # Evaluation tab merges it with explicit-judge runs of the same model),
        # scrubbed of its api_key
        assert result.resolved_config["judge"]["model"] == "gpt-teacher"
        assert "api_key" not in result.resolved_config["judge"]

    @pytest.mark.asyncio
    async def test_build_rubric_without_resolvable_judge_errors(self) -> None:
        config = {**EVAL_BODY, "rubric": [{"name": "x", "description": "y"}], "judge": None}
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
        config["judge"] = JUDGE
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




class TestRubricDriftWarning:
    @pytest.mark.asyncio
    async def test_validate_warns_on_paraphrased_rubric(self, client: httpx.AsyncClient) -> None:
        # An earlier eval on this dataset used these criteria.
        earlier = await _create_eval(
            client,
            rubric=[{"name": "verdict_accuracy", "description": "original text"}],
        )
        assert earlier.status_code == 201, earlier.text
        job_id = earlier.json()["id"]
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET status = 'succeeded' WHERE id = $1", job_id)

        response = await client.post(
            "/api/v1/jobs/eval/validate",
            json={
                **EVAL_BODY,
                "rubric": [
                    {"name": "verdict_accuracy", "description": "paraphrased text"}
                ],
            },
        )
        assert response.status_code == 200
        warnings = response.json()["warnings"]
        assert warnings and "NOT be comparable" in warnings[0]
        assert "verdict_accuracy" in warnings[0]

    @pytest.mark.asyncio
    async def test_validate_silent_on_exact_reuse(self, client: httpx.AsyncClient) -> None:
        criteria = [{"name": "verdict_accuracy", "description": "original text"}]
        earlier = await _create_eval(client, rubric=criteria)
        job_id = earlier.json()["id"]
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET status = 'succeeded' WHERE id = $1", job_id)

        response = await client.post(
            "/api/v1/jobs/eval/validate",
            json={**EVAL_BODY, "rubric": criteria},
        )
        assert response.status_code == 200
        assert response.json()["warnings"] == []

    @pytest.mark.asyncio
    async def test_validate_silent_on_other_dataset(
        self, client: httpx.AsyncClient
    ) -> None:
        await _create_eval(
            client,
            rubric=[{"name": "verdict_accuracy", "description": "original text"}],
        )
        response = await client.post(
            "/api/v1/jobs/eval/validate",
            json={
                **EVAL_BODY,
                "eval_data_run_id": "b" * 32,  # different dataset
                "rubric": [
                    {"name": "verdict_accuracy", "description": "paraphrased text"}
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["warnings"] == []


class TestRetryEvalJob:
    @pytest.mark.asyncio
    async def _seed_failed_eval(self, client: httpx.AsyncClient) -> str:
        """A failed eval job whose stored config carries runtime keys."""
        job_id = (await _create_eval(client, topic="rfe")).json()["id"]
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute(
                """UPDATE jobs
                      SET status = 'failed',
                          config = config
                              || '{"eval_data_path": "/stale", "port": 8000,
                                   "served_model_name": "stale-name"}'::jsonb
                    WHERE id = $1""",
                job_id,
            )
        return job_id

    @pytest.mark.asyncio
    async def test_retry_clones_request_config_verbatim(
        self, client: httpx.AsyncClient
    ) -> None:
        job_id = await self._seed_failed_eval(client)
        response = await client.post(f"/api/v1/jobs/{job_id}/retry")
        assert response.status_code == 201, response.text
        new_job = response.json()
        assert new_job["id"] != job_id
        assert new_job["status"] == "queued"
        assert new_job["retry_of"] == job_id
        # request_config snapshot wins over the injected runtime keys
        config = new_job["config"]
        assert "eval_data_path" not in config
        assert "port" not in config
        assert config["topic"] == "rfe"

    @pytest.mark.asyncio
    async def test_retry_legacy_row_strips_runtime_keys(
        self, client: httpx.AsyncClient
    ) -> None:
        """Rows whose snapshot was taken from a resolved config (the
        migration's best-effort backfill) still lose the runtime keys —
        the builder recomputes them."""
        job_id = await self._seed_failed_eval(client)
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            # Simulate the migration backfill: snapshot = resolved config
            # (rubric present, runtime keys present).
            await conn.execute(
                "UPDATE jobs SET request_config = config WHERE id = $1", job_id
            )
        response = await client.post(f"/api/v1/jobs/{job_id}/retry")
        assert response.status_code == 201, response.text
        config = response.json()["config"]
        assert "eval_data_path" not in config
        assert "port" not in config
        assert "served_model_name" not in config
        assert config["topic"] == "rfe"  # preserved from the stored config

    @pytest.mark.asyncio
    async def test_retry_clones_rubric_verbatim(self, client: httpx.AsyncClient) -> None:
        criteria = [{"name": "verdict_accuracy", "description": "original text"}]
        job_id = (await _create_eval(client, rubric=criteria, topic="rfe")).json()["id"]
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute("UPDATE jobs SET status = 'failed' WHERE id = $1", job_id)
        retried = (await client.post(f"/api/v1/jobs/{job_id}/retry")).json()
        assert retried["config"]["rubric"] == criteria

    @pytest.mark.asyncio
    async def test_retry_requires_failed_status(self, client: httpx.AsyncClient) -> None:
        job_id = (await _create_eval(client)).json()["id"]  # queued
        response = await client.post(f"/api/v1/jobs/{job_id}/retry")
        assert response.status_code == 422
        assert "only failed or cancelled" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_retry_requires_eval_type(self, client: httpx.AsyncClient) -> None:
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO jobs (id, type, status, config, created_at)
                   VALUES ('sdg-1', 'sdg', 'failed', '{}', now())"""
            )
        response = await client.post("/api/v1/jobs/sdg-1/retry")
        assert response.status_code == 422
        assert "only eval jobs" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_retry_unknown_job_404(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs/missing/retry")
        assert response.status_code == 404


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
