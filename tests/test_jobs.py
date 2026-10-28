"""Tests for job CRUD API endpoints."""

import os

import httpx
import pytest
from conftest import TEST_DATABASE_URL

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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute("TRUNCATE jobs")
        yield c  # type: ignore[misc]


TRAINING_BODY = {
    "algorithm": "sft",
    "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "data_path": "./data.jsonl",
}

SDG_BODY = {
    "num_records": 10,
    "columns": [
        {
            "column_type": "sampler",
            "name": "q",
            "sampler_type": "category",
            "params": {"values": ["A", "B"]},
        }
    ],
}


async def _create_training(client: httpx.AsyncClient, **overrides: object) -> httpx.Response:
    body = {**TRAINING_BODY, **overrides}
    return await client.post("/api/v1/jobs/training", json=body)


async def _create_sdg(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post("/api/v1/jobs/sdg", json=SDG_BODY)


class TestCreateJob:
    @pytest.mark.asyncio
    async def test_create_training_job(self, client: httpx.AsyncClient) -> None:
        response = await _create_training(client)
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "training"
        assert data["status"] == "queued"
        assert data["config"]["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_sdg_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "num_records": 10,
                "columns": [
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                        "system_prompt": "Generate a question.",
                        "prompt": "Context: {{ content }}",
                    }
                ],
                "model_configs": [{"alias": "text", "model": "gpt-4o", "provider": "gateway"}],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "sdg"
        assert data["status"] == "queued"
        assert data["config"]["num_records"] == 10

    @pytest.mark.asyncio
    async def test_create_training_with_parent_job_id(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        response = await _create_training(client, parent_job_id="abc-123")
        assert response.status_code == 201
        assert response.json()["parent_job_id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_create_training_with_data_run_id(self, client: httpx.AsyncClient) -> None:
        # Direct dataset run reference (uploaded dataset or materialized
        # split) instead of parent_job_id / data_path.
        response = await _create_training(client, data_run_id="d" * 32, data_path="")
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "training"
        assert data["config"]["data_run_id"] == "d" * 32

    @pytest.mark.asyncio
    async def test_training_requires_some_data_reference(self, client: httpx.AsyncClient) -> None:
        response = await _create_training(client, data_path="")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_training_missing_algorithm(self, client: httpx.AsyncClient) -> None:
        # The algorithm key-guard moved off the request model into _validate_training_data
        # (semantic check), so a missing algorithm is still a 422 — now naming algorithm.
        response = await client.post(
            "/api/v1/jobs/training",
            json={"model_name_or_path": "test"},
        )
        assert response.status_code == 422
        assert "algorithm" in str(response.json())

    @pytest.mark.asyncio
    async def test_training_missing_model(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/training",
            json={"algorithm": "sft"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_sdg_missing_columns_accepted_as_empty(self, client: httpx.AsyncClient) -> None:
        # The columns key-guard is gone: a wholly missing columns key defaults to the empty
        # list, which DD permits (see test_sdg_empty_columns_accepted), so it is accepted
        # identically rather than rejected at the request layer.
        response = await client.post("/api/v1/jobs/sdg", json={})
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_sdg_empty_columns_accepted(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs/sdg", json={"columns": []})
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_sdg_unknown_column_type(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={"columns": [{"column_type": "llm_text", "name": "q"}]},
        )
        assert response.status_code == 422
        assert "llm_text" in str(response.json())

    @pytest.mark.asyncio
    async def test_sdg_valid_config(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/sdg",
            json={
                "columns": [
                    {
                        "column_type": "sampler",
                        "name": "difficulty",
                        "sampler_type": "category",
                        "params": {"values": ["Easy", "Hard"]},
                    },
                    {
                        "column_type": "llm-text",
                        "name": "question",
                        "model_alias": "text",
                        "system_prompt": "Generate a question.",
                        "prompt": "Difficulty: {{ difficulty }}",
                    },
                ],
                "model_configs": [{"alias": "text", "model": "gpt-4o", "provider": "gateway"}],
            },
        )
        assert response.status_code == 201


class TestUpstreamDependencyPrecondition:
    """The 'don't advance past a non-succeeded upstream' invariant is enforced at the TOOL
    boundary: validate_*/create_* call _require_dependency_succeeded and return HTTP 422, which
    fastapi-mcp surfaces to the agent as a readable tool error it self-corrects from. This is
    the structural home of the invariant (it replaces the former agent.py prose stage-gate —
    see docs/agent-gates-proposal.md), so it is enforced for every driver tier."""

    @pytest.mark.asyncio
    async def test_training_rejects_nonexistent_parent(self, client: httpx.AsyncClient) -> None:
        # parent_job_id as the data source (no data_path) referencing a job that doesn't exist.
        resp = await _create_training(client, parent_job_id="no-such-job", data_path="")
        assert resp.status_code == 422
        assert "parent_job_id" in str(resp.json()).lower()

    @pytest.mark.asyncio
    async def test_training_rejects_unsucceeded_parent(self, client: httpx.AsyncClient) -> None:
        # A freshly-created SDG job is 'queued', not 'succeeded' — training on it must be rejected.
        sdg = await _create_sdg(client)
        assert sdg.status_code == 201
        parent_id = sdg.json()["id"]
        resp = await _create_training(client, parent_job_id=parent_id, data_path="")
        assert resp.status_code == 422
        assert "succeeded" in str(resp.json()).lower()

    @pytest.mark.asyncio
    async def test_validate_training_also_rejects_unsucceeded_parent(
        self, client: httpx.AsyncClient
    ) -> None:
        # The same invariant fires on the AGENT-facing validate_* tool (the model sees this 422
        # as a tool error), not only on the frontend create_* path.
        sdg = await _create_sdg(client)
        parent_id = sdg.json()["id"]
        body = {**TRAINING_BODY, "data_path": "", "parent_job_id": parent_id}
        resp = await client.post("/api/v1/jobs/training/validate", json=body)
        assert resp.status_code == 422
        assert "succeeded" in str(resp.json()).lower()

    @pytest.mark.asyncio
    async def test_validate_eval_rejects_unsucceeded_dataset_parent(
        self, client: httpx.AsyncClient
    ) -> None:
        # Eval has TWO upstream deps; the dataset parent (parent_job_id) must be 'succeeded'.
        sdg = await _create_sdg(client)
        parent_id = sdg.json()["id"]
        body = {
            "model_name_or_path": "Qwen/Qwen3.5-0.8B",
            "judge": {"base_url": "http://x/v1", "model": "gpt-oss"},
            "rubric": [{"name": "acc", "description": "Is the answer correct?"}],
            "parent_job_id": parent_id,
        }
        resp = await client.post("/api/v1/jobs/eval/validate", json=body)
        assert resp.status_code == 422
        assert "succeeded" in str(resp.json()).lower()

    @pytest.mark.asyncio
    async def test_validate_eval_rejects_unsucceeded_training_model(
        self, client: httpx.AsyncClient
    ) -> None:
        # Eval's other upstream dep: the model under eval (training_job_id) must be 'succeeded'.
        # A fresh training job is 'queued', so evaluating it must be rejected at validate time.
        tr = await _create_training(client)
        tr_id = tr.json()["id"]
        body = {
            "training_job_id": tr_id,
            "judge": {"base_url": "http://x/v1", "model": "gpt-oss"},
            "rubric": [{"name": "acc", "description": "Is the answer correct?"}],
            "eval_data_run_id": "d" * 32,
        }
        resp = await client.post("/api/v1/jobs/eval/validate", json=body)
        assert resp.status_code == 422
        assert "succeeded" in str(resp.json()).lower()


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client: httpx.AsyncClient) -> None:
        await _create_training(client)
        await _create_sdg(client)
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_filter_by_type(self, client: httpx.AsyncClient) -> None:
        await _create_training(client)
        await _create_sdg(client)
        response = await client.get("/api/v1/jobs?type=training")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 1
        assert jobs[0]["type"] == "training"


class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_existing_job(self, client: httpx.AsyncClient) -> None:
        create_resp = await _create_training(client)
        job_id = create_resp.json()["id"]
        response = await client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["id"] == job_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404


class TestCancelJob:
    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, client: httpx.AsyncClient) -> None:
        create_resp = await _create_training(client)
        job_id = create_resp.json()["id"]
        response = await client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        response = await client.delete("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404


class TestErrorFieldSerialization:
    @pytest.mark.asyncio
    async def test_error_is_null_not_string_none(self, client: httpx.AsyncClient) -> None:
        create_resp = await _create_training(client)
        job_id = create_resp.json()["id"]
        response = await client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_has_gpu_info(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "gpu" in data
        assert "timestamp" in data


class TestConfig:
    @pytest.mark.asyncio
    async def test_config_endpoint(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert "default_compute_backend" in data
        assert "compute_namespace" in data
        assert "mlflow_tracking_uri" in data
        assert "mlflow_gateway_uri" in data
        assert "available_backends" in data
        assert "version" in data


class TestJobLogs:
    @pytest.mark.asyncio
    async def test_logs_no_handle(self, client: httpx.AsyncClient) -> None:
        create_resp = await _create_training(client)
        job_id = create_resp.json()["id"]
        response = await client.get(f"/api/v1/jobs/{job_id}/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["logs"] == []

    @pytest.mark.asyncio
    async def test_logs_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/jobs/nonexistent/logs")
        assert response.status_code == 404
