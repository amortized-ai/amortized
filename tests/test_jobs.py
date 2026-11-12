"""Tests for job CRUD API endpoints."""

import os

import httpx
import pytest
from conftest import TEST_DATABASE_URL

from amortized import main
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
    async def test_validate_training_rejects_gpu_over_budget(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The budget check used to run only at create (the user's Confirm click), so the
        # agent never saw it; the validate tool now 422s and names the knob to change.
        from amortized.core import jobs as jobs_mod

        monkeypatch.setattr(jobs_mod._settings, "gpu_budget", 0)
        body = {
            "algorithm": "embedding_sft",
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "data_path": "./data.parquet",
            "train_on_gpu": True,
        }
        resp = await client.post("/api/v1/jobs/training/validate", json=body)
        assert resp.status_code == 422
        assert "set train_on_gpu=false" in str(resp.json())

        resp = await client.post(
            "/api/v1/jobs/training/validate", json={**body, "train_on_gpu": False}
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_classifier_peft_over_budget_names_use_peft(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # classifier_sft with use_peft needs a GPU even with train_on_gpu unset, so the
        # error must point at use_peft — "set train_on_gpu=false" would change nothing.
        from amortized.core import jobs as jobs_mod

        monkeypatch.setattr(jobs_mod._settings, "gpu_budget", 0)
        body = {
            "algorithm": "classifier_sft",
            "model_name_or_path": "Qwen/Qwen3-0.6B",
            "data_path": "./data.parquet",
            "use_peft": True,
        }
        resp = await client.post("/api/v1/jobs/training/validate", json=body)
        assert resp.status_code == 422
        detail = str(resp.json())
        assert "set use_peft=false" in detail
        assert "train_on_gpu=false" not in detail

        resp = await client.post(
            "/api/v1/jobs/training/validate", json={**body, "use_peft": False}
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_eval_rejects_gpu_over_budget(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An eval that serves its model in-pod needs a GPU; with none in the budget the
        # validate tool must say so instead of the create call failing at Confirm.
        from amortized.core import jobs as jobs_mod

        monkeypatch.setattr(jobs_mod._settings, "gpu_budget", 0)
        body = {"model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "eval_data_path": "./e.jsonl"}
        resp = await client.post("/api/v1/jobs/eval/validate", json=body)
        assert resp.status_code == 422
        detail = str(resp.json())
        assert "per-user GPU budget is 0" in detail
        # nproc_per_node can't go below 1, so a zero budget points at an endpoint instead.
        assert "evaluate through an endpoint" in detail
        assert "reduce nproc_per_node" not in detail

    @pytest.mark.asyncio
    async def test_training_rejects_parent_and_data_run_together(
        self, client: httpx.AsyncClient
    ) -> None:
        # Both set once trained on the parent's full SDG output, held-out rows included.
        # The agent-facing validate tool and create both 422 and name the one to keep.
        body = {
            **TRAINING_BODY,
            "data_path": "",
            "parent_job_id": "abc-123",
            "data_run_id": "d" * 32,
        }
        for path in ("/api/v1/jobs/training/validate", "/api/v1/jobs/training"):
            resp = await client.post(path, json=body)
            assert resp.status_code == 422
            detail = str(resp.json())
            assert "keep data_run_id" in detail
            assert "remove parent_job_id" in detail

    @pytest.mark.asyncio
    async def test_retry_legacy_training_with_both_sources_keeps_data_run(
        self, client: httpx.AsyncClient
    ) -> None:
        # A training job stored before both data sources were rejected carries a parent
        # and a data_run_id. Retrying it must not hit a 422 the caller can't act on (the
        # stored parent wins over overrides): the retry keeps data_run_id, drops the parent.
        created = await _create_training(client, data_run_id="d" * 32, data_path="")
        job_id = created.json()["id"]
        import amortized.db.connection as _db_conn

        async with _db_conn._pool.acquire() as conn:
            await conn.execute(
                "UPDATE jobs SET status = 'failed', parent_job_id = 'legacy-sdg' WHERE id = $1",
                job_id,
            )
        resp = await client.post(f"/api/v1/jobs/{job_id}/retry")
        assert resp.status_code == 201, resp.text
        new_job = resp.json()
        assert new_job["config"]["data_run_id"] == "d" * 32
        assert not new_job.get("parent_job_id")

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


class TestEvalConfirmCardFacts:
    @pytest.mark.asyncio
    async def test_validate_eval_returns_record_count(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Like training, the eval card shows the dataset's record count, resolved by the
        # backend from the run the job will score.
        import amortized.api.jobs as jobs_mod

        seen: list[str] = []

        async def _count(run_id: str) -> int:
            seen.append(run_id)
            return 150

        monkeypatch.setattr(jobs_mod, "_dataset_record_count", _count)
        run_id = "d" * 32
        body = {
            "model_name_or_path": "Qwen/Qwen3.5-0.8B",
            "judge": {"base_url": "http://x/v1", "model": "gpt-oss"},
            "rubric": [{"name": "acc", "description": "Is the answer correct?"}],
            "eval_data_run_id": run_id,
        }
        resp = await client.post("/api/v1/jobs/eval/validate", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["data_record_count"] == 150
        assert seen == [run_id]
        assert "data_record_count" not in data["config"]

    @pytest.mark.asyncio
    async def test_validate_eval_record_count_prefers_eval_data_run_over_parent(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With BOTH a parent and eval_data_run_id set, the eval job scores eval_data_run_id
        # (e.g. a held-out split; the parent stays lineage only), so the card counts it.
        import amortized.api.jobs as jobs_mod
        import amortized.db.connection as _db_conn
        from amortized.db.repository import Repository

        sdg = await _create_sdg(client)
        parent_id = sdg.json()["id"]
        async with _db_conn._pool.acquire() as conn:
            await Repository(conn).update_job(
                parent_id, status="succeeded", mlflow_run_id="parent-run-xyz"
            )

        seen: list[str] = []

        async def _count(run_id: str) -> int:
            seen.append(run_id)
            return 150

        monkeypatch.setattr(jobs_mod, "_dataset_record_count", _count)
        eval_run = "e" * 32
        body = {
            "model_name_or_path": "Qwen/Qwen3.5-0.8B",
            "judge": {"base_url": "http://x/v1", "model": "gpt-oss"},
            "rubric": [{"name": "acc", "description": "Is the answer correct?"}],
            "parent_job_id": parent_id,
            "eval_data_run_id": eval_run,
        }
        resp = await client.post("/api/v1/jobs/eval/validate", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json()["data_record_count"] == 150
        # eval_data_run_id wins over the parent's run (matches resolve_parent_artifacts).
        assert seen == [eval_run]


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
        assert data["gpu_budget"] == main._settings.gpu_budget


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


class TestEvalEndpointBaseUrl:
    @pytest.mark.asyncio
    async def test_validate_eval_rejects_unknown_base_url(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "gateway" is neither a URL nor a configured provider: the eval pod called it as
        # a URL and every sample errored. The validate tool now 422s with usable names.
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        body = {
            "eval_data_path": "./e.jsonl",
            "endpoint": {"base_url": "gateway", "model": "gpt-4o-mini"},
            "judge": {"base_url": "gateway", "model": "gpt-4o-mini"},
            "rubric": [{"name": "relevance", "description": "stays on topic"}],
        }
        resp = await client.post("/api/v1/jobs/eval/validate", json=body)
        assert resp.status_code == 422
        detail = str(resp.json())
        assert "endpoint.base_url 'gateway' is neither" in detail
        assert "judge.base_url 'gateway' is neither" in detail
        assert "openai" in detail

    @pytest.mark.asyncio
    async def test_validate_eval_accepts_provider_name_and_url(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        body = {
            "eval_data_run_id": "e" * 32,
            "endpoint": {"base_url": "openai", "model": "gpt-4o-mini"},
            "judge": {"base_url": "https://llm.example.com/v1", "model": "judge-model"},
            "rubric": [{"name": "relevance", "description": "stays on topic"}],
        }
        resp = await client.post("/api/v1/jobs/eval/validate", json=body)
        assert resp.status_code == 200, resp.json()
