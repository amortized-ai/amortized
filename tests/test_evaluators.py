"""Tests for evaluator and evaluation CRUD API endpoints."""

import os

import httpx
import pytest

from amortized.main import app


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    """Use a temporary database for each test."""
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


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


def _evaluator_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "Test Evaluator",
        "description": "A test evaluator",
        "type": "llm",
        "prompt": "Evaluate: {{response}}",
        "judgment_type": "bool",
        "response_format": "json",
        "variables": ["response"],
    }
    base.update(overrides)
    return base


def _evaluation_payload(evaluator_id: str, dataset: str = "/data/test.jsonl") -> dict[str, str]:
    return {
        "evaluator_id": evaluator_id,
        "dataset": dataset,
        "model_override": "openai/gpt-4o-mini",
    }


class TestCreateEvaluator:
    @pytest.mark.asyncio
    async def test_create_evaluator(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/api/v1/evaluators", json=_evaluator_payload())
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Evaluator"
        assert data["type"] == "llm"
        assert data["judgment_type"] == "bool"
        assert data["id"]
        assert data["created_at"]

    @pytest.mark.asyncio
    async def test_create_evaluator_validation_error(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/api/v1/evaluators", json={"description": "missing name"})
        assert resp.status_code == 422


class TestListEvaluators:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/evaluators")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_list_after_create(self, client: httpx.AsyncClient) -> None:
        await client.post("/api/v1/evaluators", json=_evaluator_payload(name="Eval A"))
        await client.post("/api/v1/evaluators", json=_evaluator_payload(name="Eval B"))
        resp = await client.get("/api/v1/evaluators")
        assert resp.status_code == 200
        names = [e["name"] for e in resp.json()]
        assert "Eval A" in names
        assert "Eval B" in names


class TestGetEvaluator:
    @pytest.mark.asyncio
    async def test_get_existing(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/v1/evaluators", json=_evaluator_payload())
        eid = create_resp.json()["id"]

        resp = await client.get(f"/api/v1/evaluators/{eid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == eid
        assert resp.json()["name"] == "Test Evaluator"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/evaluators/nonexistent-id")
        assert resp.status_code == 404


class TestUpdateEvaluator:
    @pytest.mark.asyncio
    async def test_update_evaluator(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/v1/evaluators", json=_evaluator_payload())
        eid = create_resp.json()["id"]

        updated_payload = _evaluator_payload(
            name="Updated Evaluator",
            prompt="New prompt: {{response}}",
        )
        resp = await client.put(f"/api/v1/evaluators/{eid}", json=updated_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated Evaluator"
        assert data["prompt"] == "New prompt: {{response}}"
        assert data["id"] == eid

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, client: httpx.AsyncClient) -> None:
        resp = await client.put("/api/v1/evaluators/nonexistent-id", json=_evaluator_payload())
        assert resp.status_code == 404


class TestDeleteEvaluator:
    @pytest.mark.asyncio
    async def test_delete_evaluator(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/v1/evaluators", json=_evaluator_payload())
        eid = create_resp.json()["id"]

        resp = await client.delete(f"/api/v1/evaluators/{eid}")
        assert resp.status_code == 204

        get_resp = await client.get(f"/api/v1/evaluators/{eid}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, client: httpx.AsyncClient) -> None:
        resp = await client.delete("/api/v1/evaluators/nonexistent-id")
        assert resp.status_code == 404


class TestCreateEvaluation:
    @pytest.mark.asyncio
    async def test_create_evaluation(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/v1/evaluators", json=_evaluator_payload())
        eid = create_resp.json()["id"]

        resp = await client.post(
            "/api/v1/evaluations",
            json=_evaluation_payload(eid),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["evaluator_id"] == eid
        assert data["job_id"]
        assert data["id"]

    @pytest.mark.asyncio
    async def test_create_evaluation_nonexistent_evaluator(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/evaluations",
            json=_evaluation_payload("nonexistent"),
        )
        assert resp.status_code == 404


class TestListEvaluations:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/evaluations")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/v1/evaluators", json=_evaluator_payload())
        eid = create_resp.json()["id"]

        await client.post(
            "/api/v1/evaluations",
            json=_evaluation_payload(eid, "/data/a.jsonl"),
        )
        await client.post(
            "/api/v1/evaluations",
            json=_evaluation_payload(eid, "/data/b.jsonl"),
        )

        resp = await client.get("/api/v1/evaluations")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @pytest.mark.asyncio
    async def test_filter_by_evaluator_id(self, client: httpx.AsyncClient) -> None:
        r1 = await client.post("/api/v1/evaluators", json=_evaluator_payload(name="A"))
        r2 = await client.post("/api/v1/evaluators", json=_evaluator_payload(name="B"))
        eid1 = r1.json()["id"]
        eid2 = r2.json()["id"]

        await client.post(
            "/api/v1/evaluations",
            json=_evaluation_payload(eid1, "/data/a.jsonl"),
        )
        await client.post(
            "/api/v1/evaluations",
            json=_evaluation_payload(eid2, "/data/b.jsonl"),
        )

        resp = await client.get(f"/api/v1/evaluations?evaluator_id={eid1}")
        assert resp.status_code == 200
        evals = resp.json()
        assert len(evals) == 1
        assert evals[0]["evaluator_id"] == eid1


class TestGetEvaluation:
    @pytest.mark.asyncio
    async def test_get_existing(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post("/api/v1/evaluators", json=_evaluator_payload())
        eid = create_resp.json()["id"]

        eval_resp = await client.post(
            "/api/v1/evaluations",
            json=_evaluation_payload(eid),
        )
        eval_id = eval_resp.json()["id"]

        resp = await client.get(f"/api/v1/evaluations/{eval_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == eval_id
        assert resp.json()["evaluator_id"] == eid

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/evaluations/nonexistent-id")
        assert resp.status_code == 404


class TestDefaultEvaluatorsSeeded:
    @pytest.mark.asyncio
    async def test_default_evaluators_seeded(self, client: httpx.AsyncClient) -> None:
        from amortized.api.evaluators import seed_default_evaluators
        from amortized.db import get_db

        async for db in get_db():
            await seed_default_evaluators(db)

        resp = await client.get("/api/v1/evaluators")
        assert resp.status_code == 200
        evaluators = resp.json()
        names = {e["name"] for e in evaluators}
        assert "Safety" in names
        assert "Instruction Following" in names
        assert "Truthfulness" in names
        assert "Groundedness" in names
        assert "Code Quality" in names
        assert len(evaluators) >= 5

    @pytest.mark.asyncio
    async def test_seed_is_idempotent(self, client: httpx.AsyncClient) -> None:
        from amortized.api.evaluators import seed_default_evaluators
        from amortized.db import get_db

        async for db in get_db():
            await seed_default_evaluators(db)
            await seed_default_evaluators(db)

        resp = await client.get("/api/v1/evaluators")
        evaluators = resp.json()
        assert len(evaluators) == 5
