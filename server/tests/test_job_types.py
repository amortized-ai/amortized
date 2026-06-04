"""Tests for job type registry, universal endpoint, and schema validation."""

import os

import httpx
import pytest

from amortized.core.job_types import (
    UnknownJobTypeError,
    get_schema,
    list_job_types,
    validate_config,
)
from amortized.main import app


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


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


class TestJobTypeRegistry:
    def test_list_job_types(self) -> None:
        types = list_job_types()
        assert len(types) == 4
        type_names = [t["type"] for t in types]
        assert "training" in type_names
        assert "sdg" in type_names
        assert "inference" in type_names
        assert "eval" in type_names

    def test_get_schema_training(self) -> None:
        schema = get_schema("training")
        assert schema["title"] == "TrainingJobConfig"
        assert "model_path" in schema["properties"]
        assert "model_path" in schema["required"]

    def test_get_schema_sdg(self) -> None:
        schema = get_schema("sdg")
        assert schema["title"] == "SDGJobConfig"
        assert "flow_id" in schema["properties"]

    def test_get_schema_unknown_type(self) -> None:
        with pytest.raises(UnknownJobTypeError):
            get_schema("unknown")

    def test_validate_config_valid_training(self) -> None:
        errors = validate_config("training", {
            "model_path": "test/model",
            "data_path": "./data.jsonl",
            "ckpt_output_dir": "./out",
        })
        assert errors == []

    def test_validate_config_missing_required(self) -> None:
        errors = validate_config("training", {"model_path": "test"})
        assert len(errors) > 0
        assert any("data_path" in e for e in errors)

    def test_validate_config_invalid_type(self) -> None:
        errors = validate_config("training", {
            "model_path": "test",
            "data_path": "test",
            "ckpt_output_dir": "test",
            "lora_r": "not-an-int",
        })
        assert len(errors) > 0

    def test_validate_config_valid_sdg(self) -> None:
        errors = validate_config("sdg", {
            "flow_id": "knowledge-qa",
            "dataset_path": "./docs.jsonl",
            "model": "openai/gpt-4o",
        })
        assert errors == []

    def test_get_schema_inference(self) -> None:
        schema = get_schema("inference")
        assert schema["title"] == "InferenceJobConfig"
        assert "model_path" in schema["properties"]
        assert "input_data" in schema["required"]
        assert "output_path" in schema["required"]

    def test_get_schema_eval(self) -> None:
        schema = get_schema("eval")
        assert schema["title"] == "EvalJobConfig"
        assert "model" in schema["required"]
        assert "judge_model" in schema["required"]
        assert "dataset" in schema["required"]

    def test_validate_config_valid_inference(self) -> None:
        errors = validate_config("inference", {
            "model_path": "test/model",
            "input_data": "./input.jsonl",
            "output_path": "./output.jsonl",
        })
        assert errors == []

    def test_validate_config_valid_eval(self) -> None:
        errors = validate_config("eval", {
            "model": "test/model",
            "judge_model": "openai/gpt-4o",
            "dataset": "./eval_data.jsonl",
        })
        assert errors == []

    def test_validate_config_inference_missing_required(self) -> None:
        errors = validate_config("inference", {"model_path": "test"})
        assert len(errors) > 0
        assert any("input_data" in e for e in errors)

    def test_validate_config_eval_missing_required(self) -> None:
        errors = validate_config("eval", {"model": "test"})
        assert len(errors) > 0
        assert any("judge_model" in e for e in errors)

    def test_validate_config_unknown_type(self) -> None:
        with pytest.raises(UnknownJobTypeError):
            validate_config("unknown", {})


class TestUniversalJobEndpoint:
    @pytest.mark.asyncio
    async def test_create_training_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "training",
            "config": {
                "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "./outputs",
            },
            "dry_run": False,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "training"
        assert data["status"] == "queued"
        assert data["config"]["model_path"] == "Qwen/Qwen2.5-1.5B-Instruct"

    @pytest.mark.asyncio
    async def test_create_sdg_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "sdg",
            "config": {
                "flow_id": "knowledge-qa",
                "dataset_path": "./docs.jsonl",
                "model": "openai/gpt-4o",
            },
            "dry_run": False,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "sdg"
        assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_create_inference_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "inference",
            "config": {
                "model_path": "test/model",
                "input_data": "./input.jsonl",
                "output_path": "./output.jsonl",
            },
            "dry_run": False,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "inference"
        assert data["status"] == "queued"
        assert data["config"]["model_path"] == "test/model"

    @pytest.mark.asyncio
    async def test_create_eval_job(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "eval",
            "config": {
                "model": "test/model",
                "judge_model": "openai/gpt-4o",
                "dataset": "./eval_data.jsonl",
            },
            "dry_run": False,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "eval"
        assert data["status"] == "queued"
        assert data["config"]["judge_model"] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_create_job_with_metadata(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "training",
            "config": {
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
            },
            "metadata": {"owner": "test-user", "experiment": "run-42"},
            "dry_run": False,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["metadata"]["owner"] == "test-user"
        assert data["metadata"]["experiment"] == "run-42"

    @pytest.mark.asyncio
    async def test_create_job_unknown_type(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "unknown",
            "config": {},
            "dry_run": False,
        })
        assert response.status_code == 400
        assert "Unknown job type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_job_validation_error(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "training",
            "config": {"model_path": "test"},
            "dry_run": False,
        })
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert isinstance(errors, list)
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_create_job_invalid_field_type(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "training",
            "config": {
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
                "num_epochs": "not-a-number",
            },
            "dry_run": False,
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_universal_job_visible_via_list(self, client: httpx.AsyncClient) -> None:
        await client.post("/api/v1/jobs", json={
            "type": "training",
            "config": {
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
            },
            "dry_run": False,
        })
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 200
        assert len(response.json()) == 1

    @pytest.mark.asyncio
    async def test_dry_run_returns_preview(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "training",
            "config": {
                "model_path": "test",
                "data_path": "test",
                "ckpt_output_dir": "test",
            },
        })
        assert response.status_code == 201
        data = response.json()
        assert data["dry_run"] is True
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["type"] == "training"

    @pytest.mark.asyncio
    async def test_dry_run_with_errors(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs", json={
            "type": "training",
            "config": {"model_path": "test"},
        })
        assert response.status_code == 201
        data = response.json()
        assert data["dry_run"] is True
        assert data["valid"] is False
        assert len(data["errors"]) > 0


class TestJobTypesEndpoints:
    @pytest.mark.asyncio
    async def test_list_job_types(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/job-types")
        assert response.status_code == 200
        types = response.json()
        assert len(types) == 4
        type_names = [t["type"] for t in types]
        assert "training" in type_names
        assert "sdg" in type_names
        assert "inference" in type_names
        assert "eval" in type_names

    @pytest.mark.asyncio
    async def test_get_training_schema(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/job-types/training/schema")
        assert response.status_code == 200
        schema = response.json()
        assert schema["title"] == "TrainingJobConfig"
        assert "model_path" in schema["properties"]

    @pytest.mark.asyncio
    async def test_get_sdg_schema(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/job-types/sdg/schema")
        assert response.status_code == 200
        schema = response.json()
        assert schema["title"] == "SDGJobConfig"

    @pytest.mark.asyncio
    async def test_get_inference_schema(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/job-types/inference/schema")
        assert response.status_code == 200
        schema = response.json()
        assert schema["title"] == "InferenceJobConfig"

    @pytest.mark.asyncio
    async def test_get_eval_schema(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/job-types/eval/schema")
        assert response.status_code == 200
        schema = response.json()
        assert schema["title"] == "EvalJobConfig"

    @pytest.mark.asyncio
    async def test_get_unknown_schema(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/job-types/unknown/schema")
        assert response.status_code == 404


class TestOldEndpointsBackwardCompat:
    @pytest.mark.asyncio
    async def test_old_training_endpoint_still_works(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs/training", json={
            "model_path": "test",
            "data_path": "test",
            "ckpt_output_dir": "test",
        })
        assert response.status_code == 201
        assert response.json()["type"] == "training"

    @pytest.mark.asyncio
    async def test_old_sdg_endpoint_still_works(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/jobs/sdg", json={
            "flow_id": "test",
            "dataset_path": "test",
            "model": "openai/gpt-4o",
        })
        assert response.status_code == 201
        assert response.json()["type"] == "sdg"

    @pytest.mark.asyncio
    async def test_old_and_universal_jobs_coexist(self, client: httpx.AsyncClient) -> None:
        await client.post("/api/v1/jobs/training", json={
            "model_path": "t1",
            "data_path": "t1",
            "ckpt_output_dir": "t1",
        })
        await client.post("/api/v1/jobs", json={
            "type": "training",
            "config": {
                "model_path": "t2",
                "data_path": "t2",
                "ckpt_output_dir": "t2",
            },
            "dry_run": False,
        })
        response = await client.get("/api/v1/jobs")
        assert len(response.json()) == 2
