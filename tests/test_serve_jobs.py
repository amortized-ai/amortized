"""Tests for serve job API endpoints and builder."""

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


async def _insert_succeeded_training(client: httpx.AsyncClient, job_id: str) -> None:
    """Insert a fake succeeded training job row for serve to chain from."""
    import asyncpg

    conn = await asyncpg.connect(TEST_DATABASE_URL)
    try:
        await conn.execute(
            """INSERT INTO jobs (id, type, status, config, created_at, user_id,
               mlflow_run_id, k8s_namespace)
               VALUES ($1, 'training', 'succeeded', $2, now(), '', $3, 'test-ns')""",
            job_id,
            '{"model_name_or_path": "Qwen/Qwen3.5-2B", "algorithm": "lora_sft",'
            ' "model_display_name": "mdl-test-1234"}',
            "b" * 32,
        )
    finally:
        await conn.close()


class TestCreateServeJob:
    @pytest.mark.asyncio
    async def test_create_serve_job_from_model_name(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/serve",
            json={"model_name_or_path": "Qwen/Qwen3.5-2B"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "serve"
        assert data["status"] == "queued"
        assert data["config"]["model_name_or_path"] == "Qwen/Qwen3.5-2B"

    @pytest.mark.asyncio
    async def test_create_serve_job_from_training_job(
        self, client: httpx.AsyncClient
    ) -> None:
        training_id = "11111111-1111-1111-1111-111111111111"
        await _insert_succeeded_training(client, training_id)
        response = await client.post(
            "/api/v1/jobs/serve",
            json={"training_job_id": training_id},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "serve"
        # Lineage: serve job is a child of the training job
        assert data["parent_job_id"] == training_id
        assert data["config"]["training_job_id"] == training_id

    @pytest.mark.asyncio
    async def test_create_serve_job_requires_model_source(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post("/api/v1/jobs/serve", json={})
        assert response.status_code == 422
        assert "training_job_id" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_validate_serve_job_round_trip(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/jobs/serve/validate",
            json={"model_name_or_path": "Qwen/Qwen3.5-2B", "served_model_name": "my-model"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_type"] == "serve"
        assert data["config"]["served_model_name"] == "my-model"


class TestServeBuilder:
    @pytest.mark.asyncio
    async def test_build_named_model(self) -> None:
        from amortized.jobs import serve as serve_builder

        result = await serve_builder.build(
            {"id": "j1", "type": "serve"},
            {"model_name_or_path": "Qwen/Qwen3.5-2B"},
            {},
        )
        assert result.command[:2] == ["sh", "-c"]
        assert "vllm serve" in result.command[2]
        assert "Qwen/Qwen3.5-2B" in result.command[2]
        assert "--served-model-name Qwen/Qwen3.5-2B" in result.command[2]
        assert "--port 8000" in result.command[2]
        assert result.ports == {8000: 8000}
        assert result.resources.gpus == 1
        assert result.image == serve_builder.IMAGE
        assert result.resolved_config["served_model_name"] == "Qwen/Qwen3.5-2B"

    @pytest.mark.asyncio
    async def test_build_custom_port_and_gpus(self) -> None:
        from amortized.jobs import serve as serve_builder

        result = await serve_builder.build(
            {"id": "j1", "type": "serve"},
            {"model_name_or_path": "m", "port": 9000, "nproc_per_node": 2},
            {},
        )
        assert result.ports == {9000: 9000}
        assert result.resources.gpus == 2
        assert "--port 9000" in result.command[2]

    @pytest.mark.asyncio
    async def test_build_requires_model_source(self) -> None:
        from amortized.jobs.base import JobBuildError
        from amortized.jobs import serve as serve_builder

        with pytest.raises(JobBuildError):
            await serve_builder.build({"id": "j1", "type": "serve"}, {}, {})

    @pytest.mark.asyncio
    async def test_build_training_model_downloads_artifacts(self, monkeypatch) -> None:
        from amortized.jobs import serve as serve_builder

        training_job = {
            "id": "11111111-1111-1111-1111-111111111111",
            "type": "training",
            "status": "succeeded",
            "mlflow_run_id": "r" * 32,
            "config": {"model_name_or_path": "Qwen/Qwen3.5-2B"},
        }

        class FakeConn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class FakeRepo:
            async def get_job(self, job_id):
                return training_job

        class FakePool:
            def acquire(self):
                return FakeConn()

        import amortized.db.connection as db_conn
        import amortized.jobs.serve as serve_mod

        monkeypatch.setattr(serve_mod, "get_pool", lambda: FakePool(), raising=False)
        monkeypatch.setattr(db_conn, "get_pool", lambda: FakePool())

        class FakeRepoCls:
            def __call__(self, conn):
                return FakeRepo()

        import amortized.db.repository as repo_mod

        monkeypatch.setattr(repo_mod, "Repository", FakeRepoCls())

        result = await serve_builder.build(
            {"id": "j1", "type": "serve"},
            {"training_job_id": "11111111-1111-1111-1111-111111111111"},
            {},
        )
        # Pre-command downloads the HF export from MLflow
        assert any("mlflow artifacts download" in c for c in result.pre_commands)
        assert any("SERVE_MODEL_DIR=" in c for c in result.pre_commands)
        # Command serves the resolved checkpoint dir via the shell variable
        assert 'vllm serve "$SERVE_MODEL_DIR"' in result.command[2]
        # No MLflow available in tests — falls back to the registration-name pattern
        assert "--served-model-name Qwen3.5-2B-sft-11111111" in result.command[2]
        assert result.resolved_config["served_model_name"] == "Qwen3.5-2B-sft-11111111"

    @pytest.mark.asyncio
    async def test_build_rejects_non_succeeded_training(self, monkeypatch) -> None:
        from amortized.jobs.base import JobBuildError
        from amortized.jobs import serve as serve_builder

        training_job = {
            "id": "11111111-1111-1111-1111-111111111111",
            "type": "training",
            "status": "running",
            "mlflow_run_id": "",
            "config": {},
        }

        class FakeConn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class FakeRepo:
            async def get_job(self, job_id):
                return training_job

        class FakePool:
            def acquire(self):
                return FakeConn()

        import amortized.db.connection as db_conn
        import amortized.db.repository as repo_mod

        monkeypatch.setattr(db_conn, "get_pool", lambda: FakePool())

        class FakeRepoCls:
            def __call__(self, conn):
                return FakeRepo()

        monkeypatch.setattr(repo_mod, "Repository", FakeRepoCls())

        with pytest.raises(JobBuildError, match="succeeded"):
            await serve_builder.build(
                {"id": "j1", "type": "serve"},
                {"training_job_id": "11111111-1111-1111-1111-111111111111"},
                {},
            )


class TestServeEndpointSuggestions:
    @pytest.mark.asyncio
    async def test_suggestions_include_running_serve_jobs(
        self, client: httpx.AsyncClient, monkeypatch
    ) -> None:
        import asyncpg

        conn = await asyncpg.connect(TEST_DATABASE_URL)
        try:
            await conn.execute(
                """INSERT INTO jobs (id, type, status, config, created_at, user_id,
                   k8s_job_name, k8s_namespace)
                   VALUES ($1, 'serve', 'running', $2, now(), '', 'amortized-abc',
                   'test-jobs')""",
                "22222222-2222-2222-2222-222222222222",
                '{"served_model_name": "mdl-test-1234", "port": 8000,'
                ' "training_job_id": "11111111-1111-1111-1111-111111111111"}',
            )
        finally:
            await conn.close()

        class Fakehttpx:
            class AsyncClient:
                def __init__(self, timeout=None):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

                async def get(self, url):
                    class R:
                        status_code = 200

                    return R()

        import httpx as httpx_mod

        monkeypatch.setattr(httpx_mod, "AsyncClient", Fakehttpx.AsyncClient)

        response = await client.get("/api/v1/eval/endpoint-suggestions")
        assert response.status_code == 200
        data = response.json()
        serve_eps = data["serve_endpoints"]
        assert len(serve_eps) == 1
        ep = serve_eps[0]
        assert ep["job_id"] == "22222222-2222-2222-2222-222222222222"
        assert ep["model_name"] == "mdl-test-1234"
        assert ep["healthy"] is True
        assert ep["base_url"] == "http://amortized-abc.test-jobs.svc.cluster.local:8000/v1"
