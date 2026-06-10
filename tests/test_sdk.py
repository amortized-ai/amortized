"""Tests for the Amortized Python SDK client."""

import os

import httpx
import pytest

from amortized.main import app
from amortized.sdk import Client, Job, SyncClient


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
async def sdk_client() -> Client:  # type: ignore[misc]
    transport = httpx.ASGITransport(app=app)
    client = Client(base_url="http://test")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=transport, base_url="http://test")

    from amortized.db import init_db

    await init_db()
    yield client  # type: ignore[misc]
    await client.close()


class TestAutoDiscovery:
    def test_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AMORTIZED_API_URL", "http://myhost:9000")
        from amortized.sdk.client import _discover_url

        assert _discover_url() == "http://myhost:9000"

    def test_env_var_trailing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AMORTIZED_API_URL", "http://myhost:9000/")
        from amortized.sdk.client import _discover_url

        assert _discover_url() == "http://myhost:9000"

    def test_config_yaml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
        monkeypatch.delenv("AMORTIZED_API_URL", raising=False)
        config_dir = tmp_path / ".amortized"  # type: ignore[operator]
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("api_url: http://configured:8080\n")
        monkeypatch.setattr("amortized.sdk.client.Path.home", lambda: tmp_path)

        from amortized.sdk.client import _discover_url

        assert _discover_url() == "http://configured:8080"

    def test_default_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
        monkeypatch.delenv("AMORTIZED_API_URL", raising=False)
        monkeypatch.setattr("amortized.sdk.client.Path.home", lambda: tmp_path)

        from amortized.sdk.client import _discover_url

        assert _discover_url() == "http://localhost:8000"

    def test_client_uses_discovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AMORTIZED_API_URL", "http://discovered:1234")
        c = Client()
        assert c.base_url == "http://discovered:1234"

    def test_client_explicit_url_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AMORTIZED_API_URL", "http://env:1234")
        c = Client(base_url="http://explicit:5678")
        assert c.base_url == "http://explicit:5678"


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submit_training_job(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "data_path": "./data.jsonl",
                "output_dir": "./outputs",
            },
        )
        assert isinstance(job, Job)
        assert job.type == "training"
        assert job.status == "queued"
        assert job.id

    @pytest.mark.asyncio
    async def test_submit_with_metadata(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "data_path": "./data.jsonl",
                "output_dir": "./outputs",
            },
            metadata={"team": "ml-infra"},
        )
        assert job.metadata["team"] == "ml-infra"
        assert "backend" not in job.metadata

    @pytest.mark.asyncio
    async def test_submit_with_compute(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "data_path": "./data.jsonl",
                "output_dir": "./outputs",
            },
            compute={"backend": "local", "gpus": 1},
        )
        assert job.status == "queued"


class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_existing_job(self, sdk_client: Client) -> None:
        created = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test-model",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        fetched = await sdk_client.get_job(created.id)
        assert fetched.id == created.id
        assert fetched.type == "training"

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, sdk_client: Client) -> None:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await sdk_client.get_job("nonexistent-id")
        assert exc_info.value.response.status_code == 404


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_empty(self, sdk_client: Client) -> None:
        jobs = await sdk_client.list_jobs()
        assert jobs == []

    @pytest.mark.asyncio
    async def test_list_after_submit(self, sdk_client: Client) -> None:
        await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        jobs = await sdk_client.list_jobs()
        assert len(jobs) == 1
        assert isinstance(jobs[0], Job)

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, sdk_client: Client) -> None:
        await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        queued = await sdk_client.list_jobs(status="queued")
        assert len(queued) == 1
        running = await sdk_client.list_jobs(status="running")
        assert len(running) == 0


class TestCancelJob:
    @pytest.mark.asyncio
    async def test_cancel(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        cancelled = await sdk_client.cancel_job(job.id)
        assert cancelled.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_via_job_method(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        cancelled = await job.cancel()
        assert cancelled.status == "cancelled"


class TestJobWait:
    @pytest.mark.asyncio
    async def test_wait_already_terminal(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        await job.cancel()
        await job.refresh()
        result = await job.wait(poll_interval=0.1)
        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_job_repr(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        r = repr(job)
        assert "Job(" in r
        assert job.id in r
        assert "training" in r


class TestJobRaw:
    @pytest.mark.asyncio
    async def test_raw_data(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        raw = job.raw
        assert raw["id"] == job.id
        assert raw["type"] == "training"
        assert "created_at" in raw


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, sdk_client: Client) -> None:
        result = await sdk_client.health()
        assert result["status"] == "ok"


class TestContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        async with Client(base_url="http://test") as c:
            assert c.base_url == "http://test"


class TestListEndpoints:
    @pytest.mark.asyncio
    async def test_list_job_types(self, sdk_client: Client) -> None:
        result = await sdk_client.list_job_types()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_recipes(self, sdk_client: Client) -> None:
        result = await sdk_client.list_recipes()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_artifacts(self, sdk_client: Client) -> None:
        result = await sdk_client.list_artifacts()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_backends(self, sdk_client: Client) -> None:
        result = await sdk_client.list_backends()
        assert isinstance(result, list)


class TestArtifactRef:
    @pytest.mark.asyncio
    async def test_artifact_ref_format(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        ref = job.artifact_ref("model")
        assert ref == f"artifact:{job.id}/model"

    @pytest.mark.asyncio
    async def test_artifact_ref_with_slash_in_name(self, sdk_client: Client) -> None:
        job = await sdk_client.submit(
            type="training",
            config={
                "algorithm": "sft",
                "model_name_or_path": "test",
                "data_path": "./data.jsonl",
                "output_dir": "./out",
            },
        )
        ref = job.artifact_ref("checkpoint-100/adapter_model")
        assert ref == f"artifact:{job.id}/checkpoint-100/adapter_model"


class TestUpload:
    @pytest.mark.asyncio
    async def test_upload_registers_artifact(self, sdk_client: Client, tmp_path: object) -> None:
        test_file = tmp_path / "data.jsonl"  # type: ignore[operator]
        test_file.write_text('{"text": "hello"}\n')

        result = await sdk_client.upload(str(test_file), artifact_type="dataset")
        assert result["name"] == "data.jsonl"
        assert result["artifact_type"] == "dataset"

    @pytest.mark.asyncio
    async def test_upload_with_custom_name(self, sdk_client: Client, tmp_path: object) -> None:
        test_file = tmp_path / "data.jsonl"  # type: ignore[operator]
        test_file.write_text('{"text": "hello"}\n')

        result = await sdk_client.upload(str(test_file), artifact_type="dataset", name="my-dataset")
        assert result["name"] == "my-dataset"


class TestSyncClient:
    def test_sync_client_health(self) -> None:
        transport = httpx.ASGITransport(app=app)
        sync = SyncClient(base_url="http://test")
        import asyncio

        asyncio.run(sync._async._http.aclose())
        sync._async._http = httpx.AsyncClient(transport=transport, base_url="http://test")

        from amortized.db import init_db

        asyncio.run(init_db())
        result = sync.health()
        assert result["status"] == "ok"
        sync.close()

    def test_sync_client_context_manager(self) -> None:
        with SyncClient(base_url="http://test") as client:
            assert client.base_url == "http://test"


class TestSubmitRecipe:
    @pytest.mark.asyncio
    async def test_submit_nonexistent_recipe(self, sdk_client: Client) -> None:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await sdk_client.submit_recipe("nonexistent/recipe")
        assert exc_info.value.response.status_code == 404
