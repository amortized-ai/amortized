"""Tests for issues #43-#49: auth, config/validate, capability-gating,
slurm, container fallbacks, resume, pre-signed URLs."""

import os

import httpx
import pytest

from amortized.backends import Capability
from amortized.backends.slurm import (
    FRONTIER,
    PERLMUTTER,
    POLARIS,
    SlurmBackend,
    SlurmProfile,
)
from amortized.core.compute import MissingCapabilityError, check_capabilities
from amortized.core.storage import LocalStorage, reset_storage
from amortized.main import app
from amortized.models import ConfigValidateRequest, ConfigValidateResponse, ResumeRequest


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db as db_mod
    import amortized.db.connection as db_conn_mod

    db_path = str(tmp_path) + "/test.db"
    os.environ["AMORTIZED_DB_PATH"] = db_path
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    os.environ.pop("AMORTIZED_API_KEY", None)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    from amortized.db import init_db

    await init_db()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ---- #46: Config validate endpoint ----


class TestConfigValidate:
    @pytest.mark.asyncio
    async def test_valid_training_config(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/config/validate",
            json={
                "type": "training",
                "config": {
                    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
                    "data_path": "./data.jsonl",
                    "ckpt_output_dir": "./out",
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["errors"] == []

    @pytest.mark.asyncio
    async def test_invalid_job_type(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/config/validate",
            json={"type": "nonexistent", "config": {}},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/config/validate",
            json={"type": "training", "config": {}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    @pytest.mark.asyncio
    async def test_valid_sdg_config(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/config/validate",
            json={
                "type": "sdg",
                "config": {"model": "openai/gpt-4o", "num_samples": 10},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True


# ---- #43: Auth middleware ----


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_health_skips_auth(self) -> None:
        import amortized.main as main_mod

        original = main_mod._settings
        try:
            main_mod._settings = type(original)(api_key="test-key-123")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.get("/api/v1/health")
            assert resp.status_code == 200
        finally:
            main_mod._settings = original

    @pytest.mark.asyncio
    async def test_no_auth_when_key_empty(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_rejects_missing_token(self) -> None:
        import amortized.main as main_mod

        original = main_mod._settings
        try:
            main_mod._settings = type(original)(api_key="test-key-123")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.post(
                    "/api/v1/config/validate",
                    json={"type": "training", "config": {}},
                )
            assert resp.status_code == 401
            assert resp.json()["code"] == "unauthorized"
        finally:
            main_mod._settings = original

    @pytest.mark.asyncio
    async def test_accepts_valid_token(self, client: httpx.AsyncClient) -> None:
        import amortized.main as main_mod

        original = main_mod._settings
        try:
            main_mod._settings = type(original)(api_key="test-key-123")
            resp = await client.post(
                "/api/v1/config/validate",
                json={"type": "training", "config": {}},
                headers={"Authorization": "Bearer test-key-123"},
            )
            assert resp.status_code == 200
        finally:
            main_mod._settings = original

    @pytest.mark.asyncio
    async def test_rejects_wrong_token(self) -> None:
        import amortized.main as main_mod

        original = main_mod._settings
        try:
            main_mod._settings = type(original)(api_key="test-key-123")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.post(
                    "/api/v1/config/validate",
                    json={"type": "training", "config": {}},
                    headers={"Authorization": "Bearer wrong-key"},
                )
            assert resp.status_code == 401
        finally:
            main_mod._settings = original


# ---- #44: Capability gating ----


class _MockBackend:
    name = "mock"

    def __init__(self, caps: set[Capability]) -> None:
        self._caps = caps

    def capabilities(self) -> set[Capability]:
        return self._caps


class TestCapabilityGating:
    def test_passes_when_capabilities_met(self) -> None:
        backend = _MockBackend({Capability.GPU, Capability.STOP})
        check_capabilities(backend, {Capability.GPU})

    def test_raises_when_capability_missing(self) -> None:
        backend = _MockBackend({Capability.STOP})
        with pytest.raises(MissingCapabilityError) as exc_info:
            check_capabilities(backend, {Capability.GPU})
        assert "gpu" in str(exc_info.value)
        assert exc_info.value.backend_name == "mock"

    def test_empty_required_always_passes(self) -> None:
        backend = _MockBackend(set())
        check_capabilities(backend, set())

    def test_multiple_missing(self) -> None:
        backend = _MockBackend(set())
        with pytest.raises(MissingCapabilityError) as exc_info:
            check_capabilities(backend, {Capability.GPU, Capability.RESUME})
        assert exc_info.value.missing == {Capability.GPU, Capability.RESUME}


# ---- #45: Slurm backend stub ----


class TestSlurmBackend:
    def test_profiles_exist(self) -> None:
        assert FRONTIER.name == "frontier"
        assert PERLMUTTER.name == "perlmutter"
        assert POLARIS.name == "polaris"

    def test_slurm_capabilities(self) -> None:
        backend = SlurmBackend(host="example.com")
        caps = backend.capabilities()
        assert Capability.GPU in caps
        assert Capability.LOG_STREAM in caps
        assert Capability.STOP in caps

    def test_slurm_profile_defaults(self) -> None:
        profile = SlurmProfile(name="test", partition="debug")
        assert profile.scheduler == "slurm"
        assert profile.account == ""
        assert profile.module_loads == []

    def test_sbatch_script_generation(self) -> None:
        from amortized.backends import JobSpec, Resources

        backend = SlurmBackend(
            host="example.com",
            profile=SlurmProfile(
                name="test",
                partition="gpu",
                account="myacct",
                module_loads=["cuda", "python"],
            ),
        )
        spec = JobSpec(
            job_id="abc12345-6789",
            command=["python", "train.py"],
            work_dir="/scratch/jobs/abc",
            resources=Resources(gpus=2),
        )
        script = backend._generate_sbatch_script(spec)
        assert "partition=gpu" in script
        assert "account=myacct" in script
        assert "gres:gpu:2" in script
        assert "module load cuda" in script
        assert "python train.py" in script

    def test_perlmutter_gpu_format(self) -> None:
        assert "a100" in PERLMUTTER.gpu_resource_format


# ---- #48: Resume endpoint ----


class TestResumeEndpoint:
    @pytest.mark.asyncio
    async def test_resume_nonexistent_job(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/api/v1/jobs/nonexistent/resume")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_resume_non_failed_job(self, client: httpx.AsyncClient) -> None:
        create_resp = await client.post(
            "/api/v1/jobs/training",
            json={
                "model_path": "test-model",
                "data_path": "./data.jsonl",
                "ckpt_output_dir": "/tmp/test_resume",
            },
        )
        assert create_resp.status_code == 201
        job_id = create_resp.json()["id"]

        resp = await client.post(f"/api/v1/jobs/{job_id}/resume")
        assert resp.status_code == 400
        assert "failed" in resp.json()["message"].lower()

    @pytest.mark.asyncio
    async def test_resume_request_model(self) -> None:
        req = ResumeRequest(checkpoint_id="ckpt-123")
        assert req.checkpoint_id == "ckpt-123"

    @pytest.mark.asyncio
    async def test_resume_request_no_checkpoint(self) -> None:
        req = ResumeRequest()
        assert req.checkpoint_id is None


# ---- #49: Storage backend ----


class TestStorageBackends:
    def test_local_storage_raises(self) -> None:
        storage = LocalStorage()
        with pytest.raises(NotImplementedError):
            storage.generate_upload_url("key", "text/plain")
        with pytest.raises(NotImplementedError):
            storage.generate_download_url("key")

    @pytest.mark.asyncio
    async def test_upload_url_rejects_local(self, client: httpx.AsyncClient) -> None:
        reset_storage()
        resp = await client.post(
            "/api/v1/artifacts/upload-url",
            json={"name": "test.bin", "content_type": "application/octet-stream"},
        )
        assert resp.status_code == 400
        body = resp.json()
        msg = body.get("detail") or body.get("message", "")
        assert "local" in msg.lower()


# ---- Model validation ----


class TestNewModels:
    def test_config_validate_request(self) -> None:
        req = ConfigValidateRequest(type="training", config={"model_path": "x"})
        assert req.type == "training"

    def test_config_validate_response(self) -> None:
        resp = ConfigValidateResponse(valid=True)
        assert resp.valid is True
        assert resp.errors == []
        assert resp.warnings == []

    def test_backend_handle_scheduler_id(self) -> None:
        from amortized.backends import BackendHandle

        handle = BackendHandle(
            backend_name="slurm",
            job_id="test-123",
            scheduler_id="12345",
        )
        assert handle.scheduler_id == "12345"
