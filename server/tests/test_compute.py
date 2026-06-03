"""Tests for compute backends, registry, and API endpoints."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from amortized.backends import BackendHandle, Capability, JobSpec
from amortized.backends.local import LocalBackend
from amortized.backends.ssh import SSHBackend
from amortized.core.compute import get_backend, list_backends, register_backend, reset
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


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    reset()
    register_backend(LocalBackend())
    yield  # type: ignore[misc]
    reset()


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


class TestCapability:
    def test_capability_values(self) -> None:
        assert Capability.GPU == "gpu"
        assert Capability.MULTI_NODE == "multi_node"
        assert Capability.LOG_STREAM == "log_stream"
        assert Capability.STOP == "stop"
        assert Capability.RESUME == "resume"


class TestJobSpec:
    def test_defaults(self) -> None:
        spec = JobSpec(job_id="j1", command=["echo", "hi"])
        assert spec.env == {}
        assert spec.work_dir == "."
        assert spec.image is None
        assert spec.timeout is None

    def test_full(self) -> None:
        spec = JobSpec(
            job_id="j1",
            command=["python", "train.py"],
            env={"CUDA_VISIBLE_DEVICES": "0"},
            work_dir="/tmp/work",
            image="train:latest",
            timeout=3600,
        )
        assert spec.env["CUDA_VISIBLE_DEVICES"] == "0"
        assert spec.timeout == 3600


class TestBackendRegistry:
    def test_register_and_get(self) -> None:
        reset()
        backend = LocalBackend()
        register_backend(backend)
        assert get_backend("local") is backend

    def test_get_unknown_raises(self) -> None:
        reset()
        with pytest.raises(KeyError, match="Unknown compute backend"):
            get_backend("nonexistent")

    def test_list_backends(self) -> None:
        reset()
        register_backend(LocalBackend())
        backends = list_backends()
        assert len(backends) == 1
        assert backends[0]["name"] == "local"
        assert "log_stream" in backends[0]["capabilities"]
        assert "stop" in backends[0]["capabilities"]


class TestLocalBackend:
    def test_capabilities(self) -> None:
        backend = LocalBackend()
        caps = backend.capabilities()
        assert Capability.LOG_STREAM in caps
        assert Capability.STOP in caps
        assert Capability.GPU not in caps

    @pytest.mark.asyncio
    async def test_submit_and_status(self, tmp_path: object) -> None:
        backend = LocalBackend()
        work_dir = str(tmp_path) + "/job1"
        spec = JobSpec(
            job_id="test-job-1",
            command=["python", "-c", "print('hello')"],
            work_dir=work_dir,
        )

        handle = await backend.submit(spec)
        assert handle.backend_name == "local"
        assert handle.job_id == "test-job-1"
        assert handle.remote_pid is not None

        import asyncio

        await asyncio.sleep(0.5)

        status = await backend.status(handle)
        assert not status.running
        assert status.exit_code == 0

    @pytest.mark.asyncio
    async def test_submit_failing_command(self, tmp_path: object) -> None:
        backend = LocalBackend()
        work_dir = str(tmp_path) + "/job-fail"
        spec = JobSpec(
            job_id="fail-job",
            command=["python", "-c", "import sys; sys.exit(42)"],
            work_dir=work_dir,
        )

        handle = await backend.submit(spec)
        import asyncio

        await asyncio.sleep(0.5)

        status = await backend.status(handle)
        assert not status.running
        assert status.exit_code == 42

    @pytest.mark.asyncio
    async def test_cancel(self, tmp_path: object) -> None:
        backend = LocalBackend()
        work_dir = str(tmp_path) + "/job-cancel"
        spec = JobSpec(
            job_id="cancel-job",
            command=["python", "-c", "import time; time.sleep(60)"],
            work_dir=work_dir,
        )

        handle = await backend.submit(spec)
        status = await backend.status(handle)
        assert status.running

        await backend.cancel(handle)

        import asyncio

        await asyncio.sleep(0.2)
        status = await backend.status(handle)
        assert not status.running

    @pytest.mark.asyncio
    async def test_logs(self, tmp_path: object) -> None:
        backend = LocalBackend()
        work_dir = str(tmp_path) + "/job-logs"
        spec = JobSpec(
            job_id="log-job",
            command=["python", "-c", "print('line1'); print('line2')"],
            work_dir=work_dir,
        )

        handle = await backend.submit(spec)
        import asyncio

        await asyncio.sleep(0.5)

        lines = [line async for line in backend.logs(handle)]
        assert "line1" in lines
        assert "line2" in lines

    @pytest.mark.asyncio
    async def test_logs_no_dir(self) -> None:
        backend = LocalBackend()
        handle = BackendHandle(backend_name="local", job_id="x", remote_dir=None)
        lines = [line async for line in backend.logs(handle)]
        assert lines == []


class TestSSHBackend:
    def test_capabilities(self) -> None:
        backend = SSHBackend(host="example.com")
        caps = backend.capabilities()
        assert Capability.GPU in caps
        assert Capability.LOG_STREAM in caps
        assert Capability.STOP in caps

    @pytest.mark.asyncio
    async def test_submit_mock(self) -> None:
        backend = SSHBackend(host="gpu-node", user="ml")

        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.stdout = "12345\n"
        mock_conn.run = AsyncMock(return_value=mock_result)
        mock_conn.close = MagicMock()

        with patch.object(backend, "_connect", return_value=mock_conn):
            spec = JobSpec(
                job_id="ssh-job-1",
                command=["python", "train.py"],
                env={"CUDA_VISIBLE_DEVICES": "0"},
            )
            handle = await backend.submit(spec)

        assert handle.backend_name == "ssh"
        assert handle.job_id == "ssh-job-1"
        assert handle.remote_pid == 12345
        assert handle.remote_dir == "~/amortized-jobs/ssh-job-1"
        assert mock_conn.run.call_count == 3

    @pytest.mark.asyncio
    async def test_status_running_mock(self) -> None:
        backend = SSHBackend(host="gpu-node")

        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.stdout = "alive"
        mock_conn.run = AsyncMock(return_value=mock_result)
        mock_conn.close = MagicMock()

        handle = BackendHandle(
            backend_name="ssh", job_id="j1", remote_pid=999
        )

        with patch.object(backend, "_connect", return_value=mock_conn):
            status = await backend.status(handle)

        assert status.running

    @pytest.mark.asyncio
    async def test_status_dead_mock(self) -> None:
        backend = SSHBackend(host="gpu-node")

        mock_conn = AsyncMock()
        alive_result = MagicMock()
        alive_result.stdout = "dead"
        exit_result = MagicMock()
        exit_result.stdout = "0"
        mock_conn.run = AsyncMock(side_effect=[alive_result, exit_result])
        mock_conn.close = MagicMock()

        handle = BackendHandle(
            backend_name="ssh", job_id="j1", remote_pid=999
        )

        with patch.object(backend, "_connect", return_value=mock_conn):
            status = await backend.status(handle)

        assert not status.running
        assert status.exit_code == 0

    @pytest.mark.asyncio
    async def test_cancel_mock(self) -> None:
        backend = SSHBackend(host="gpu-node")

        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock()
        mock_conn.close = MagicMock()

        handle = BackendHandle(
            backend_name="ssh", job_id="j1", remote_pid=999
        )

        with patch.object(backend, "_connect", return_value=mock_conn):
            await backend.cancel(handle)

        mock_conn.run.assert_called_once_with("kill 999 2>/dev/null || true")

    @pytest.mark.asyncio
    async def test_cancel_no_pid(self) -> None:
        backend = SSHBackend(host="gpu-node")
        handle = BackendHandle(backend_name="ssh", job_id="j1", remote_pid=None)
        await backend.cancel(handle)

    @pytest.mark.asyncio
    async def test_logs_mock(self) -> None:
        backend = SSHBackend(host="gpu-node")

        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.stdout = "training started\nloss=0.5\n"
        mock_conn.run = AsyncMock(return_value=mock_result)
        mock_conn.close = MagicMock()

        handle = BackendHandle(
            backend_name="ssh",
            job_id="j1",
            remote_pid=999,
            remote_dir="~/amortized-jobs/j1",
        )

        with patch.object(backend, "_connect", return_value=mock_conn):
            lines = [line async for line in backend.logs(handle)]

        assert lines == ["training started", "loss=0.5"]

    @pytest.mark.asyncio
    async def test_status_no_pid(self) -> None:
        backend = SSHBackend(host="gpu-node")
        handle = BackendHandle(backend_name="ssh", job_id="j1", remote_pid=None)
        status = await backend.status(handle)
        assert not status.running
        assert status.error == "No PID recorded"

    @pytest.mark.asyncio
    async def test_logs_no_dir(self) -> None:
        backend = SSHBackend(host="gpu-node")
        handle = BackendHandle(backend_name="ssh", job_id="j1", remote_dir=None)
        lines = [line async for line in backend.logs(handle)]
        assert lines == []


class TestComputeAPI:
    @pytest.mark.asyncio
    async def test_list_backends(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/compute")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        names = [b["name"] for b in data]
        assert "local" in names

    @pytest.mark.asyncio
    async def test_backend_status(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/compute/local/status")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "local"
        assert data["healthy"] is True
        assert "log_stream" in data["capabilities"]

    @pytest.mark.asyncio
    async def test_backend_status_not_found(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/compute/nonexistent/status")
        assert response.status_code == 404
