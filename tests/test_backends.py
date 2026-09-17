"""Tests for compute backend data types."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amortized.backends import BackendHandle, BackendStatus, JobSpec, Resources
from amortized.backends.kubernetes import KubernetesBackend
from amortized.backends.ssh import SSHBackend
from amortized.core.compute import list_backends, register_backend, reset
from amortized.main import _load_backends


class TestResources:
    def test_defaults(self) -> None:
        r = Resources()
        assert r.gpus == 1
        assert r.gpu_type is None
        assert r.cpus is None
        assert r.memory_gb is None
        assert r.nodes == 1

    def test_custom_values(self) -> None:
        r = Resources(gpus=4, gpu_type="A100", cpus=32, memory_gb=256, nodes=2)
        assert r.gpus == 4
        assert r.gpu_type == "A100"
        assert r.cpus == 32
        assert r.memory_gb == 256
        assert r.nodes == 2


class TestJobSpec:
    def test_default_resources(self) -> None:
        spec = JobSpec(job_id="j1", command=["python", "run.py"])
        assert isinstance(spec.resources, Resources)
        assert spec.resources.gpus == 1
        assert spec.resources.nodes == 1

    def test_custom_resources(self) -> None:
        r = Resources(gpus=8, nodes=4)
        spec = JobSpec(job_id="j2", command=["train"], resources=r)
        assert spec.resources.gpus == 8
        assert spec.resources.nodes == 4

    def test_independent_defaults(self) -> None:
        spec1 = JobSpec(job_id="a", command=["x"])
        spec2 = JobSpec(job_id="b", command=["y"])
        assert spec1.resources is not spec2.resources


class TestBackendHandle:
    def test_fields(self) -> None:
        h = BackendHandle(backend_name="ssh", job_id="j1", remote_pid=1234)
        assert h.backend_name == "ssh"
        assert h.container_id is None


class TestBackendStatus:
    def test_running(self) -> None:
        s = BackendStatus(running=True)
        assert s.running is True
        assert s.exit_code is None


class TestSSHBackendName:
    def test_default_name(self) -> None:
        backend = SSHBackend(host="example.com")
        assert backend.name == "ssh"

    def test_custom_name(self) -> None:
        backend = SSHBackend(host="example.com", name="gpu-node")
        assert backend.name == "gpu-node"

    def test_backend_handle_preserves_custom_name(self) -> None:
        backend = SSHBackend(host="example.com", name="my-gpu")
        handle = BackendHandle(backend_name=backend.name, job_id="j1")
        assert handle.backend_name == "my-gpu"

    def test_multiple_backends_different_names(self) -> None:
        b1 = SSHBackend(host="10.0.0.1", name="gpu-a")
        b2 = SSHBackend(host="10.0.0.2", name="gpu-b")
        assert b1.name == "gpu-a"
        assert b2.name == "gpu-b"
        assert b1.name != b2.name

    def test_register_custom_name(self) -> None:
        reset()
        backend = SSHBackend(host="10.0.0.1", name="my-cluster")
        register_backend(backend)
        names = [b["name"] for b in list_backends()]
        assert "my-cluster" in names


class TestLoadBackends:
    def setup_method(self) -> None:
        reset()

    def test_no_config_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with patch("amortized.main.Path.home", return_value=tmp_path):
            _load_backends()
        names = [b["name"] for b in list_backends()]
        assert names == ["local"]

    def test_ssh_backend_from_config(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        config_dir = tmp_path / ".amortized"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text(
            "compute:\n"
            "  backends:\n"
            "    gpu-box:\n"
            "      type: ssh\n"
            "      host: 10.0.0.5\n"
            "      user: trainer\n"
        )
        with patch("amortized.main.Path.home", return_value=tmp_path):
            _load_backends()
        names = [b["name"] for b in list_backends()]
        assert "local" in names
        assert "gpu-box" in names

    def test_invalid_yaml_logs_warning(self, tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
        config_dir = tmp_path / ".amortized"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_bytes(b"\x80\x81\x82")
        with patch("amortized.main.Path.home", return_value=tmp_path):
            _load_backends()
        names = [b["name"] for b in list_backends()]
        assert names == ["local"]

    def test_multiple_ssh_backends_from_config(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        config_dir = tmp_path / ".amortized"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text(
            "compute:\n"
            "  backends:\n"
            "    gpu-node-1:\n"
            "      type: ssh\n"
            "      host: 10.0.0.5\n"
            "    gpu-node-2:\n"
            "      type: ssh\n"
            "      host: 10.0.0.6\n"
        )
        with patch("amortized.main.Path.home", return_value=tmp_path):
            _load_backends()
        names = [b["name"] for b in list_backends()]
        assert "local" in names
        assert "gpu-node-1" in names
        assert "gpu-node-2" in names

    def test_unknown_backend_type_skipped(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        config_dir = tmp_path / ".amortized"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text(
            "compute:\n  backends:\n    mystery:\n      type: kubernetes\n      host: k8s.local\n"
        )
        with patch("amortized.main.Path.home", return_value=tmp_path):
            _load_backends()
        names = [b["name"] for b in list_backends()]
        assert names == ["local"]


def _shm_volume(pod: Any) -> Any:
    return next(v for v in pod.volumes if v.name == "shm")


class TestKubernetesCpuPodSpec:
    """CPU pods (gpus=0): explicit CPU/memory requests+limits, no GPU plumbing."""

    def _backend(self) -> KubernetesBackend:
        return KubernetesBackend()

    def _cpu_spec(self) -> JobSpec:
        return JobSpec(
            job_id="cpu-job",
            command=["train"],
            resources=Resources(gpus=0, cpus=4, memory_gb=8),
            timeout=3600,
        )

    def test_no_gpu_keys_node_selector_or_runtime_class(self) -> None:
        pod = self._backend()._build_pod_spec(self._cpu_spec(), "amortized-cpu-job")
        assert pod.node_selector is None
        assert pod.runtime_class_name is None
        res = pod.containers[0].resources
        assert "nvidia.com/gpu" not in (res.requests or {})
        assert "nvidia.com/gpu" not in (res.limits or {})

    def test_cpu_memory_requests_and_limits(self) -> None:
        pod = self._backend()._build_pod_spec(self._cpu_spec(), "amortized-cpu-job")
        res = pod.containers[0].resources
        assert res.requests == {"cpu": "4", "memory": "8Gi"}
        assert res.limits == {"cpu": "4", "memory": "8Gi"}

    def test_shm_shrunk_to_fit_memory_limit(self) -> None:
        pod = self._backend()._build_pod_spec(self._cpu_spec(), "amortized-cpu-job")
        # 12Gi shm would exceed the 8Gi memory limit — capped at half.
        assert _shm_volume(pod).empty_dir.size_limit == "4Gi"

    def test_shm_kept_when_memory_limit_large(self) -> None:
        spec = JobSpec(
            job_id="cpu-big",
            command=["train"],
            resources=Resources(gpus=0, cpus=4, memory_gb=32),
        )
        pod = self._backend()._build_pod_spec(spec, "amortized-cpu-big")
        assert _shm_volume(pod).empty_dir.size_limit == "12Gi"

    def test_openshift_scc_compatible(self) -> None:
        pod = self._backend()._build_pod_spec(self._cpu_spec(), "amortized-cpu-job")
        # No hostPath volumes, no privilege escalation, no runtime class.
        assert all(v.host_path is None for v in pod.volumes)
        container = pod.containers[0]
        assert container.security_context is not None
        assert container.security_context.allow_privilege_escalation is False


class TestKubernetesGpuPodSpecUnchanged:
    """GPU pods keep the exact pre-CPU-support pod spec shape."""

    def _gpu_pod(self) -> Any:
        spec = JobSpec(
            job_id="gpu-job",
            command=["train"],
            resources=Resources(gpus=1),
        )
        return KubernetesBackend()._build_pod_spec(spec, "amortized-gpu-job")

    def test_gpu_keys_node_selector_and_runtime_class(self) -> None:
        pod = self._gpu_pod()
        assert pod.node_selector == {"nvidia.com/gpu.present": "true"}
        assert pod.runtime_class_name == "nvidia"
        res = pod.containers[0].resources
        assert res.requests == {"nvidia.com/gpu": "1"}
        assert res.limits == {"nvidia.com/gpu": "1"}

    def test_gpu_shm_unchanged(self) -> None:
        assert _shm_volume(self._gpu_pod()).empty_dir.size_limit == "12Gi"

    def test_gpu_pod_without_cpu_resources_has_no_cpu_entries(self) -> None:
        res = self._gpu_pod().containers[0].resources
        assert "cpu" not in (res.requests or {})
        assert "cpu" not in (res.limits or {})


class TestKubernetesJobDeadline:
    def _backend(self) -> KubernetesBackend:
        return KubernetesBackend()

    def _job(self, spec: JobSpec) -> Any:
        backend = self._backend()
        pod = backend._build_pod_spec(spec, "amortized-x")
        return backend._build_job(spec, "amortized-x", pod)

    def test_timeout_sets_active_deadline_seconds(self) -> None:
        spec = JobSpec(
            job_id="cpu-job",
            command=["train"],
            resources=Resources(gpus=0, cpus=4, memory_gb=8),
            timeout=3600,
        )
        job = self._job(spec)
        assert job.spec.active_deadline_seconds == 3600

    def test_cpu_job_backoff_limit_zero(self) -> None:
        spec = JobSpec(
            job_id="cpu-job",
            command=["train"],
            resources=Resources(gpus=0, cpus=4, memory_gb=8),
            timeout=3600,
        )
        assert self._job(spec).spec.backoff_limit == 0

    def test_no_timeout_leaves_deadline_unset(self) -> None:
        spec = JobSpec(job_id="gpu-job", command=["train"], resources=Resources(gpus=1))
        assert self._job(spec).spec.active_deadline_seconds is None


class TestKubernetesStatusTranslation:
    def _backend_with_mocked_job(self, job: Any) -> tuple[KubernetesBackend, MagicMock]:
        backend = KubernetesBackend()
        backend._client = object()  # skip in-cluster config
        batch_cls = MagicMock()
        batch_cls.return_value.read_namespaced_job = AsyncMock(return_value=job)
        return backend, batch_cls

    @staticmethod
    def _failed_job(reason: str) -> Any:
        job = MagicMock()
        job.status.succeeded = 0
        job.status.failed = 1
        cond = MagicMock()
        cond.type = "Failed"
        cond.reason = reason
        job.status.conditions = [cond]
        return job

    @pytest.mark.asyncio
    async def test_deadline_exceeded_maps_to_timed_out(self) -> None:
        backend, batch_cls = self._backend_with_mocked_job(self._failed_job("DeadlineExceeded"))
        with patch("kubernetes_asyncio.client.BatchV1Api", batch_cls):
            status = await backend.status(
                BackendHandle(backend_name="kubernetes", job_id="j1", scheduler_id="amortized-j1")
            )
        assert status.running is False
        assert status.error is not None
        assert "timed out" in status.error

    @pytest.mark.asyncio
    async def test_plain_failure_not_reported_as_timeout(self) -> None:
        backend, batch_cls = self._backend_with_mocked_job(self._failed_job("BackoffLimitExceeded"))
        with patch("kubernetes_asyncio.client.BatchV1Api", batch_cls):
            status = await backend.status(
                BackendHandle(backend_name="kubernetes", job_id="j1", scheduler_id="amortized-j1")
            )
        assert status.running is False
        assert status.error is not None
        assert "timed out" not in status.error


class TestSSHGpuFlag:
    @staticmethod
    async def _docker_run_command(gpus: int) -> str:
        backend = SSHBackend(host="example.com")
        conn = MagicMock()
        conn.run = AsyncMock(return_value=MagicMock(stdout=""))
        with patch.object(SSHBackend, "_connect", new=AsyncMock(return_value=conn)):
            await backend.submit(
                JobSpec(
                    job_id="job-gpu-flag",
                    command=["train"],
                    image="example.com/img:latest",
                    resources=Resources(gpus=gpus),
                )
            )
        docker_cmds = [
            str(c.args[0]) for c in conn.run.call_args_list if "run -d" in str(c.args[0])
        ]
        assert len(docker_cmds) == 1
        return docker_cmds[0]

    @pytest.mark.asyncio
    async def test_cpu_job_omits_gpus_flag(self) -> None:
        cmd = await self._docker_run_command(gpus=0)
        assert "--gpus all" not in cmd

    @pytest.mark.asyncio
    async def test_gpu_job_includes_gpus_flag(self) -> None:
        cmd = await self._docker_run_command(gpus=1)
        assert "--gpus all" in cmd
