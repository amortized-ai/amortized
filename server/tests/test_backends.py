"""Tests for compute backend data types."""

from unittest.mock import patch

from amortized.backends import BackendHandle, BackendStatus, JobSpec, Resources
from amortized.core.compute import list_backends, reset
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
        assert "ssh" in names

    def test_invalid_yaml_logs_warning(self, tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
        config_dir = tmp_path / ".amortized"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_bytes(b"\x80\x81\x82")
        with patch("amortized.main.Path.home", return_value=tmp_path):
            _load_backends()
        names = [b["name"] for b in list_backends()]
        assert names == ["local"]

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
