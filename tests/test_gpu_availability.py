"""Tests for the GPU inventory helpers (DaemonSet-backed sharing)."""

from types import SimpleNamespace

from amortized.core import gpu_inventory as gi


def _pod(
    name: str,
    namespace: str,
    uuids: str,
    phase: str = "Running",
    annotations: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    start_time=None,
):
    """A minimal fake k8s pod with an NVIDIA_VISIBLE_DEVICES env var.

    Serve jobs get the pin via a Secret-backed env var (value empty in the
    pod spec) plus a mirroring annotation; the annotation is authoritative.
    """
    env = [SimpleNamespace(name="NVIDIA_VISIBLE_DEVICES", value=uuids)]
    container = SimpleNamespace(env=env)
    spec = SimpleNamespace(containers=[container])
    metadata = SimpleNamespace(
        name=name,
        namespace=namespace,
        annotations=annotations or {},
        labels=labels or {},
    )
    status = SimpleNamespace(phase=phase, start_time=start_time)
    return SimpleNamespace(spec=spec, metadata=metadata, status=status)


class TestPodGpuUuids:
    def test_extracts_uuids(self) -> None:
        pod = _pod("p1", "ns", "GPU-abc,GPU-def")
        assert gi._pod_gpu_uuids(pod) == {"GPU-abc", "GPU-def"}

    def test_ignores_non_uuid_values(self) -> None:
        assert gi._pod_gpu_uuids(_pod("p1", "ns", "all")) == set()
        assert gi._pod_gpu_uuids(_pod("p1", "ns", "0,1,2")) == set()
        assert gi._pod_gpu_uuids(_pod("p1", "ns", "none")) == set()

    def test_mixed_values_keep_uuids_only(self) -> None:
        pod = _pod("p1", "ns", "0,GPU-xyz")
        assert gi._pod_gpu_uuids(pod) == {"GPU-xyz"}

    def test_annotation_takes_precedence_over_env(self) -> None:
        # Secret-backed env var: value is empty in the pod spec.
        pod = _pod(
            "p1",
            "ns",
            "",
            annotations={"amortized.io/gpu-uuids": "GPU-de052fad,GPU-83733b9f"},
        )
        assert gi._pod_gpu_uuids(pod) == {"GPU-de052fad", "GPU-83733b9f"}

    def test_annotation_ignores_non_uuid_values(self) -> None:
        pod = _pod("p1", "ns", "all", annotations={"amortized.io/gpu-uuids": "all"})
        assert gi._pod_gpu_uuids(pod) == set()

    def test_no_env(self) -> None:
        pod = SimpleNamespace(
            spec=SimpleNamespace(containers=[SimpleNamespace(env=[])]),
            metadata=SimpleNamespace(name="p", namespace="ns"),
        )
        assert gi._pod_gpu_uuids(pod) == set()


class TestIsBusy:
    def test_busy_by_memory(self) -> None:
        assert gi._is_busy({"memory_total_mb": 81559, "memory_free_mb": 5407})
        assert gi._is_busy({"memory_total_mb": 81559, "memory_free_mb": 3243})

    def test_free_gpu_not_busy(self) -> None:
        assert not gi._is_busy({"memory_total_mb": 81559, "memory_free_mb": 81081})
        # just under the threshold's slack is still considered free
        assert not gi._is_busy({"memory_total_mb": 81559, "memory_free_mb": 74000})


class TestUserOfNamespace:
    def test_user_job_namespace(self) -> None:
        assert gi._user_of_namespace("amortized-xyang-jobs") == "xyang"

    def test_other_namespace(self) -> None:
        assert gi._user_of_namespace("amortized") == "amortized"
        assert gi._user_of_namespace("kube-system") == "kube-system"


class TestAssignServeGpu:
    def test_reuses_my_pinned_gpu_even_when_busy(self, monkeypatch) -> None:
        # my serve pod pins GPU-B (only 5GB free — sharing is the point)
        pods = [
            _pod("mine", "amortized-me-jobs", "GPU-B"),
            _pod("theirs", "amortized-other-jobs", "GPU-A"),
        ]
        _patch(monkeypatch, [
            {"uuid": "GPU-A", "memory_free_mb": 5000},
            {"uuid": "GPU-B", "memory_free_mb": 5000},
            {"uuid": "GPU-C", "memory_free_mb": 80000},
        ], pods)
        chosen = _run(gi.assign_serve_gpu("amortized-me-jobs"))
        assert chosen == ["GPU-B"]

    def test_new_user_gets_free_gpu_not_busy_one(self, monkeypatch) -> None:
        # GPU-A busy from a training job (memory in use, nobody pins it);
        # GPU-B pinned by another user but idle; GPU-C fully free
        pods = [_pod("theirs", "amortized-other-jobs", "GPU-B")]
        _patch(monkeypatch, [
            {"uuid": "GPU-A", "memory_free_mb": 5000},
            {"uuid": "GPU-B", "memory_free_mb": 80000},
            {"uuid": "GPU-C", "memory_free_mb": 79000},
        ], pods)
        chosen = _run(gi.assign_serve_gpu("amortized-me-jobs"))
        assert chosen == ["GPU-C"]

    def test_multiple_gpus_sorted_by_free(self, monkeypatch) -> None:
        _patch(monkeypatch, [
            {"uuid": "GPU-A", "memory_free_mb": 80000},
            {"uuid": "GPU-B", "memory_free_mb": 75000},
        ], [])
        chosen = _run(gi.assign_serve_gpu("amortized-me-jobs", needed_gpus=2))
        assert chosen == ["GPU-A", "GPU-B"]

    def test_nothing_available_raises(self, monkeypatch) -> None:
        # everything busy by memory
        _patch(monkeypatch, [{"uuid": "GPU-A", "memory_free_mb": 5000}], [])
        from amortized.jobs.base import JobBuildError

        import pytest

        with pytest.raises(JobBuildError):
            _run(gi.assign_serve_gpu("amortized-me-jobs"))


class TestDescribeGpus:
    def test_mine_busy_and_free(self, monkeypatch) -> None:
        pods = [
            _pod("mine", "amortized-me-jobs", "GPU-B"),
            _pod("theirs", "amortized-xingliu-jobs", "GPU-A"),
        ]
        _patch(monkeypatch, [
            {"uuid": "GPU-A", "memory_free_mb": 1000, "memory_total_mb": 81559,
             "node": "n1", "index": 0, "updated": "t"},
            {"uuid": "GPU-B", "memory_free_mb": 79000, "memory_total_mb": 81559,
             "node": "n1", "index": 1, "updated": "t"},
            {"uuid": "GPU-C", "memory_free_mb": 5000, "memory_total_mb": 81559,
             "node": "n1", "index": 2, "updated": "t"},
        ], pods)
        out = _run(gi.describe_gpus("amortized-me-jobs"))
        by_uuid = {g["uuid"]: g for g in out["gpus"]}
        # mine: pinned by my namespace
        assert by_uuid["GPU-B"]["mine"] is True
        assert by_uuid["GPU-B"]["held_by"] == ["me"]
        # pinned by someone else
        assert by_uuid["GPU-A"]["mine"] is False
        assert by_uuid["GPU-A"]["held_by"] == ["xingliu"]
        # busy with no pinner (training / host) -> "in use"
        assert by_uuid["GPU-C"]["held_by"] == ["in use"]
        assert by_uuid["GPU-C"]["busy"] is True
        assert out["my_uuids"] == ["GPU-B"]


# --- helpers ---------------------------------------------------------------

def _run(coro):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


class _FakeApiClient:
    async def close(self):
        return None


def _patch(monkeypatch, gpus, pods):
    async def fake_read_inventory():
        return [
            {
                "node": g.get("node", "n1"),
                "index": g.get("index", 0),
                "uuid": g["uuid"],
                "memory_free_mb": g["memory_free_mb"],
                "memory_total_mb": g.get("memory_total_mb", 81559),
                "updated": g.get("updated", "t"),
            }
            for g in gpus
        ]

    async def fake_pods():
        core = SimpleNamespace(
            list_pod_for_all_namespaces=_async_result(SimpleNamespace(items=pods))
        )
        return _FakeApiClient(), core

    monkeypatch.setattr(gi, "read_inventory", fake_read_inventory)
    monkeypatch.setattr(gi, "_core_v1", fake_pods)


def _async_result(value):
    async def result():
        return value

    return result


class TestPinnedOccupancyAttribution:
    def test_holders_carry_job_labels(self, monkeypatch) -> None:
        from datetime import datetime, UTC

        pod = _pod(
            "pod-1",
            "amortized-me-jobs",
            "GPU-abc",
            labels={
                "amortized/job-id": "cc7cbd77-1111",
                "amortized/job-type": "serve",
            },
            start_time=datetime(2026, 9, 14, 18, 18, tzinfo=UTC),
        )

        async def fake_pods():
            return [pod]

        monkeypatch.setattr(gi, "_pinned_gpu_pods", fake_pods)
        import asyncio

        occ = asyncio.run(gi._pinned_occupancy())
        assert occ["GPU-abc"][0]["job_id"] == "cc7cbd77-1111"
        assert occ["GPU-abc"][0]["job_type"] == "serve"
        assert occ["GPU-abc"][0]["started_at"].startswith("2026-09-14T18:18")

    def test_holders_without_labels_default_empty(self, monkeypatch) -> None:
        pod = _pod("pod-2", "amortized-me-jobs", "GPU-abc")

        async def fake_pods():
            return [pod]

        monkeypatch.setattr(gi, "_pinned_gpu_pods", fake_pods)
        import asyncio

        occ = asyncio.run(gi._pinned_occupancy())
        assert occ["GPU-abc"][0]["job_id"] == ""
        assert occ["GPU-abc"][0]["job_type"] == ""
