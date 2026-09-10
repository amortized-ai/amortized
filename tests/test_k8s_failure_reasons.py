"""Tests for KubernetesBackend pod-failure diagnostics."""

from types import SimpleNamespace

from amortized.backends.kubernetes import KubernetesBackend


def _backend() -> KubernetesBackend:
    return KubernetesBackend(namespace="test-jobs")


def _pod(phase: str, waiting: dict | None = None, terminated: dict | None = None) -> SimpleNamespace:
    state = SimpleNamespace(
        waiting=SimpleNamespace(**waiting) if waiting else None,
        terminated=SimpleNamespace(**terminated) if terminated else None,
    )
    cs = SimpleNamespace(
        state=state,
        image="ghcr.io/amortized-ai/eval:latest",
    )
    return SimpleNamespace(
        status=SimpleNamespace(phase=phase, container_statuses=[cs])
    )


def _pods_response(pods: list) -> SimpleNamespace:
    return SimpleNamespace(items=pods)


def _job(conditions: list[dict] | None = None) -> SimpleNamespace:
    conds = [SimpleNamespace(**c) for c in (conditions or [])]
    return SimpleNamespace(
        status=SimpleNamespace(succeeded=0, failed=0, conditions=conds)
    )


class _FakeCore:
    def __init__(self, pods: list) -> None:
        self._pods = pods

    async def list_namespaced_pod(self, namespace: str, label_selector: str) -> SimpleNamespace:
        return _pods_response(self._pods)


def _patch_core(monkeypatch, pods: list) -> None:
    import kubernetes_asyncio.client as k8s_client

    fake = _FakeCore(pods)
    monkeypatch.setattr(k8s_client, "CoreV1Api", lambda api_client: fake)


def _run(coro):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


class TestStuckPodReason:
    def test_image_pull_backoff_on_pending_pod(self, monkeypatch) -> None:
        pod = _pod(
            "Pending",
            waiting={
                "reason": "ImagePullBackOff",
                "message": 'Back-off pulling image "ghcr.io/amortized-ai/eval:latest": ... 403 Forbidden',
            },
        )
        _patch_core(monkeypatch, [pod])
        reason = _run(_backend()._get_stuck_pod_reason("job1", api_client=None))
        assert reason is not None
        assert "ghcr.io/amortized-ai/eval:latest" in reason
        assert "ImagePullBackOff" in reason
        assert "403 Forbidden" in reason

    def test_create_container_config_error(self, monkeypatch) -> None:
        pod = _pod(
            "Pending",
            waiting={
                "reason": "CreateContainerConfigError",
                "message": 'secret "job-env" not found',
            },
        )
        _patch_core(monkeypatch, [pod])
        reason = _run(_backend()._get_stuck_pod_reason("job1", api_client=None))
        assert reason is not None
        assert "CreateContainerConfigError" in reason

    def test_running_pod_not_stuck(self, monkeypatch) -> None:
        pod = _pod("Running")
        _patch_core(monkeypatch, [pod])
        assert _run(_backend()._get_stuck_pod_reason("job1", api_client=None)) is None

    def test_pending_pod_with_other_reason_not_stuck(self, monkeypatch) -> None:
        pod = _pod("Pending", waiting={"reason": "PodInitializing", "message": ""})
        _patch_core(monkeypatch, [pod])
        assert _run(_backend()._get_stuck_pod_reason("job1", api_client=None)) is None

    def test_no_pods(self, monkeypatch) -> None:
        _patch_core(monkeypatch, [])
        assert _run(_backend()._get_stuck_pod_reason("job1", api_client=None)) is None

    def test_long_message_truncated(self, monkeypatch) -> None:
        pod = _pod(
            "Pending",
            waiting={"reason": "ImagePullBackOff", "message": "x" * 1000},
        )
        _patch_core(monkeypatch, [pod])
        reason = _run(_backend()._get_stuck_pod_reason("job1", api_client=None))
        assert reason is not None
        assert "x" * 300 in reason
        assert "x" * 301 not in reason


class TestPodFailureReason:
    def test_waiting_image_pull(self, monkeypatch) -> None:
        pod = _pod(
            "Pending",
            waiting={"reason": "ImagePullBackOff", "message": "403"},
        )
        _patch_core(monkeypatch, [pod])
        reason = _run(_backend()._get_pod_failure_reason("job1", api_client=None))
        assert reason is not None
        assert "ImagePullBackOff" in reason

    def test_terminated_nonzero(self, monkeypatch) -> None:
        pod = _pod("Failed", terminated={"exit_code": 1, "reason": "Error"})
        _patch_core(monkeypatch, [pod])
        reason = _run(_backend()._get_pod_failure_reason("job1", api_client=None))
        assert reason is not None
        assert "exited with code 1" in reason

    def test_no_pods_returns_none(self, monkeypatch) -> None:
        _patch_core(monkeypatch, [])
        assert _run(_backend()._get_pod_failure_reason("job1", api_client=None)) is None


class TestJobConditionMessage:
    def test_backoff_limit_exceeded(self) -> None:
        job = _job(
            conditions=[
                {
                    "type": "FailureTarget",
                    "reason": "BackoffLimitExceeded",
                    "message": "Job has reached the specified backoff limit",
                }
            ]
        )
        msg = KubernetesBackend._job_condition_message(job)
        assert msg == "Job has reached the specified backoff limit"

    def test_no_failure_condition(self) -> None:
        job = _job(conditions=[{"type": "Complete", "reason": "Completed", "message": ""}])
        assert KubernetesBackend._job_condition_message(job) is None
