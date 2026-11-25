"""Contracts at approval, durable dispatch, engine ownership and task scoring."""

import copy
import json
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from conftest import TEST_DATABASE_URL
from pydantic import ValidationError

from amortized.db.repository import Repository
from amortized.evaluation.spec import TaskEvaluation
from amortized.integrations.evalhub import SERVER_LIMIT, client, lifecycle
from amortized.integrations.evalhub.results import present
from amortized.models import EvalJobRequest

REAL_VALIDATE = client.validate_definition


@pytest.fixture(autouse=True)
def _hub_connected(monkeypatch):
    """Hub jobs are followed only while the server has a Hub URL (MemoryStore's hub)."""
    import amortized.config as config_mod

    monkeypatch.setattr(config_mod.settings, "evalhub_url", "http://hub")


def snapshot(state="pending", remote_id="hub-123", metrics=None):
    result = {
        "resource": {"id": remote_id, "created_at": "2026-11-03T12:00:00Z"},
        "name": "amortized-product-123",
        "tags": ["amortized-product-123"],
        "model": {"name": "test", "url": "http://model/v1"},
        "status": {"state": state, "benchmarks": []},
    }
    if metrics is not None:
        result["results"] = {
            "benchmarks": [
                {
                    "id": "task",
                    "provider_id": "amortized",
                    "metrics": metrics,
                    "mlflow_run_id": "engine-run",
                    "artifacts": {},
                }
            ]
        }
    return result


REQUEST = {
    "name": "amortized-product-123",
    "model": {"url": "http://model/v1", "name": "test"},
    "benchmarks": [{"id": "task", "provider_id": "amortized"}],
}


class MemoryStore:
    def __init__(self, state="ready", cancel=False):
        self.execution = {
            "job_id": "product-123",
            "tenant": "user-namespace",
            "hub_url": "http://hub",
            "correlation": "amortized-product-123",
            "state": state,
            "request": copy.deepcopy(REQUEST),
            "cancel_requested": cancel,
            "remote_id": "hub-123" if state == "accepted" else None,
            "snapshot": {},
            "sync_error": "",
            "observed_at": None,
            "inputs": {},
        }
        self.job = {
            "id": "product-123",
            "type": "eval",
            "status": "provisioning",
            "execution_owner": "evalhub",
            "config": {},
        }
        self.conn = MagicMock()
        self.conn.transaction = self.transaction
        self.repo = MagicMock()
        self.repo.update_job = AsyncMock(side_effect=self.update_job)
        self.repo.get_job = AsyncMock(side_effect=lambda _: self.job.copy())

    @asynccontextmanager
    async def transaction(self):
        saved = copy.deepcopy((self.execution, self.job))
        try:
            yield
        except BaseException:
            for live, kept in zip((self.execution, self.job), saved, strict=True):
                live.clear()
                live.update(kept)
            raise

    async def get(self, _):
        return copy.deepcopy(self.execution)

    async def update(self, _, **fields):
        self.execution.update(fields)

    async def cancel(self, _):
        self.execution["cancel_requested"] = True

    async def cancel_unclaimed(self, _):
        if self.job["status"] != "queued":
            return False
        self.job.update(status="cancelled", completed_at=datetime.now(UTC))
        self.execution.update(state="finished", cancel_requested=True)
        return True

    async def update_job(self, _, **fields):
        self.job.update(fields)
        return self.job.copy()


@pytest.mark.asyncio
async def test_lost_submit_response_never_posts_again(monkeypatch):
    store = MemoryStore()
    submit = AsyncMock(side_effect=httpx.ReadTimeout("response lost"))
    monkeypatch.setattr(client, "submit", submit)
    monkeypatch.setattr(client, "reconcile", AsyncMock(return_value=[]))
    await lifecycle.tick(store.job, store)
    assert store.execution["state"] == "uncertain"
    for _ in range(3):
        await lifecycle.tick(store.job, store)
    assert submit.await_count == 1
    assert store.job["status"] == "provisioning"
    monkeypatch.setattr(client, "reconcile", AsyncMock(return_value=[snapshot("running")]))
    await lifecycle.tick(store.job, store)
    assert store.execution["remote_id"] == "hub-123"
    assert store.job["status"] == "running"


@pytest.mark.asyncio
async def test_restart_during_post_reconciles_without_submission(monkeypatch):
    store = MemoryStore("submitting")
    submit = AsyncMock()
    monkeypatch.setattr(client, "submit", submit)
    monkeypatch.setattr(client, "reconcile", AsyncMock(return_value=[snapshot()]))
    await lifecycle.tick(store.job, store)
    submit.assert_not_awaited()
    assert store.execution["state"] == "accepted"


@pytest.mark.asyncio
async def test_ambiguous_matches_are_never_adopted(monkeypatch):
    store = MemoryStore("uncertain")
    monkeypatch.setattr(
        client, "reconcile", AsyncMock(return_value=[snapshot(), snapshot(remote_id="other")])
    )
    await lifecycle.tick(store.job, store)
    assert store.execution["remote_id"] is None
    assert "Multiple" in store.execution["sync_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["preparing", "ready"])
async def test_cancel_before_submission_suppresses_execution(monkeypatch, state):
    store = MemoryStore(state, cancel=True)
    submit = AsyncMock()
    monkeypatch.setattr(client, "submit", submit)
    monkeypatch.setattr(lifecycle.preparation, "prepare", AsyncMock())
    cleanup = AsyncMock()
    monkeypatch.setattr(lifecycle.preparation, "cleanup", cleanup)
    await lifecycle.tick(store.job, store)
    await lifecycle.tick(store.job, store)
    assert store.job["status"] == "cancelled"
    assert store.execution["state"] == "finished"
    submit.assert_not_awaited()
    lifecycle.preparation.prepare.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_after_uncertain_submit_recovers_exact_remote_job(monkeypatch):
    store = MemoryStore("uncertain", cancel=True)
    monkeypatch.setattr(client, "reconcile", AsyncMock(return_value=[snapshot("running")]))
    cancel = AsyncMock()
    monkeypatch.setattr(client, "cancel", cancel)
    monkeypatch.setattr(client, "observe", AsyncMock(return_value=snapshot("running")))
    await lifecycle.tick(store.job, store)
    await lifecycle.tick(store.job, store)
    assert cancel.call_args.args[0]["remote_id"] == "hub-123"
    assert store.job["status"] == "running"
    monkeypatch.setattr(client, "observe", AsyncMock(return_value=snapshot("cancelled")))
    await lifecycle.tick(store.job, store)
    assert store.job["status"] == "cancelled"


@pytest.mark.asyncio
async def test_unsent_submit_can_be_cancelled(monkeypatch):
    store = MemoryStore("ready")
    monkeypatch.setattr(client, "submit", AsyncMock(side_effect=httpx.ConnectError("refused")))
    monkeypatch.setattr(client, "reconcile", AsyncMock(return_value=[]))
    cleanup = AsyncMock()
    monkeypatch.setattr(lifecycle.preparation, "cleanup", cleanup)
    await lifecycle.tick(store.job, store)
    store.execution["cancel_requested"] = True
    for _ in range(50):
        await lifecycle.tick(store.job, store)
    assert store.job["status"] == "cancelled"
    cleanup.assert_awaited()


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ConnectTimeout("no route")])
async def test_a_submission_that_never_connected_is_sent_again(monkeypatch, error):
    store = MemoryStore()
    submit = AsyncMock(side_effect=[error, snapshot("running")])
    reconcile = AsyncMock()
    monkeypatch.setattr(client, "submit", submit)
    monkeypatch.setattr(client, "reconcile", reconcile)
    await lifecycle.tick(store.job, store)
    assert store.execution["state"] == "ready"
    assert store.execution["observed_at"] is None
    assert "unreachable" in store.execution["sync_error"]
    await lifecycle.tick(store.job, store)
    assert submit.await_count == 2
    assert store.execution["state"] == "accepted"
    assert store.job["status"] == "running"
    reconcile.assert_not_awaited()


@pytest.mark.parametrize("problem", ["invalid request", "insecure origin"])
async def test_a_submission_that_cannot_be_sent_fails_without_posting(monkeypatch, problem):
    import amortized.config as config_mod

    store = MemoryStore()
    if problem == "invalid request":
        store.execution["request"] = {"name": "no model or benchmarks"}
    else:
        monkeypatch.setattr(config_mod.settings, "evalhub_token_file", "/var/run/hub/token")
    submit = AsyncMock()
    monkeypatch.setattr(client, "submit", submit)
    await lifecycle.tick(store.job, store)
    assert store.execution["state"] == "cleanup"
    assert store.job["status"] == "failed"
    assert store.job["error"].startswith("Cannot submit to Eval Hub")
    submit.assert_not_awaited()


async def test_an_unreadable_submit_response_is_reconciled_rather_than_failed(monkeypatch):
    store = MemoryStore()
    monkeypatch.setattr(
        client, "submit", AsyncMock(side_effect=json.JSONDecodeError("not JSON", "", 0))
    )
    await lifecycle.tick(store.job, store)
    assert store.execution["state"] == "uncertain"
    assert store.job["status"] == "provisioning"


async def test_an_unconfirmed_submission_fails_at_its_deadline_and_cancels_a_late_hub_job(
    monkeypatch,
):
    import amortized.config as config_mod

    monkeypatch.setattr(config_mod.settings, "evalhub_uncertain_timeout", 60.0)
    store = MemoryStore("uncertain")
    store.execution["observed_at"] = datetime.now(UTC) - timedelta(seconds=30)
    reconcile = AsyncMock(return_value=[])
    cancel, cleanup = AsyncMock(), AsyncMock()
    monkeypatch.setattr(client, "reconcile", reconcile)
    monkeypatch.setattr(client, "cancel", cancel)
    monkeypatch.setattr(lifecycle.preparation, "cleanup", cleanup)
    await lifecycle.tick(store.job, store)
    assert store.execution["state"] == "uncertain"
    store.execution["observed_at"] = datetime.now(UTC) - timedelta(seconds=61)
    reconcile.side_effect = httpx.ConnectError("offline")
    await lifecycle.tick(store.job, store)
    assert store.job["status"] == "failed"
    assert store.job["error"] == "Eval Hub did not confirm the submission within 60 seconds"
    assert store.execution["state"] == "cleanup"
    reconcile.side_effect = None
    reconcile.return_value = [snapshot("running", remote_id="late")]
    assert await lifecycle.tick(store.job, store)
    assert cancel.await_args.args[0]["remote_id"] == "late"
    cleanup.assert_awaited_once()
    assert store.execution["state"] == "finished"


@pytest.mark.parametrize("hub", ["no match", "unreachable", "ambiguous"])
async def test_cancel_while_the_submission_is_unconfirmed_ends_the_job(monkeypatch, hub):
    store = MemoryStore("uncertain", cancel=True)
    store.execution["observed_at"] = datetime.now(UTC)
    found = {"no match": [], "ambiguous": [snapshot(), snapshot(remote_id="other")]}
    reconcile = (
        AsyncMock(side_effect=httpx.ConnectError("offline"))
        if hub == "unreachable"
        else AsyncMock(return_value=found[hub])
    )
    cancel, cleanup = AsyncMock(), AsyncMock()
    monkeypatch.setattr(client, "reconcile", reconcile)
    monkeypatch.setattr(client, "cancel", cancel)
    monkeypatch.setattr(lifecycle.preparation, "cleanup", cleanup)
    await lifecycle.tick(store.job, store)
    assert store.job["status"] == "cancelled"
    assert store.execution["state"] == "cleanup"
    assert store.execution["remote_id"] is None
    assert await lifecycle.tick(store.job, store)
    cleanup.assert_awaited_once()
    cancelled = [call.args[0]["remote_id"] for call in cancel.await_args_list]
    assert cancelled == (["hub-123", "other"] if hub == "ambiguous" else [])


TERMINAL_BRANCHES = {
    # branch: (state, cancel, execution fields, (state, status) a failed write leaves)
    "preparing cancelled": ("preparing", True, {}, ("preparing", "provisioning")),
    "preparing failed": ("preparing", False, {}, ("preparing", "provisioning")),
    "ready cancelled": ("ready", True, {}, ("ready", "provisioning")),
    "ready unsendable": ("ready", False, {"request": {}}, ("ready", "provisioning")),
    "ready rejected": ("ready", False, {}, ("submitting", "provisioning")),
    "uncertain cancelled": ("uncertain", True, {}, ("uncertain", "provisioning")),
    "uncertain expired": (
        "uncertain",
        False,
        {"observed_at": "2026-11-03T12:00:00Z"},
        ("uncertain", "provisioning"),
    ),
    "adapter stopped": ("accepted", False, {}, ("accepted", "running")),
}


@pytest.mark.parametrize("failing", ["job", "execution"])
@pytest.mark.parametrize("branch", TERMINAL_BRANCHES)
async def test_a_verdict_commits_together_with_its_cleanup_or_not_at_all(
    monkeypatch, branch, failing
):
    from amortized.jobs.base import JobBuildError

    state, cancel, fields, left = TERMINAL_BRANCHES[branch]
    rejected = httpx.Response(422, request=httpx.Request("POST", "http://hub"))
    monkeypatch.setattr(
        client,
        "submit",
        AsyncMock(
            side_effect=httpx.HTTPStatusError("x", request=rejected.request, response=rejected)
        ),
    )
    monkeypatch.setattr(client, "reconcile", AsyncMock(return_value=[]))
    monkeypatch.setattr(client, "observe", AsyncMock(return_value=snapshot("running")))
    monkeypatch.setattr(client, "cancel", AsyncMock())
    monkeypatch.setattr(lifecycle, "adapter_stopped", AsyncMock(return_value="adapter gone"))
    monkeypatch.setattr(
        lifecycle.preparation, "prepare", AsyncMock(side_effect=JobBuildError("bad input"))
    )
    store = MemoryStore(state, cancel=cancel)
    store.execution.update(fields)
    write_execution, write_job = store.update, store.update_job

    async def update(job_id, **fields):
        if failing == "execution" and fields.get("state") == "cleanup":
            raise ConnectionError("database connection lost")
        await write_execution(job_id, **fields)

    async def update_job(job_id, **fields):
        if failing == "job" and "completed_at" in fields:
            raise ConnectionError("database connection lost")
        return await write_job(job_id, **fields)

    store.update = update
    store.repo.update_job = AsyncMock(side_effect=update_job)
    with pytest.raises(ConnectionError):
        await lifecycle.tick(store.job, store)
    assert (store.execution["state"], store.job["status"]) == left


@pytest.mark.asyncio
async def test_completion_wins_cancel_race_and_cleanup_is_retriable(monkeypatch):
    from evalhub.client import JobCanNotBeCancelledError

    store = MemoryStore("accepted", cancel=True)
    monkeypatch.setattr(
        client, "cancel", AsyncMock(side_effect=JobCanNotBeCancelledError("hub-123"))
    )
    monkeypatch.setattr(
        client, "observe", AsyncMock(return_value=snapshot("completed", metrics={"mae": 12.5}))
    )
    await lifecycle.tick(store.job, store)
    assert store.job["status"] == "succeeded"
    assert store.job["mlflow_run_id"] == "engine-run"
    monkeypatch.setattr(
        lifecycle.preparation, "cleanup", AsyncMock(side_effect=OSError("unavailable"))
    )
    with pytest.raises(OSError):
        await lifecycle.tick(store.job, store)
    assert store.execution["state"] == "cleanup"
    assert store.job["status"] == "succeeded"


@pytest.mark.asyncio
async def test_hub_get_outage_does_not_fail_remote_job(monkeypatch):
    store = MemoryStore("accepted")
    monkeypatch.setattr(client, "observe", AsyncMock(side_effect=httpx.ConnectError("offline")))
    with pytest.raises(httpx.ConnectError):
        await lifecycle.tick(store.job, store)
    assert store.job["status"] == "provisioning"
    assert store.execution["state"] == "accepted"


def fake_k8s(monkeypatch):
    """Hub's adapter Job and pod as the Kubernetes API returns them."""
    from types import SimpleNamespace

    import kubernetes_asyncio.client as k8s_client

    from amortized.backends.kubernetes import KubernetesBackend

    backend = KubernetesBackend.__new__(KubernetesBackend)
    backend._client = object()
    monkeypatch.setattr("amortized.core.compute.get_backend", lambda _: backend)
    state = SimpleNamespace(jobs=[], pods=[], log="", selectors=[])

    async def listed(items, _namespace, label_selector):
        state.selectors.append(label_selector)
        return SimpleNamespace(items=items())

    monkeypatch.setattr(
        k8s_client,
        "BatchV1Api",
        lambda _: SimpleNamespace(
            list_namespaced_job=lambda *a, **k: listed(lambda: state.jobs, *a, **k)
        ),
    )
    monkeypatch.setattr(
        k8s_client,
        "CoreV1Api",
        lambda _: SimpleNamespace(
            list_namespaced_pod=lambda *a, **k: listed(lambda: state.pods, *a, **k),
            read_namespaced_pod_log=AsyncMock(side_effect=lambda *a, **k: state.log),
        ),
    )
    return state


def k8s_job(ended="", age=0):
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    condition = SimpleNamespace(
        type=ended,
        status="True",
        reason="BackoffLimitExceeded" if ended == "Failed" else "Completed",
        last_transition_time=datetime.now(UTC) - timedelta(seconds=age),
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(name="hub-123-guid", labels={"benchmark_index": "0"}),
        status=SimpleNamespace(conditions=[condition] if ended else None),
    )


def k8s_pod(age=0, waiting=""):
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    reason = SimpleNamespace(reason=waiting, message="Back-off pulling image") if waiting else None
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="hub-123-guid-x7k2p",
            labels={"job-name": "hub-123-guid"},
            creation_timestamp=datetime.now(UTC) - timedelta(seconds=age),
        ),
        status=SimpleNamespace(
            init_container_statuses=None,
            container_statuses=[SimpleNamespace(state=SimpleNamespace(waiting=reason))],
        ),
    )


@pytest.mark.asyncio
async def test_adapter_job_watch_reads_hubs_labels_and_waits_out_its_graces(monkeypatch):
    from datetime import UTC, datetime

    k8s = fake_k8s(monkeypatch)
    execution = {"tenant": "user-namespace"}
    pending = snapshot("pending")  # Hub created it days ago
    k8s.jobs, k8s.pods = [k8s_job("Failed", age=120)], [k8s_pod()]
    k8s.log = (
        "INFO LMEval EvalHub Adapter\n"
        "pydantic_core._pydantic_core.ValidationError: 1 validation error for JobSpec\n"
        "benchmark_config\n  Field required"
    )
    reason = await lifecycle.adapter_stopped(execution, pending)
    assert reason.startswith("The benchmark adapter failed (BackoffLimitExceeded)")
    assert reason.endswith("ValidationError: 1 validation error for JobSpec")
    assert k8s.selectors[0] == "app=evalhub,component=evaluation-job,job_id=hub-123"
    # Hub may still be storing a report the adapter sent just before it exited.
    k8s.jobs = [k8s_job("Failed", age=10)]
    assert await lifecycle.adapter_stopped(execution, pending) == ""
    k8s.jobs = [k8s_job("Complete", age=120)]
    reported = snapshot("running")
    reported["status"]["benchmarks"] = [{"benchmark_index": 0, "status": "completed"}]
    assert await lifecycle.adapter_stopped(execution, reported) == ""
    k8s.log = ""
    assert await lifecycle.adapter_stopped(execution, pending) == (
        "The benchmark adapter exited without reporting a result to Eval Hub"
    )
    k8s.jobs, k8s.pods = [k8s_job()], [k8s_pod(age=900, waiting="ImagePullBackOff")]
    assert "cannot start: ImagePullBackOff: Back-off" in (
        await lifecycle.adapter_stopped(execution, pending)
    )
    k8s.pods = [k8s_pod(age=60, waiting="ImagePullBackOff")]
    assert await lifecycle.adapter_stopped(execution, pending) == ""
    k8s.jobs, k8s.pods = [], []
    assert "Job is gone" in await lifecycle.adapter_stopped(execution, pending)
    pending["resource"]["created_at"] = datetime.now(UTC).isoformat()  # Hub creates it async
    assert await lifecycle.adapter_stopped(execution, pending) == ""


@pytest.mark.asyncio
async def test_adapter_that_stops_without_reporting_fails_and_frees_the_model(monkeypatch):
    store = MemoryStore("accepted")
    monkeypatch.setattr(client, "observe", AsyncMock(return_value=snapshot("pending")))
    cancel = AsyncMock(side_effect=httpx.ConnectError("hub busy"))
    monkeypatch.setattr(client, "cancel", cancel)
    stopped = "The benchmark adapter failed (BackoffLimitExceeded) without reporting a result"
    monkeypatch.setattr(lifecycle, "adapter_stopped", AsyncMock(return_value=stopped))
    await lifecycle.tick(store.job, store)
    assert (store.job["status"], store.job["error"]) == ("failed", stopped)
    assert store.execution["state"] == "cleanup"
    cancel.assert_awaited_once()  # closes Hub's record; its failure does not matter
    cleanup = AsyncMock()
    monkeypatch.setattr(lifecycle.preparation, "cleanup", cleanup)
    assert await lifecycle.tick(store.job, store)
    cleanup.assert_awaited_once()  # the model server and its GPU go here


@pytest.mark.asyncio
async def test_a_result_hub_stores_during_the_job_watch_wins(monkeypatch):
    store = MemoryStore("accepted")
    done = snapshot("completed", metrics={"match": 0.9})
    monkeypatch.setattr(client, "observe", AsyncMock(side_effect=[snapshot("running"), done]))
    cancel = AsyncMock()
    monkeypatch.setattr(client, "cancel", cancel)
    stopped = "The benchmark adapter exited without reporting a result to Eval Hub"
    monkeypatch.setattr(lifecycle, "adapter_stopped", AsyncMock(return_value=stopped))
    await lifecycle.tick(store.job, store)
    assert (store.job["status"], store.execution["state"]) == ("succeeded", "cleanup")
    assert "error" not in store.job
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_job_watch_errors_leave_hub_authoritative(monkeypatch):
    store = MemoryStore("accepted")
    monkeypatch.setattr(client, "observe", AsyncMock(return_value=snapshot("running")))
    watch = AsyncMock(side_effect=RuntimeError("jobs is forbidden"))
    monkeypatch.setattr(lifecycle, "adapter_stopped", watch)
    await lifecycle.tick(store.job, store)
    assert (store.job["status"], store.execution["state"]) == ("running", "accepted")
    store.execution["cancel_requested"] = True  # the cancel path owns this job now
    monkeypatch.setattr(client, "cancel", AsyncMock())
    await lifecycle.tick(store.job, store)
    watch.assert_awaited_once()


def test_partial_results_keep_raw_units_and_artifact_references():
    execution = {
        "remote_id": "hub-123",
        "snapshot": snapshot("partially_failed", metrics={"mae": 12.5}),
    }
    result = present(execution)
    assert result["scores"]["mae"] == 12.5
    assert result["metric_metadata"]["mae"]["direction"] is None
    assert result["benchmarks"][0]["mlflow_run_id"] == "engine-run"


def test_fixed_scorers_are_gone_and_duplicate_criteria_are_rejected():
    with pytest.raises(ValidationError):
        TaskEvaluation(scorers=[{"id": "exact_match", "name": "match"}])
    for definition in ({"evaluation": {"kind": "task"}}, {}):
        with pytest.raises(ValidationError, match="rubric criterion names must be unique"):
            EvalJobRequest(
                endpoint={"base_url": "http://model/v1", "model": "m"},
                rubric=[{"name": "same", "description": "a"}, {"name": "same", "description": "b"}],
                **definition,
            )


@pytest.mark.asyncio
async def test_task_generates_once_and_does_not_leak_reference(tmp_path, monkeypatch):
    from amortized.evaluation import generation

    data = tmp_path / "data"
    data.mkdir()
    (data / "rows.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "secret answer"},
                ]
            }
        )
        + "\n"
    )
    generation_call = AsyncMock(return_value=[{"output": "secret answer", "error": ""}])
    monkeypatch.setattr(generation, "eval_endpoint", generation_call)
    monkeypatch.setattr(generation, "score_one", AsyncMock(return_value={"quality": 8.0}))
    summary = await generation.run(
        {
            "eval_data_path": str(data),
            "output_dir": str(tmp_path / "out"),
            "endpoints": {"model": {}, "judge": {}},
            "rubric": [{"name": "quality", "description": "test"}],
            "evaluation": {"kind": "task"},
        }
    )
    generation_call.assert_awaited_once()
    samples = generation_call.call_args.args[2]
    assert samples[0]["prompt"] == [{"role": "user", "content": "question"}]
    assert summary["results"]["scores"] == {"quality": 0.8}


@pytest.mark.asyncio
async def test_task_counts_samples_the_judge_gave_no_score(tmp_path, monkeypatch):
    from amortized.evaluation import generation

    data = tmp_path / "data"
    data.mkdir()
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    (data / "rows.jsonl").write_text((json.dumps(row) + "\n") * 2)
    monkeypatch.setattr(
        generation, "eval_endpoint", AsyncMock(return_value=[{"output": "a", "error": ""}] * 2)
    )
    # One sample scored, one where the judge errored (score_one then returns all None).
    monkeypatch.setattr(
        generation, "score_one", AsyncMock(side_effect=[{"quality": 8.0}, {"quality": None}])
    )
    summary = await generation.run(
        {
            "eval_data_path": str(data),
            "output_dir": str(tmp_path / "out"),
            "endpoints": {"model": {}, "judge": {}},
            "rubric": [{"name": "quality", "description": "test"}],
        }
    )
    assert summary["results"]["judge"] == {"num_samples": 2, "num_succeeded": 1, "error_rate": 0.5}
    assert summary["results"]["scores"] == {"quality": 0.8}


def test_native_eval_fails_when_the_judge_mostly_gave_no_score(tmp_path, monkeypatch):
    from amortized.evaluation import generation

    config = tmp_path / "config.json"
    config.write_text("{}")
    summary = {
        "results": {
            "model": {"num_samples": 3, "num_succeeded": 3, "error_rate": 0.0},
            "judge": {"num_samples": 3, "num_succeeded": 1, "error_rate": 0.6667},
        }
    }
    monkeypatch.setattr(generation, "run", AsyncMock(return_value=summary))
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--config", str(config)])
    with pytest.raises(SystemExit, match="judge gave no score for 67% of judged samples"):
        generation.main()


@pytest.fixture
async def hub_http_client(monkeypatch):
    import amortized.config as config_mod
    import amortized.db.connection as db_mod
    from amortized.main import app

    settings = config_mod.Settings(
        database_url=TEST_DATABASE_URL,
        compute_namespace="hub-test",
        compute_backend="kubernetes",
        evalhub_url="http://evalhub",
        evalhub_staging_bucket="test",
    )
    monkeypatch.setattr(config_mod, "settings", settings)
    monkeypatch.setattr(db_mod, "settings", settings)
    import amortized.core.jobs as core_jobs

    monkeypatch.setattr(core_jobs, "_settings", settings)
    env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
    subprocess.run(["alembic", "upgrade", "head"], check=True, capture_output=True, env=env)
    await db_mod.init_db()
    async with db_mod.get_pool().acquire() as conn:
        await conn.execute("TRUNCATE jobs, evalhub_executions")
    from amortized.api import jobs

    for name in (
        "_persist_metric_set",
        "_rubric_drift_warning",
        "_training_sdg_mirror_warning",
        "_eval_overlap_warning",
    ):
        monkeypatch.setattr(jobs, name, AsyncMock(return_value=[]))
    monkeypatch.setattr(client, "validate_definition", AsyncMock(return_value=[]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as transport:
        yield transport


TASK = {
    "endpoint": {"base_url": "http://model/v1", "model": "test"},
    "eval_data_run_id": "a" * 32,
    "evaluation": {"kind": "task"},
    "rubric": [{"name": "match", "description": "Matches the reference"}],
    "judge": {"base_url": "http://judge/v1", "model": "judge"},
}


@pytest.mark.asyncio
async def test_validate_is_read_only_then_confirm_creates_durable_intent(
    hub_http_client, monkeypatch
):
    from amortized.db.connection import get_pool

    submit = AsyncMock()
    monkeypatch.setattr(client, "submit", submit)
    validated = await hub_http_client.post("/api/v1/jobs/eval/validate", json=TASK)
    assert validated.status_code == 200, validated.text
    async with get_pool().acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM jobs") == 0
        assert await conn.fetchval("SELECT count(*) FROM evalhub_executions") == 0
    confirmed = await hub_http_client.post("/api/v1/jobs/eval", json=TASK)
    assert confirmed.status_code == 201, confirmed.text
    job = confirmed.json()
    assert job["execution_owner"] == "evalhub"
    async with get_pool().acquire() as conn:
        intent = await conn.fetchrow(
            "SELECT * FROM evalhub_executions WHERE job_id = $1", job["id"]
        )
        assert intent["state"] == "preparing"
        assert intent["tenant"] == "hub-test"
        assert intent["remote_id"] is None
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_warning_blocks_card_without_submitting(hub_http_client, monkeypatch):
    from amortized.api import jobs

    monkeypatch.setattr(
        jobs, "_eval_overlap_warning", AsyncMock(return_value=["Evaluation overlaps training data"])
    )
    submit = AsyncMock()
    monkeypatch.setattr(client, "submit", submit)
    response = await hub_http_client.post("/api/v1/jobs/eval/validate", json=TASK)
    assert response.status_code == 422
    assert "acknowledge_warnings" in response.text
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_create_gate_writes_no_hub_intent(hub_http_client):
    from amortized.db.connection import get_pool

    response = await hub_http_client.post(
        "/api/v1/jobs/eval", json={**TASK, "training_job_id": "missing-job"}
    )
    assert response.status_code == 422
    async with get_pool().acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM evalhub_executions") == 0


@pytest.mark.asyncio
async def test_sdk_submit_has_no_automatic_retry_and_uses_snapshot_tenant(monkeypatch):
    from evalhub import AsyncEvalHubClient

    calls = []

    def transport(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response", request=request)

    @asynccontextmanager
    async def sdk_client(*_, **__):
        async with AsyncEvalHubClient(
            base_url="http://hub", tenant="user-namespace", max_retries=0
        ) as sdk:
            sdk._client._transport = httpx.MockTransport(transport)
            sdk._client._mounts = {}
            yield sdk

    monkeypatch.setattr(client, "client", sdk_client)
    payload = {
        "name": "test",
        "model": {"url": "http://model", "name": "m"},
        "benchmarks": [{"id": "task", "provider_id": "amortized"}],
    }
    with pytest.raises(httpx.ReadTimeout):
        await client.submit(payload, {"hub_url": "http://hub", "tenant": "user-namespace"})
    assert len(calls) == 1
    assert calls[0].headers["X-Tenant"] == "user-namespace"


@pytest.mark.parametrize("status,expected", [(422, "cleanup"), (503, "uncertain")])
async def test_rejected_and_ambiguous_submissions_have_distinct_recovery(
    monkeypatch, status, expected
):
    store = MemoryStore()
    response = httpx.Response(status, request=httpx.Request("POST", "http://hub"))
    submit = AsyncMock(
        side_effect=httpx.HTTPStatusError("rejected", request=response.request, response=response)
    )
    monkeypatch.setattr(client, "submit", submit)
    await lifecycle.tick(store.job, store)
    assert store.execution["state"] == expected
    assert store.job["status"] == ("failed" if status == 422 else "provisioning")
    submit.assert_awaited_once()


async def test_cancel_api_ends_an_unclaimed_job_at_once(hub_http_client, monkeypatch):
    from amortized.db.connection import get_pool

    hub = AsyncMock(side_effect=AssertionError("Eval Hub was called"))
    monkeypatch.setattr(client, "cancel", hub)
    created = await hub_http_client.post("/api/v1/jobs/eval", json=TASK)
    job_id = created.json()["id"]
    cancelled = await hub_http_client.delete(f"/api/v1/jobs/{job_id}")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["completed_at"]
    async with get_pool().acquire() as conn:
        intent = await conn.fetchrow(
            "SELECT * FROM evalhub_executions WHERE job_id = $1", job_id
        )
        # No worker resumes a finished execution, and nothing was prepared to release.
        assert intent["state"] == "finished"
        assert intent["cancel_requested"] is True
        assert await Repository(conn).claim_job(job_id) is None
    hub.assert_not_awaited()


async def test_cancel_racing_the_claim_falls_back_to_recording_intent(monkeypatch):
    store = MemoryStore("preparing")
    monkeypatch.setattr(lifecycle, "EvalHubRepository", lambda _repo: store)
    # The API read the job while queued; the worker claimed it before the cancel wrote.
    updated = await lifecycle.request_cancel(store.repo, {**store.job, "status": "queued"})
    assert updated["status"] == "provisioning"
    assert store.execution["state"] == "preparing"
    assert store.execution["cancel_requested"]
    assert not await lifecycle.tick(store.job, store)
    assert store.job["status"] == "cancelled"
    assert store.execution["state"] == "cleanup"


async def test_cancel_api_records_intent_without_inventing_terminal_outcome(hub_http_client):
    from amortized.db.connection import get_pool

    created = await hub_http_client.post("/api/v1/jobs/eval", json=TASK)
    job_id = created.json()["id"]
    async with get_pool().acquire() as conn:
        assert await Repository(conn).claim_job(job_id) is not None
    cancelled = await hub_http_client.delete(f"/api/v1/jobs/{job_id}")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "provisioning"
    assert cancelled.json()["cancel_requested"] is True
    async with get_pool().acquire() as conn:
        intent = await conn.fetchrow(
            "SELECT * FROM evalhub_executions WHERE job_id = $1", job_id
        )
        assert intent["state"] == "preparing"


async def test_hub_eval_refuses_an_inline_api_key(hub_http_client):
    created = await hub_http_client.post(
        "/api/v1/jobs/eval",
        json={**TASK, "endpoint": {**TASK["endpoint"], "api_key": "private-key"}},
    )
    assert created.status_code == 422
    assert "private-key" not in created.text


async def test_restart_routes_hub_job_away_from_native_builder_and_primary_mlflow(monkeypatch):
    import amortized.worker as worker

    dispatch = AsyncMock()
    monkeypatch.setattr(lifecycle, "run_job", dispatch)
    job = MemoryStore().job
    await worker._run_job(job)
    await worker._resume_job(job)
    assert dispatch.await_count == 2


async def test_reconciliation_pages_and_rejects_other_tenants_and_nonexact_matches(monkeypatch):
    from evalhub import AsyncEvalHubClient

    execution = MemoryStore("uncertain").execution
    requests = []

    def transport(request):
        requests.append(request)
        if request.url.params["offset"] == "0":
            wrong_tenant = snapshot(remote_id="not-ours")
            wrong_tenant["resource"]["tenant"] = "another-user"
            wrong_tag = snapshot(remote_id="not-tagged")
            wrong_tag["tags"] = []
            return httpx.Response(200, json={"items": [wrong_tenant, wrong_tag], "total_count": 3})
        return httpx.Response(200, json={"items": [snapshot()], "total_count": 3})

    @asynccontextmanager
    async def sdk_client(url, tenant):
        async with AsyncEvalHubClient(base_url=url, tenant=tenant, max_retries=0) as sdk:
            sdk._client._transport = httpx.MockTransport(transport)
            sdk._client._mounts = {}
            yield sdk

    monkeypatch.setattr(client, "client", sdk_client)
    matches = await client.reconcile(execution)
    assert [item["resource"]["id"] for item in matches] == ["hub-123"]
    assert [r.url.params["offset"] for r in requests] == ["0", "2"]
    assert all(r.headers["X-Tenant"] == "user-namespace" for r in requests)


def test_only_task_definitions_are_accepted():
    with pytest.raises(ValidationError, match="kind"):
        EvalJobRequest(
            **{**TASK, "evaluation": {"kind": "benchmark", "provider_id": "p", "benchmark_id": "b"}}
        )


HUB_SETTINGS = {
    "compute_backend": "kubernetes",
    "compute_namespace": "tenant",
    "evalhub_url": "http://hub",
    "evalhub_staging_bucket": "snapshots",
    "mlflow_tracking_uri": "https://mlflow",
    "mlflow_tracking_token_file": "/var/run/secrets/mlflow/token",
    "mlflow_workspace": "tenant",
}


@pytest.mark.parametrize("image_field", ["Image", "image"])
async def test_preconfirmation_checks_catalog_and_routability_without_creating_resources(
    monkeypatch,
    image_field,
):
    from evalhub.models import Provider

    import amortized.config as config_mod

    monkeypatch.setattr(config_mod, "settings", config_mod.Settings(**HUB_SETTINGS))
    provider = Provider(
        resource={"id": "amortized"},
        name="Amortized",
        benchmarks=[{"id": "task", "name": "Task", "category": ""}],
        runtime={"k8s": {image_field: "adapter@sha256:pinned"}},
    )
    sdk = MagicMock()
    sdk.providers.get = AsyncMock(return_value=provider)

    @asynccontextmanager
    async def sdk_client():
        yield sdk

    monkeypatch.setattr(client, "client", sdk_client)
    assert await client.validate_definition(TASK) == []
    provider.runtime = {"k8s": {image_field: "adapter:latest"}}
    [unpinned] = await client.validate_definition(TASK)
    assert "pinned adapter image" in unpinned and SERVER_LIMIT in unpinned
    provider.runtime = {"k8s": {image_field: "adapter@sha256:pinned"}}
    assert (
        "loopback"
        in (
            await client.validate_definition(
                {**TASK, "endpoint": {"base_url": "http://localhost:8000/v1", "model": "m"}}
            )
        )[0]
    )
    # A malformed endpoint is a validation error, not a server error.
    assert await client.validate_definition(
        {**TASK, "endpoint": {"base_url": "https://model.test/v1"}}
    ) == ["endpoint: base_url and model are required"]
    assert await client.validate_definition(
        {**TASK, "endpoint_base": {"base_url": "https://model.test/v1"}}
    ) == ["Submit one model per evaluation using endpoint, rather than legacy endpoint pairs"]
    sdk.providers.get.side_effect = httpx.ConnectError("unavailable")
    assert "Cannot verify" in (await client.validate_definition(TASK))[0]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"compute_backend": "local"}, "Kubernetes backend"),
        ({"mlflow_tracking_uri": ""}, "no MLflow artifact storage"),
        ({"mlflow_tracking_token_file": ""}, "isn't set up with workspaces"),
        ({"mlflow_workspace": ""}, "isn't set up with workspaces"),
        ({"evalhub_staging_bucket": ""}, "no input staging bucket"),
    ],
)
async def test_server_limits_say_the_request_cannot_fix_them(monkeypatch, overrides, reason):
    import amortized.config as config_mod

    settings = config_mod.Settings(**{**HUB_SETTINGS, **overrides})
    monkeypatch.setattr(config_mod, "settings", settings)
    [error] = await client.validate_definition(TASK)
    assert reason in error and SERVER_LIMIT in error


@pytest.mark.parametrize("url", ["", "http://hub"], ids=["no url", "url"])
@pytest.mark.parametrize("accept", [True, False], ids=["accepting", "not accepting"])
@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "eval_on_gpu"])
def test_one_rule_decides_where_a_new_eval_runs(monkeypatch, url, accept, gpu):
    import amortized.config as config_mod
    from amortized.integrations.evalhub import accepts_new_jobs, runs_on_hub

    settings = config_mod.Settings(evalhub_url=url, evalhub_accept_new_jobs=accept)
    monkeypatch.setattr(config_mod, "settings", settings)
    assert accepts_new_jobs() is (bool(url) and accept)
    config = {"eval_mode": "classification", "eval_on_gpu": True} if gpu else {}
    assert runs_on_hub(config) is (bool(url) and accept and not gpu)


def test_helm_renders_the_new_jobs_flag_as_a_quoted_bool(monkeypatch):
    import amortized.config as config_mod

    monkeypatch.setenv("AMORTIZED_EVALHUB_ACCEPT_NEW_JOBS", "false")
    assert config_mod.Settings().evalhub_accept_new_jobs is False


def _create_repo():
    repo = MagicMock()
    repo.create_job = AsyncMock(side_effect=lambda **row: row)
    repo.conn.execute = AsyncMock()
    return repo


@pytest.mark.parametrize(
    ("accept", "extra", "owner"),
    [
        (True, {}, "evalhub"),
        (True, {"eval_mode": "classification", "eval_on_gpu": True}, "native"),
        (False, {}, "native"),
    ],
    ids=["task eval", "use_peft head", "new jobs off"],
)
async def test_create_routes_a_new_eval_by_the_same_rule(monkeypatch, accept, extra, owner):
    import amortized.config as config_mod
    from amortized.core import jobs as core_jobs
    from amortized.models import JobType

    settings = config_mod.Settings(**HUB_SETTINGS, evalhub_accept_new_jobs=accept, gpu_budget=1)
    monkeypatch.setattr(config_mod, "settings", settings)
    monkeypatch.setattr(core_jobs, "_settings", settings)  # read by the GPU budget check
    repo = _create_repo()
    config = {"training_job_id": "train-1", "eval_data_run_id": "a" * 32, **extra}
    row = await core_jobs.create_job(
        repo, job_type=JobType.eval, config=config, models_checked=True
    )
    assert row["execution_owner"] == owner
    # Only a Hub-owned job gets a Hub execution record.
    assert repo.conn.execute.await_count == (1 if owner == "evalhub" else 0)


async def test_a_use_peft_head_is_validated_estimated_and_created_as_a_native_gpu_eval(
    monkeypatch,
):
    import amortized.config as config_mod
    from amortized.api import jobs as jobs_api
    from amortized.core import jobs as core_jobs
    from amortized.models import JobType

    settings = config_mod.Settings(**HUB_SETTINGS, gpu_budget=1)
    monkeypatch.setattr(config_mod, "settings", settings)
    monkeypatch.setattr(core_jobs, "_settings", settings)  # read by the GPU budget check
    parent = {"type": "training", "config": {"algorithm": "classifier_sft", "use_peft": True}}
    monkeypatch.setattr(
        jobs_api, "Repository", lambda _db: MagicMock(get_job=AsyncMock(return_value=parent))
    )
    monkeypatch.setattr(client, "client", MagicMock(side_effect=AssertionError("Hub was read")))
    config = {"training_job_id": "train-1", "eval_data_run_id": "a" * 32}
    assert await jobs_api._apply_classification_eval_mode(config, db=None) == []
    assert config["eval_on_gpu"] is True
    assert await client.validate_definition(config) == []
    assert core_jobs.estimate_gpu_demand({"type": "eval", "config": config}) == 1
    assert core_jobs.gpu_budget_error(JobType.eval, config) is None
    row = await core_jobs.create_job(
        _create_repo(), job_type=JobType.eval, config=config, models_checked=True
    )
    assert row["execution_owner"] == "native"

    # The stamp is the server's: a client sending it can't move an eval off Hub.
    generative = {**TASK, "eval_on_gpu": True}
    assert await jobs_api._apply_classification_eval_mode(generative, db=None) == []
    assert "eval_on_gpu" not in generative


@pytest.mark.parametrize("cancel", [False, True], ids=["running", "cancel requested"])
@pytest.mark.parametrize("state", ["preparing", "ready", "uncertain", "accepted"])
async def test_a_hub_job_left_without_a_hub_url_ends_and_frees_what_it_holds(
    monkeypatch, state, cancel
):
    import amortized.config as config_mod

    monkeypatch.setattr(config_mod.settings, "evalhub_url", "")
    store = MemoryStore(state)
    # A submission was attempted, so a connected cleanup would reconcile with Hub.
    store.execution["observed_at"] = datetime.now(UTC) if state != "preparing" else None
    hub = AsyncMock(side_effect=AssertionError("Eval Hub was called"))
    for name in ("submit", "observe", "cancel", "reconcile"):
        monkeypatch.setattr(client, name, hub)
    monkeypatch.setattr(client, "client", MagicMock(side_effect=AssertionError("Hub client")))
    monkeypatch.setattr(lifecycle.preparation, "prepare", AsyncMock(side_effect=AssertionError))
    cleanup = AsyncMock()
    monkeypatch.setattr(lifecycle.preparation, "cleanup", cleanup)
    if cancel:
        monkeypatch.setattr(lifecycle, "EvalHubRepository", lambda _repo: store)
        await lifecycle.request_cancel(store.repo, store.job)
        assert store.execution["cancel_requested"]
    assert not await lifecycle.tick(store.job, store)
    if cancel and state in {"preparing", "ready"}:
        assert store.job["status"] == "cancelled"  # nothing reached Hub
    else:
        assert store.job["status"] == "failed"
        assert store.job["error"].startswith("Eval Hub is no longer configured on this server")
    assert store.execution["state"] == "cleanup"
    assert await lifecycle.tick(store.job, store)
    cleanup.assert_awaited_once()
    assert store.execution["state"] == "finished"
    hub.assert_not_awaited()


async def test_hub_taking_no_new_jobs_still_follows_its_jobs_to_the_end(monkeypatch):
    import amortized.config as config_mod

    monkeypatch.setattr(config_mod.settings, "evalhub_accept_new_jobs", False)
    store = MemoryStore("accepted", cancel=True)
    cancel = AsyncMock()
    monkeypatch.setattr(client, "cancel", cancel)
    monkeypatch.setattr(client, "observe", AsyncMock(return_value=snapshot("cancelled")))
    await lifecycle.tick(store.job, store)
    cancel.assert_awaited_once()
    assert store.job["status"] == "cancelled"


async def test_tuned_classifier_scores_with_its_training_settings():
    from amortized.integrations.evalhub import preparation

    trained = {"algorithm": "classifier_sft", "max_length": 256, "trust_remote_code": True}
    repo = MagicMock(get_job=AsyncMock(return_value={"config": json.dumps(trained)}))
    config = {"eval_mode": "classification", "training_job_id": "train-1"}
    assert await preparation._classifier_parameters(config, repo) == {
        "max_length": 256,
        "trust_remote_code": True,
    }
    repo.get_job.assert_awaited_once_with("train-1")
    # A base-model baseline (no training job) and a generative eval add nothing.
    assert await preparation._classifier_parameters({"eval_mode": "classification"}, repo) == {}
    assert await preparation._classifier_parameters({"training_job_id": "train-1"}, repo) == {}


@pytest.mark.asyncio
async def test_validate_api_refuses_hub_eval_when_mlflow_has_no_workspaces(
    hub_http_client, monkeypatch
):
    import amortized.config as config_mod

    monkeypatch.setattr(client, "validate_definition", REAL_VALIDATE)
    monkeypatch.setattr(config_mod.settings, "mlflow_tracking_uri", "http://mlflow")
    response = await hub_http_client.post("/api/v1/jobs/eval/validate", json=TASK)
    assert response.status_code == 422
    assert "workspace 'hub-test'" in response.text and SERVER_LIMIT in response.text


async def test_model_serving_survives_readiness_timeout_without_a_second_gpu_job(monkeypatch):
    from amortized.backends import BackendHandle, BackendStatus
    from amortized.backends.kubernetes import KubernetesBackend
    from amortized.integrations.evalhub import preparation
    from amortized.jobs import eval as eval_builder

    settings = preparation.config_mod.settings
    monkeypatch.setattr(settings, "forward_env", ["HF_TOKEN"])
    monkeypatch.setattr(settings, "mlflow_tracking_uri", "http://tracking")
    monkeypatch.setattr(settings, "mlflow_proxy_multipart_download", True)
    monkeypatch.setenv("HF_TOKEN", "test-model-download-token")
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", "http://object-storage")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test-temporary-storage-token")
    store = MemoryStore("preparing")
    store.job["config"] = {"model_name_or_path": "hf/model", "nproc_per_node": 1}
    backend = MagicMock(spec=KubernetesBackend)
    handle = BackendHandle(
        backend_name="kubernetes",
        job_id="product-123-serve",
        scheduler_id="amortized-product-123-serve",
    )
    backend.reconstruct_handle.return_value = handle
    backend.status = AsyncMock(
        side_effect=[BackendStatus(running=False, not_found=True), BackendStatus(running=True)]
    )
    backend.submit = AsyncMock(return_value=handle)
    monkeypatch.setattr(preparation, "get_backend", lambda _: backend)
    # The real serving builder preserves its model preparation and GPU checks.
    monkeypatch.setattr(
        preparation, "_build_embedded_serving", eval_builder._build_embedded_serving
    )
    original_client = httpx.AsyncClient
    attempts = 0

    def transport(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("model loading", request=request)
        return httpx.Response(200, json={"data": [{"id": "hf/model"}]})

    monkeypatch.setattr(
        preparation.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(transport), **kwargs),
    )
    with pytest.raises(httpx.ConnectError):
        await preparation.prepare_serving(store.job, store, await store.get("product-123"))
    endpoint = await preparation.prepare_serving(store.job, store, await store.get("product-123"))
    backend.submit.assert_awaited_once()
    submitted = backend.submit.call_args.args[0]
    assert submitted.resources.gpus == 1
    assert submitted.env["HF_TOKEN"] == "test-model-download-token"
    assert submitted.env["FSSPEC_S3_ENDPOINT_URL"] == "http://object-storage"
    assert submitted.env["AWS_SESSION_TOKEN"] == "test-temporary-storage-token"
    assert submitted.env["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] == "true"
    assert "run_eval.py" not in " ".join(submitted.command)
    assert endpoint["base_url"].endswith(":8000/v1")


async def test_comparisons_use_content_and_recipe_not_job_prefixes(monkeypatch):
    from amortized.integrations.evalhub import results

    job = {"id": "a", "config": TASK, "status": "succeeded", "created_at": "now"}
    execution = {
        "snapshot": snapshot("completed", metrics={"mae": 3.5}),
        "request": {},
        "inputs": {
            "dataset_identity": "sha256:dataset",
            "dataset_run_id": "run-a",
            "catalog": {"runtime": {"k8s": {"image": "adapter@sha256:one"}}},
        },
    }
    executions = {"a": copy.deepcopy(execution), "b": copy.deepcopy(execution)}
    executions["b"]["inputs"]["dataset_run_id"] = "run-b"
    store = MagicMock()
    store.get = AsyncMock(side_effect=lambda job_id: executions[job_id])
    monkeypatch.setattr(results, "EvalHubRepository", lambda _: store)
    other = {
        **job,
        "id": "b",
        "config": {**TASK, "endpoint": {**TASK["endpoint"], "model": "other"}},
    }
    groups = await results.comparison_groups(MagicMock(), [job, other])
    assert len(groups) == 1
    assert len(groups[0]["evals"]) == 2
    assert groups[0]["dataset"]["run_id"] == "run-a"
    executions["b"]["inputs"]["catalog"]["runtime"]["k8s"]["image"] = "adapter@sha256:two"
    assert len(await results.comparison_groups(MagicMock(), [job, other])) == 2


@pytest.mark.parametrize("status", ["failed", "cancelled", "running"])
async def test_only_succeeded_hub_runs_are_compared(monkeypatch, status):
    from amortized.integrations.evalhub import results

    execution = {
        "snapshot": snapshot("completed", metrics={"mae": 3.5}),
        "request": {},
        "inputs": {"dataset_identity": "sha256:dataset", "dataset_run_id": "run-a"},
    }
    store = MagicMock()
    store.get = AsyncMock(return_value=execution)
    monkeypatch.setattr(results, "EvalHubRepository", lambda _: store)
    job = {"id": "a", "config": TASK, "status": status, "created_at": "now"}
    assert await results.comparison_groups(MagicMock(), [job]) == []
    job["status"] = "succeeded"
    assert len(await results.comparison_groups(MagicMock(), [job])) == 1


def test_task_adapter_generates_scores_and_artifacts_using_sdk_contract(tmp_path, monkeypatch):
    from evalhub.adapter.models import JobSpec

    from amortized.evaluation import adapter, generation

    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "2 + 3?"},
                    {"role": "assistant", "content": "5"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setattr(adapter, "TASK_DATA_PATH", str(data))
    monkeypatch.setattr(
        adapter, "resolve_model_credentials", lambda: MagicMock(api_key="api-key:ref")
    )
    monkeypatch.setattr(
        generation, "eval_endpoint", AsyncMock(return_value=[{"output": "5", "error": ""}])
    )
    monkeypatch.setattr(generation, "score_one", AsyncMock(return_value={"accuracy": 8.0}))
    monkeypatch.setattr(
        adapter,
        "read_model_auth_key",
        {"judge_url": "http://judge/v1", "judge_api-key": "judge-key"}.get,
    )
    spec = JobSpec(
        id="hub-123",
        provider_id="amortized",
        benchmark_id="task",
        benchmark_index=0,
        callback_url="http://sidecar/callback",
        model={"url": "http://sidecar/v1", "name": "model"},
        parameters={
            "evaluation": {"kind": "task"},
            "rubric": [{"name": "accuracy", "description": "Matches the reference"}],
            "judge_model": "judge",
        },
    )
    instance = object.__new__(adapter.AmortizedTaskAdapter)
    results = instance.run_benchmark_job(spec, MagicMock())
    assert results.results[0].metric_name == "accuracy"
    assert results.results[0].metric_value == 0.8
    assert results.results[0].metadata["direction"] == "higher"
    assert results.results[0].metadata["scorer_id"] == "rubric_judge"
    assert [a.path for a in instance.mlflow_artifacts] == [
        "eval_results/metrics.json",
        "eval_results/results.jsonl",
    ]
    assert all(a.local_path.is_file() for a in instance.mlflow_artifacts)
    assert json.loads((instance.output_dir / "results.jsonl").read_text())["reference"] == "5"


def test_more_than_half_failed_is_decided_from_the_counts():
    from amortized.evaluation.generation import mostly_failed

    # 5001 of 10001 failed: the rounded rate reads 0.5, the counts say more than half.
    assert mostly_failed({"num_samples": 10001, "num_succeeded": 5000, "error_rate": 0.5})
    assert not mostly_failed({"num_samples": 10000, "num_succeeded": 5000, "error_rate": 0.5})
    assert mostly_failed({"num_samples": 0, "num_succeeded": 0, "error_rate": 1.0})
    assert not mostly_failed({"num_samples": 0, "num_succeeded": 0, "error_rate": 0.0})
    assert not mostly_failed(None)


def test_task_adapter_fails_when_every_judge_call_fails(tmp_path, monkeypatch):
    from evalhub.adapter.models import JobSpec

    from amortized.evaluation import adapter, generation

    data = tmp_path / "data"
    data.mkdir()
    row = {
        "messages": [{"role": "user", "content": "2 + 3?"}, {"role": "assistant", "content": "5"}]
    }
    (data / "sample.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(adapter, "TASK_DATA_PATH", str(data))
    monkeypatch.setattr(
        adapter, "resolve_model_credentials", lambda: MagicMock(api_key="api-key:ref")
    )
    monkeypatch.setattr(
        generation, "eval_endpoint", AsyncMock(return_value=[{"output": "5", "error": ""}])
    )
    # The judge endpoint answers 502: call_model raises, and score_one turns that into None.
    monkeypatch.setattr(
        generation, "call_model", AsyncMock(side_effect=RuntimeError("HTTP 502 Bad Gateway"))
    )
    monkeypatch.setattr(
        adapter,
        "read_model_auth_key",
        {"judge_url": "http://judge/v1", "judge_api-key": "judge-key"}.get,
    )
    spec = JobSpec(
        id="hub-123",
        provider_id="amortized",
        benchmark_id="task",
        benchmark_index=0,
        callback_url="http://sidecar/callback",
        model={"url": "http://sidecar/v1", "name": "model"},
        parameters={
            "evaluation": {"kind": "task"},
            "rubric": [{"name": "accuracy", "description": "Matches the reference"}],
            "judge_model": "judge",
        },
    )
    instance = object.__new__(adapter.AmortizedTaskAdapter)
    with pytest.raises(RuntimeError, match="Judge gave no score"):
        instance.run_benchmark_job(spec, MagicMock())


async def test_embedding_download_secret_obeys_forward_env_policy(monkeypatch):
    import kubernetes_asyncio.client as kube

    from amortized.backends.kubernetes import KubernetesBackend
    from amortized.integrations.evalhub import preparation

    store = MemoryStore("preparing")
    backend = MagicMock(spec=KubernetesBackend)
    backend._get_client = AsyncMock()
    api = MagicMock()
    api.create_namespaced_secret = AsyncMock()
    monkeypatch.setattr(preparation, "get_backend", lambda _: backend)
    monkeypatch.setattr(kube, "CoreV1Api", lambda _: api)
    monkeypatch.setenv("HF_TOKEN", "test-private-model-token")
    settings = preparation.config_mod.settings
    monkeypatch.setattr(settings, "forward_env", [])
    endpoint = {"base_url": "http://model/v1", "model": "private-embedding"}
    model = preparation._endpoint_spec({"endpoint": endpoint}, "endpoint")
    await preparation._credential_secret(store.job, store, model, None)
    assert "hf-token" not in api.create_namespaced_secret.call_args.args[1].string_data
    monkeypatch.setattr(settings, "forward_env", ["HF_TOKEN"])
    await preparation._credential_secret(store.job, store, model, None)
    assert (
        api.create_namespaced_secret.call_args.args[1].string_data["hf-token"]
        == "test-private-model-token"
    )


async def test_prepare_sends_a_provider_endpoint_with_its_own_key_and_the_chosen_dataset(
    monkeypatch,
):
    import kubernetes_asyncio.client as kube

    import amortized.config as config_mod
    import amortized.jobs.eval as eval_job
    from amortized.backends.kubernetes import KubernetesBackend
    from amortized.integrations.evalhub import preparation

    monkeypatch.setattr(config_mod, "settings", config_mod.Settings(**HUB_SETTINGS))
    provider = {"endpoint": "https://api.provider.test/v1", "api_key": "PROVIDER_API_KEY"}
    monkeypatch.setattr(
        eval_job, "resolve_provider", lambda name: provider if name == "openai" else None
    )
    monkeypatch.setenv("PROVIDER_API_KEY", "provider-key")
    backend = MagicMock(spec=KubernetesBackend)
    backend._get_client = AsyncMock()
    api = MagicMock()
    api.create_namespaced_secret = AsyncMock()
    monkeypatch.setattr(preparation, "get_backend", lambda _: backend)
    monkeypatch.setattr(kube, "CoreV1Api", lambda _: api)
    store = MemoryStore("preparing")
    store.job.update(
        parent_job_id="sdg-1",
        config={
            "endpoint": {"base_url": "openai", "model": "frontier"},
            "eval_data_run_id": "split-run",
            "rubric": [{"name": "match", "description": "Matches the reference"}],
        },
    )
    recorded_before_upload = []

    async def stage(job, repo, bucket):
        recorded_before_upload.append(store.execution["inputs"].get("staging_bucket"))
        assert bucket == "snapshots"
        return {"s3": {"bucket": bucket, "key": "k/"}}, "", "sha256:dataset", "split-run"

    monkeypatch.setattr(preparation, "stage_inputs", stage)
    execution = await store.get("product-123")
    execution["inputs"] = {"catalog": {}}
    payload = await preparation.prepare(store.job, store, execution)
    assert recorded_before_upload == ["snapshots"]
    assert store.execution["inputs"]["dataset_run_id"] == "split-run"
    assert payload["model"]["url"] == "https://api.provider.test/v1"
    assert payload["model"]["name"] == "frontier"
    secret = api.create_namespaced_secret.call_args.args[1].string_data
    assert secret["api-key"] == "provider-key"


async def test_https_model_secret_carries_cluster_ca(monkeypatch, tmp_path):
    import kubernetes_asyncio.client as kube

    from amortized.backends.kubernetes import KubernetesBackend
    from amortized.integrations.evalhub import preparation

    store = MemoryStore("preparing")
    backend = MagicMock(spec=KubernetesBackend)
    backend._get_client = AsyncMock()
    api = MagicMock()
    api.create_namespaced_secret = AsyncMock()
    monkeypatch.setattr(preparation, "get_backend", lambda _: backend)
    monkeypatch.setattr(kube, "CoreV1Api", lambda _: api)
    bundle = tmp_path / "service-ca.crt"
    bundle.write_text("-----BEGIN CERTIFICATE-----\ncluster\n-----END CERTIFICATE-----\n")
    settings = preparation.config_mod.settings
    monkeypatch.setattr(settings, "evalhub_ca_bundle", str(bundle))

    async def secret_for(url: str, judge_url: str = "") -> dict[str, str]:
        endpoint = {"base_url": url, "model": "granite"}
        model = preparation._endpoint_spec({"endpoint": endpoint}, "endpoint")
        judge = {"base_url": judge_url, "model": "judge"} if judge_url else None
        await preparation._credential_secret(store.job, store, model, judge)
        return api.create_namespaced_secret.call_args.args[1].string_data

    assert (await secret_for("https://granite.models.svc:8000/v1"))["ca_cert"] == (
        bundle.read_text()
    )
    assert "ca_cert" not in await secret_for("http://model.amortized.svc:8000/v1")
    # Hub's sidecar also proxies judge_url, so an https judge needs the CA too.
    assert "ca_cert" in await secret_for(
        "http://model.amortized.svc:8000/v1", "https://granite.models.svc:8000/v1"
    )
    monkeypatch.setattr(settings, "evalhub_ca_bundle", "")
    assert "ca_cert" not in await secret_for("https://granite.models.svc:8000/v1")


def test_embedding_adapter_uses_projected_token_and_canonical_anchor_query(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np
    from evalhub.adapter.models import JobSpec

    from amortized.evaluation import adapter

    data = tmp_path / "data"
    data.mkdir()
    rows = [{"text": f"{label} {i}", "label": label} for label in ("a", "b") for i in range(4)]
    (data / "sample.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(adapter, "TASK_DATA_PATH", str(data))
    monkeypatch.setattr(
        adapter, "resolve_model_credentials", lambda: SimpleNamespace(hf_token="test-hf-token")
    )
    monkeypatch.setitem(sys.modules, "training_hub.utils", None)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    class EmbeddingModel:
        def __init__(self, name):
            assert name == "private-embedding"
            assert os.environ["HF_TOKEN"] == "test-hf-token"

        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] if t.startswith("a") else [0.0, 1.0] for t in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=EmbeddingModel)
    )
    spec = JobSpec(
        id="hub-embedding",
        provider_id="amortized",
        benchmark_id="task",
        benchmark_index=0,
        callback_url="http://sidecar/callback",
        model={"url": "http://unused.invalid/v1", "name": "private-embedding"},
        parameters={"eval_mode": "classification", "anchors_per_class": 2},
    )
    instance = object.__new__(adapter.AmortizedTaskAdapter)
    results = instance.run_benchmark_job(spec, MagicMock())
    assert results.num_examples_evaluated == 4
    assert {result.metric_name: result.metric_value for result in results.results} == {
        "accuracy": 1.0,
        "macro_f1": 1.0,
    }
    assert len((instance.output_dir / "results.jsonl").read_text().splitlines()) == 4


def test_classifier_head_adapter_reports_head_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from evalhub.adapter.models import JobSpec

    from amortized.evaluation import adapter

    monkeypatch.setattr(
        adapter, "resolve_model_credentials", lambda: SimpleNamespace(hf_token=None)
    )
    received = {}

    def run(parameters):
        received.update(parameters)
        # The head path's shape (classify_eval._head_classifier_eval): no num_queries.
        return {
            "results": {
                "classifier_kind": "head",
                "scores": {"accuracy": 0.9, "macro_f1": 0.8},
                "model": {"num_samples": 10},
                "num_samples": 10,
            }
        }

    monkeypatch.setattr(adapter, "classification_scorer", lambda: SimpleNamespace(run=run))
    spec = JobSpec(
        id="hub-head",
        provider_id="amortized",
        benchmark_id="task",
        benchmark_index=0,
        callback_url="http://sidecar/callback",
        model={"url": "http://unused.invalid/v1", "name": "/test_data/model"},
        parameters={"eval_mode": "classification", "max_length": 256},
    )
    instance = object.__new__(adapter.AmortizedTaskAdapter)
    results = instance.run_benchmark_job(spec, MagicMock())
    assert received["model_path"] == "/test_data/model" and received["max_length"] == 256
    assert results.num_examples_evaluated == 10
    assert {r.metric_name: r.metric_value for r in results.results} == {
        "accuracy": 0.9,
        "macro_f1": 0.8,
    }
    assert {r.metadata["scorer_id"] for r in results.results} == {"classifier_head"}
    assert {r.metadata["num_samples"] for r in results.results} == {10}


async def test_failed_cancel_transport_still_observes_a_completed_engine_job(monkeypatch):
    store = MemoryStore("accepted", cancel=True)
    monkeypatch.setattr(client, "cancel", AsyncMock(side_effect=httpx.ReadTimeout("lost cancel")))
    monkeypatch.setattr(client, "observe", AsyncMock(return_value=snapshot("completed")))
    await lifecycle.tick(store.job, store)
    assert store.job["status"] == "succeeded"
    assert store.execution["state"] == "cleanup"


async def test_database_disconnect_reconnects_without_ending_the_admitted_task(monkeypatch):
    import amortized.db.connection as db_mod

    attempts = 0
    connection = MagicMock()
    connection.fetchval = AsyncMock(return_value=True)
    connection.execute = AsyncMock()

    @asynccontextmanager
    async def acquire():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("database connection lost")
        yield connection

    monkeypatch.setattr(db_mod, "get_pool", lambda: MagicMock(acquire=acquire))
    monkeypatch.setattr(lifecycle, "tick", AsyncMock(return_value=True))
    monkeypatch.setattr(lifecycle.asyncio, "sleep", AsyncMock())
    await lifecycle.run_job(MemoryStore().job)
    assert attempts == 2
    lifecycle.tick.assert_awaited_once()


def lock_pool(monkeypatch, locked=True):
    import amortized.db.connection as db_mod

    connection = MagicMock()
    connection.fetchval = AsyncMock(return_value=locked)
    connection.execute = AsyncMock()
    connection.held = 0

    @asynccontextmanager
    async def acquire():
        connection.held += 1
        try:
            yield connection
        finally:
            connection.held -= 1

    monkeypatch.setattr(db_mod, "get_pool", lambda: MagicMock(acquire=acquire))
    return connection


async def test_a_hub_job_holds_a_database_connection_only_while_it_ticks(monkeypatch):
    connection = lock_pool(monkeypatch)

    async def sleep(_):
        assert connection.held == 0
        assert "pg_advisory_unlock(" in connection.execute.await_args.args[0]

    monkeypatch.setattr(lifecycle, "tick", AsyncMock(side_effect=[False, False, True]))
    monkeypatch.setattr(lifecycle.asyncio, "sleep", AsyncMock(side_effect=sleep))
    await lifecycle.run_job(MemoryStore().job)
    assert lifecycle.asyncio.sleep.await_count == 2
    assert connection.fetchval.await_count == 3
    assert "pg_try_advisory_lock(" in connection.fetchval.await_args.args[0]


async def test_a_hub_job_locked_by_another_worker_is_left_to_it(monkeypatch):
    lock_pool(monkeypatch, locked=False)
    monkeypatch.setattr(lifecycle, "tick", AsyncMock())
    await lifecycle.run_job(MemoryStore().job)
    lifecycle.tick.assert_not_awaited()


def test_adapter_main_persists_exactly_one_primary_run_then_reports_results(monkeypatch):
    from amortized.evaluation import adapter

    instance = MagicMock()
    results = MagicMock()
    instance.run_benchmark_job.return_value = results
    callbacks = MagicMock()
    callbacks.mlflow.save.return_value = "one-primary-run"
    monkeypatch.setattr(adapter, "AmortizedTaskAdapter", lambda: instance)
    monkeypatch.setattr(adapter.DefaultCallbacks, "from_adapter", lambda _: callbacks)
    adapter.main()
    callbacks.mlflow.save.assert_called_once_with(
        results, instance.job_spec, artifacts=instance.mlflow_artifacts
    )
    assert results.mlflow_run_id == "one-primary-run"
    callbacks.report_results.assert_called_once_with(results)
