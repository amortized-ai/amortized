"""Contracts at approval, durable dispatch, engine ownership and task scoring."""

import copy
import json
import os
import subprocess
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from conftest import TEST_DATABASE_URL
from pydantic import ValidationError

from amortized.evaluation.spec import TaskEvaluation
from amortized.integrations.evalhub import SERVER_LIMIT, client, lifecycle
from amortized.integrations.evalhub.results import present
from amortized.models import EvalJobRequest

REAL_VALIDATE = client.validate_definition


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


class MemoryStore:
    def __init__(self, state="ready", cancel=False):
        self.execution = {
            "job_id": "product-123",
            "tenant": "user-namespace",
            "hub_url": "http://hub",
            "correlation": "amortized-product-123",
            "state": state,
            "request": {},
            "cancel_requested": cancel,
            "remote_id": "hub-123" if state == "accepted" else None,
            "snapshot": {},
            "sync_error": "",
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
        self.conn.transaction.return_value.__aenter__ = AsyncMock()
        self.conn.transaction.return_value.__aexit__ = AsyncMock()
        self.repo = MagicMock()
        self.repo.update_job = AsyncMock(side_effect=self.update_job)
        self.repo.get_job = AsyncMock(side_effect=lambda _: self.job.copy())
        self.repo.scrub_eval_credentials = AsyncMock()

    async def get(self, _):
        return copy.deepcopy(self.execution)

    async def update(self, _, **fields):
        self.execution.update(fields)

    async def cancel(self, _):
        self.execution["cancel_requested"] = True

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
    store = MemoryStore("uncertain", cancel=True)
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


def test_the_catalog_metric_is_the_score_and_listed_first():
    metrics = {"acc": 0.6, "acc_norm": 0.4, "acc_stderr": 0.11}
    execution = {
        "snapshot": snapshot("completed", metrics=metrics),
        "request": {"benchmarks": [{"id": "task", "primary_score": {"metric": "acc_norm"}}]},
    }
    result = present(execution)
    assert list(result["scores"]) == ["acc_norm", "acc", "acc_stderr"]
    assert (result["primary_metric"], result["primary_score"]) == ("acc_norm", 0.4)
    # Submitted before Amortized sent primary_score: the catalog Hub took it from.
    catalog = {"benchmarks": [{"id": "task", "primary_score": {"metric": "acc_norm"}}]}
    legacy = {**execution, "request": {}, "inputs": {"catalog": catalog}}
    assert present(legacy)["primary_score"] == 0.4
    # LightEval's arc:easy names acc_norm but reports only acc: no stand-in score.
    only_acc = snapshot("completed", metrics={"all.acc": 0.5, "arc:easy.acc": 0.5})
    result = present({**execution, "snapshot": only_acc})
    assert (result["primary_metric"], result["primary_score"]) == ("acc_norm", None)
    assert list(result["scores"]) == ["all.acc", "arc:easy.acc"]


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
    with pytest.raises(ValidationError, match="rubric criterion names must be unique"):
        EvalJobRequest(
            endpoint={"base_url": "http://model/v1", "model": "m"},
            evaluation={"kind": "task"},
            rubric=[{"name": "same", "description": "a"}, {"name": "same", "description": "b"}],
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


BENCHMARK_RECIPE = {
    "version": 1,
    "provider_id": "amortized-lighteval",
    "benchmark_id": "gsm8k",
    "catalog_entry": {"id": "gsm8k", "name": "GSM8K", "primary_score": {"metric": "m"}},
    "adapter_digest": "sha256:pinned",
    "parameters": {"provider": "endpoint", "num_few_shot": 8, "save_sample_results": False},
    "model_serving": "endpoint",
    "declared_capabilities": [],
}


@pytest.mark.asyncio
async def test_benchmark_requires_neither_task_dataset_nor_judge(hub_http_client, monkeypatch):
    from amortized.integrations.evalhub.catalog import fingerprint

    recipe = copy.deepcopy(BENCHMARK_RECIPE)
    monkeypatch.setattr(client, "benchmark_recipe", AsyncMock(return_value=(recipe, [])))
    request = {
        "endpoint": TASK["endpoint"],
        "evaluation": {
            "kind": "benchmark",
            "provider_id": "amortized-lighteval",
            "benchmark_id": "gsm8k",
        },
    }
    validated = await hub_http_client.post("/api/v1/jobs/eval/validate", json=request)
    assert validated.status_code == 200, validated.text
    card = validated.json()
    evaluation = card["config"]["evaluation"]
    assert evaluation["recipe_fingerprint"] == fingerprint(recipe)
    assert evaluation["parameters"]["num_few_shot"] == 8
    assert "recipe" not in evaluation or evaluation["recipe"] is None
    assert card["benchmark"]["name"] == "GSM8K"
    assert card["benchmark"]["recipe_fingerprint"] == fingerprint(recipe)

    # Confirm posts the card's config verbatim; a client-sent recipe is never trusted.
    forged = {**evaluation, "recipe": {"parameters": {"num_few_shot": 0}}}
    response = await hub_http_client.post(
        "/api/v1/jobs/eval", json={**card["config"], "evaluation": forged}
    )
    assert response.status_code == 201, response.text
    assert response.json()["config"]["evaluation"]["recipe"] == recipe

    # The setup changed between the card and Confirm: refuse, ask for a new card.
    recipe["adapter_digest"] = "sha256:changed"
    stale = await hub_http_client.post("/api/v1/jobs/eval", json=card["config"])
    assert stale.status_code == 422
    assert "changed since validation" in stale.text


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


async def test_cancel_api_records_intent_without_inventing_terminal_outcome(hub_http_client):
    created = await hub_http_client.post("/api/v1/jobs/eval", json=TASK)
    job_id = created.json()["id"]
    cancelled = await hub_http_client.delete(f"/api/v1/jobs/{job_id}")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "queued"
    assert cancelled.json()["cancel_requested"] is True


async def test_prepared_credentials_are_erased_from_both_config_snapshots(
    hub_http_client, monkeypatch
):
    from amortized.db.connection import get_pool
    from amortized.db.evalhub import EvalHubRepository
    from amortized.db.repository import Repository

    created = await hub_http_client.post(
        "/api/v1/jobs/eval",
        json={**TASK, "endpoint": {**TASK["endpoint"], "api_key": "private-key"}},
    )
    assert "private-key" not in created.text
    job_id = created.json()["id"]
    payload = {"name": "amortized-" + job_id, "model": {"auth": {"secret_ref": "owned"}}}
    monkeypatch.setattr(lifecycle.preparation, "prepare", AsyncMock(return_value=payload))
    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        store = EvalHubRepository(repo)
        await lifecycle.tick(await repo.get_job(job_id), store)
        saved = await repo.get_job(job_id)
        assert "api_key" not in saved["config"]["endpoint"]
        assert "api_key" not in json.loads(saved["request_config"])["endpoint"]
        execution = await store.get(job_id)
        assert execution["state"] == "ready"
        assert execution["request"] == payload


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


def test_benchmark_parameters_reject_code_loading_and_secrets():
    from amortized.evaluation.spec import BenchmarkEvaluation

    for parameters in (
        {"provider": "transformers"},
        {"trust_remote_code": True},
        {"api_key": "secret"},
        {"random_seed": 12},
    ):
        with pytest.raises(ValidationError):
            BenchmarkEvaluation(
                kind="benchmark",
                provider_id="lighteval",
                benchmark_id="gsm8k",
                parameters=parameters,
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
    sdk.providers.get.side_effect = httpx.ConnectError("unavailable")
    assert "Cannot verify" in (await client.validate_definition(TASK))[0]
    # Benchmark catalog membership/compatibility is checked when the recipe is pinned
    # (tests/test_evalhub_catalog.py); here only the provider must be enabled.
    benchmark = {
        "endpoint": TASK["endpoint"],
        "evaluation": {
            "kind": "benchmark",
            "provider_id": "amortized-lighteval",
            "benchmark_id": "gsm8k",
        },
    }
    assert await client.validate_definition(benchmark) == []
    disabled = {**benchmark["evaluation"], "provider_id": "lighteval"}
    assert (
        "not enabled"
        in (await client.validate_definition({**benchmark, "evaluation": disabled}))[0]
    )


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
            "catalog": {"runtime": {"image": "adapter@sha256:one"}},
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
    executions["b"]["inputs"]["catalog"]["runtime"]["image"] = "adapter@sha256:two"
    assert len(await results.comparison_groups(MagicMock(), [job, other])) == 2


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
    await preparation._credential_secret(store.job, store, endpoint, None)
    assert "hf-token" not in api.create_namespaced_secret.call_args.args[1].string_data
    monkeypatch.setattr(settings, "forward_env", ["HF_TOKEN"])
    await preparation._credential_secret(store.job, store, endpoint, None)
    assert (
        api.create_namespaced_secret.call_args.args[1].string_data["hf-token"]
        == "test-private-model-token"
    )


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


ADAPTER_MAIN = """
import json
import sys
from datetime import UTC, datetime

from evalhub.adapter import EvaluationResult, FrameworkAdapter, JobResults


class Adapter(FrameworkAdapter):
    def run_benchmark_job(self, config, callbacks):
        # As the TrustyAI lm-eval adapter: the first metric, and lm-eval's own first metric.
        self._run_info = {"primary_metric": "acc", "zero_shot": 0.6, "alt_prompting": None,
                          "alt_prompting_description": None}
        results = [EvaluationResult(metric_name="acc", metric_value=0.6),
                   EvaluationResult(metric_name="acc_norm", metric_value=0.4)]
        return JobResults(id="j", benchmark_id="arc_easy", benchmark_index=0, model_name="m",
                          results=results, overall_score=0.6, num_examples_evaluated=20,
                          duration_seconds=1.0, completed_at=datetime.now(UTC),
                          additional_info={"alt_prompting": 0.6,
                                           "alt_prompting_description": "5-Shot"})


def main():
    adapter = Adapter.__new__(Adapter)
    results = adapter.run_benchmark_job(None, None)
    print(json.dumps({"score": results.overall_score, "run_info": adapter._run_info,
                      "info": results.additional_info, "file": __file__, "argv": sys.argv}))
    return 3


if __name__ == "__main__":
    raise SystemExit("main.py must be run through main(), not as __main__")
"""


@pytest.mark.parametrize("metric,score", [("acc_norm", 0.4), ("mc1", None), (None, None)])
def test_adapter_launcher_scores_by_the_job_specs_primary_metric(tmp_path, metric, score):
    import sys
    from pathlib import Path

    launcher = Path(__file__).parents[1] / "deploy/helm/amortized/files/evalhub/adapter_launcher.py"
    (tmp_path / "main.py").write_text(ADAPTER_MAIN)
    spec = {"id": "j", "primary_score": {"metric": metric} if metric else None}
    (tmp_path / "job.json").write_text(json.dumps(spec))
    done = subprocess.run(
        [sys.executable, "-c", launcher.read_text(), str(tmp_path / "main.py")],
        env={**os.environ, "EVALHUB_JOB_SPEC_PATH": str(tmp_path / "job.json")},
        cwd=tmp_path.parent,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 3, done.stderr  # main()'s own exit code
    out = json.loads(done.stdout)
    assert out["score"] == score
    assert out["run_info"]["primary_metric"] == metric
    assert (out["run_info"]["zero_shot"], out["run_info"]["alt_prompting"]) == (score, None)
    assert out["info"]["alt_prompting"] == score  # few-shot runs carry it as alt_prompting
    assert out["file"] == str(tmp_path / "main.py")  # main.py's output/ stays beside it
    assert out["argv"] == [str(tmp_path / "main.py")]
