"""Reliable engine handoff and cancellation, without an evaluator bridge pod.

Once POST may have reached Hub, only exact correlation reconciliation can
recover its identity. An empty list is never permission to POST again.
"""

import asyncio
import logging
import re
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

import httpx
from evalhub.client import JobCanNotBeCancelledError

import amortized.config as config_mod
from amortized.db.evalhub import EvalHubRepository
from amortized.db.repository import Repository
from amortized.integrations.evalhub import client, preparation
from amortized.jobs.base import JobBuildError

logger = logging.getLogger(__name__)
_TERMINAL = {
    "completed": "succeeded",
    "failed": "failed",
    "partially_failed": "failed",
    "cancelled": "cancelled",
}

# Hub 1.0.3 learns an outcome only from the adapter's own reports and never watches the
# adapter's Kubernetes Job. An adapter that dies before reporting leaves Hub's job open for
# good, and with it the model server Amortized started, so Amortized watches the Job.
_EXIT_GRACE = 60  # seconds for Hub to store a report the adapter sent just before exiting
_START_GRACE = 600  # seconds for Hub to create the Job, or for its pod to get started
_STUCK = {
    "ErrImagePull",
    "ImagePullBackOff",
    "InvalidImageName",
    "CreateContainerConfigError",
    "CreateContainerError",
}
_UNWATCHED: set[str] = set()  # jobs whose Job watch failed, so the warning is logged once


def projection(snapshot: dict[str, Any]) -> dict[str, Any]:
    status = snapshot.get("status") or {}
    remote_state = status.get("state", "pending")
    if remote_state not in {*_TERMINAL, "pending", "running"}:
        raise ValueError(f"Unrecognized Hub execution state: {remote_state}")
    fields: dict[str, Any] = {
        "status": _TERMINAL.get(
            remote_state, "running" if remote_state == "running" else "provisioning"
        ),
    }
    benchmarks = status.get("benchmarks") or []
    started = [b["started_at"] for b in benchmarks if b.get("started_at")]
    completed = [b["completed_at"] for b in benchmarks if b.get("completed_at")]
    if started:
        fields["started_at"] = min(started)
    if remote_state in _TERMINAL:
        fields["completed_at"] = max(completed) if completed else datetime.now(UTC)
    if remote_state in {"failed", "partially_failed", "cancelled"}:
        message = status.get("message") or {}
        fields["error"] = message.get("message") or f"Eval Hub evaluation {remote_state}"
    runs = [
        b["mlflow_run_id"]
        for b in (snapshot.get("results") or {}).get("benchmarks", [])
        if b.get("mlflow_run_id")
    ]
    if runs:
        fields["mlflow_run_id"] = runs[0]
    return fields


async def _record(
    store: EvalHubRepository, execution: dict[str, Any], snapshot: dict[str, Any]
) -> bool:
    fields = projection(snapshot)
    terminal = fields["status"] in {"succeeded", "failed", "cancelled"}
    async with store.conn.transaction():
        await store.update(
            execution["job_id"],
            remote_id=snapshot["resource"]["id"],
            state="cleanup" if terminal else "accepted",
            snapshot=snapshot,
            sync_error="",
            observed_at=datetime.now(UTC),
        )
        await store.repo.update_job(execution["job_id"], **fields)
    return terminal


def _label_value(value: str) -> str:
    """Hub's sanitizeLabelValue (runtimes/k8s/job_names.go), for its job_id label."""
    safe = re.sub(r"[^a-z0-9-_.]+", "-", value.lower())[:63].strip("-_.")
    return safe or "x"


def _age(timestamp: Any, now: datetime) -> float:
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (now - timestamp).total_seconds() if timestamp else 0.0


async def _last_log_line(core: Any, namespace: str, pod: str) -> str:
    try:
        text = await core.read_namespaced_pod_log(
            pod, namespace, container="adapter", tail_lines=50
        )
    except Exception:
        return ""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    errors = [line for line in lines if re.search(r"(Error|Exception)\b", line)]
    return (errors or lines or [""])[-1][:300]


async def adapter_stopped(execution: dict[str, Any], snapshot: dict[str, Any]) -> str:
    """Why Hub's adapter can no longer report this job's result, or "" while it can."""
    from kubernetes_asyncio.client import BatchV1Api, CoreV1Api

    from amortized.backends.kubernetes import KubernetesBackend
    from amortized.core.compute import get_backend

    backend = get_backend("kubernetes")
    if not isinstance(backend, KubernetesBackend):
        return ""
    reported = {
        b.get("benchmark_index")
        for b in (snapshot.get("status") or {}).get("benchmarks") or []
        if b.get("status") in {"completed", "failed", "cancelled"}
    }
    namespace, now = execution["tenant"], datetime.now(UTC)
    job_id = _label_value(snapshot["resource"]["id"])
    selector = f"app=evalhub,component=evaluation-job,job_id={job_id}"
    api = await backend._get_client()
    core = CoreV1Api(api)
    jobs = (await BatchV1Api(api).list_namespaced_job(namespace, label_selector=selector)).items
    if not jobs:
        if _age((snapshot.get("resource") or {}).get("created_at"), now) > _START_GRACE:
            return "The benchmark adapter's Kubernetes Job is gone and Eval Hub has no result"
        return ""
    pods = (await core.list_namespaced_pod(namespace, label_selector=selector)).items
    for k8s_job in jobs:
        index = (k8s_job.metadata.labels or {}).get("benchmark_index")
        if index is not None and int(index) in reported:
            continue
        ended = next(
            (
                c
                for c in k8s_job.status.conditions or []
                if c.type in {"Failed", "Complete"} and c.status == "True"
            ),
            None,
        )
        if ended is None or _age(ended.last_transition_time, now) <= _EXIT_GRACE:
            continue
        name = k8s_job.metadata.name
        pod_name = next(
            (p.metadata.name for p in pods if (p.metadata.labels or {}).get("job-name") == name),
            "",
        )
        last = await _last_log_line(core, namespace, pod_name) if pod_name else ""
        how = f"failed ({ended.reason or 'Failed'})" if ended.type == "Failed" else "exited"
        return f"The benchmark adapter {how} without reporting a result to Eval Hub" + (
            f". Last log line: {last}" if last else ""
        )
    for pod in pods:
        if _age(pod.metadata.creation_timestamp, now) <= _START_GRACE:
            continue
        statuses = (pod.status.init_container_statuses or []) + (
            pod.status.container_statuses or []
        )
        for status in statuses:
            waiting = status.state.waiting if status.state else None
            if waiting and waiting.reason in _STUCK:
                return (
                    f"The benchmark adapter pod cannot start: {waiting.reason}"
                    + (f": {waiting.message}" if waiting.message else "")
                )[:500]
    return ""


async def request_cancel(repo: Repository, job: dict[str, Any]) -> dict[str, Any]:
    store = EvalHubRepository(repo)
    execution = await store.get(job["id"])
    if execution is None:
        raise ValueError("Hub execution mapping is missing")
    await store.cancel(job["id"])
    # The worker persists/forwards this intent even if the service is down or
    # the submit response is lost. Never fabricate a cancelled terminal status.
    if execution.get("remote_id"):
        try:
            await client.cancel(execution)
        except JobCanNotBeCancelledError:
            # Completion can win this race. GET remains authoritative.
            pass
        except Exception:
            await store.update(
                job["id"], sync_error="Cancellation requested; engine acknowledgment pending"
            )
    updated = await repo.get_job(job["id"])
    assert updated is not None
    return updated


async def tick(job: dict[str, Any], store: EvalHubRepository) -> bool:
    """One restart-safe reconciliation step. True only when no more work remains."""
    execution = await store.get(job["id"])
    if execution is None:
        raise ValueError("Missing durable Hub submission intent")
    state = execution["state"]
    if state == "finished":
        return True
    if state == "cleanup":
        await preparation.cleanup(execution)
        await store.update(job["id"], state="finished", sync_error="")
        return True
    if state == "preparing":
        if execution["cancel_requested"]:
            await store.repo.update_job(
                job["id"], status="cancelled", completed_at=datetime.now(UTC)
            )
            await store.update(job["id"], state="cleanup")
            return False
        try:
            created = execution.get("created_at")
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created and (datetime.now(UTC) - created).total_seconds() > (
                config_mod.settings.evalhub_preparation_timeout
            ):
                raise JobBuildError("Eval Hub input/model preparation exceeded its deadline")
            payload = await preparation.prepare(job, store, execution)
        except JobBuildError as exc:
            await store.repo.update_job(
                job["id"], status="failed", error=str(exc), completed_at=datetime.now(UTC)
            )
            await store.update(job["id"], state="cleanup")
            return False
        async with store.conn.transaction():
            await store.update(job["id"], state="ready", request=payload, sync_error="")
            await store.repo.scrub_eval_credentials(job["id"])
        return False
    if state == "ready":
        if execution["cancel_requested"]:
            await store.repo.update_job(
                job["id"], status="cancelled", completed_at=datetime.now(UTC)
            )
            await store.update(job["id"], state="cleanup")
            return False
        # Commit BEFORE sending. A crash anywhere after this write cannot cause
        # a second POST, even if no response/mapping was saved.
        await store.update(job["id"], state="submitting")
        try:
            snapshot = await client.submit(execution["request"], execution)
            await _record(store, execution, snapshot)
        except httpx.HTTPStatusError as exc:
            if 400 <= exc.response.status_code < 500 and exc.response.status_code not in {408, 409}:
                await store.repo.update_job(
                    job["id"],
                    status="failed",
                    error=f"Hub rejected submission (HTTP {exc.response.status_code})",
                    completed_at=datetime.now(UTC),
                )
                await store.update(job["id"], state="cleanup")
            else:
                await store.update(
                    job["id"],
                    state="uncertain",
                    sync_error="Submission outcome unknown; reconciling without resubmission",
                )
        except Exception:
            await store.update(
                job["id"],
                state="uncertain",
                sync_error="Submission outcome unknown; reconciling without resubmission",
            )
        return False
    if state in {"submitting", "uncertain"}:
        matches = await client.reconcile(execution)
        if len(matches) == 1:
            await _record(store, execution, matches[0])
        else:
            await store.update(
                job["id"],
                state="uncertain",
                sync_error=(
                    "No exact submission match visible yet; automatic resubmission is disabled"
                    if not matches
                    else "Multiple exact submission matches; operator reconciliation required"
                ),
            )
        return False
    if state == "accepted":
        if execution["cancel_requested"]:
            try:
                await client.cancel(execution)
            except JobCanNotBeCancelledError:
                pass
            except Exception:
                # Cancellation transport is not the execution verdict. Still
                # observe: the evaluation may have completed in the meantime.
                logger.warning("Hub cancellation acknowledgment pending for %s", job["id"])
        snapshot = await client.observe(execution)
        if await _record(store, execution, snapshot) or execution["cancel_requested"]:
            return False
        try:
            stopped = await adapter_stopped(execution, snapshot)
        except Exception:
            # Watching the Job is a safety net; Hub's own state stays authoritative.
            if job["id"] not in _UNWATCHED:
                _UNWATCHED.add(job["id"])
                logger.warning("Cannot watch the Hub adapter Job for %s", job["id"], exc_info=True)
            stopped = ""
        if stopped:
            # Closes Hub's open record; there is no adapter left to stop.
            with suppress(Exception):
                await client.cancel(execution)
            await store.repo.update_job(
                job["id"], status="failed", error=stopped, completed_at=datetime.now(UTC)
            )
            await store.update(job["id"], state="cleanup", sync_error="")
        return False
    raise ValueError(f"Unknown handoff state: {state}")


async def run_job(job: dict[str, Any]) -> None:
    from amortized.db.connection import get_pool

    while True:
        try:
            async with get_pool().acquire() as conn:
                # A session lock prevents concurrent POST. A connection loss
                # releases it; reconnect and inspect durable state, never resend.
                key = f"evalhub:{job['id']}"
                locked = await conn.fetchval(
                    "SELECT pg_try_advisory_lock(hashtextextended($1, 0))", key
                )
                if not locked:
                    return
                store = EvalHubRepository(Repository(conn))
                try:
                    while True:
                        try:
                            if await tick(job, store):
                                return
                        except Exception:
                            logger.warning(
                                "Hub synchronization deferred for job %s", job["id"], exc_info=True
                            )
                            await store.update(
                                job["id"], sync_error="Execution synchronization pending; retrying"
                            )
                        await asyncio.sleep(config_mod.settings.evalhub_poll_interval)
                finally:
                    with suppress(Exception):
                        await conn.execute(
                            "SELECT pg_advisory_unlock(hashtextextended($1, 0))", key
                        )
        except Exception:
            # Keep this task (and its resource admission) alive during DB loss.
            logger.warning("Reconnecting Hub intent storage for %s", job["id"], exc_info=True)
            await asyncio.sleep(config_mod.settings.evalhub_poll_interval)
