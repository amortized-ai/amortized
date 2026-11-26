"""Run a job's model-liveness checks: ask its builder which LLM endpoints it will call, then
probe them — in-process by default, under the checker's process-wide probe limit, or through the
opt-in standalone check service (over verified TLS) when ``model_check_url`` is set.

Guarantees callers can rely on:
- one total time budget for the whole check, however many endpoints there are;
- exactly one result per endpoint checked;
- checker availability problems are ``unverified``, never ``passed`` and never ``failed``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import weakref
from dataclasses import dataclass, field
from typing import Any

import httpx

import amortized.config as config_mod
from amortized.core.job_env import forwarded_env
from amortized.core.model_endpoints import (
    FAILED,
    PASSED,
    UNVERIFIED,
    CheckResult,
    ConfigError,
    ModelEndpoint,
    approved_destinations,
    dedup,
    redact,
    redact_url,
)
from amortized.jobs import get_builder

logger = logging.getLogger("amortized.core.model_check_client")

CHUNK_SIZE = 8  # the service's per-request endpoint limit
MAX_CONCURRENT_CHUNKS = 2  # <= 16 probes per check, so two checks fit the 32-probe limit
# Part of the remaining budget kept back for the service's response to arrive: the probes get
# the rest, so the whole call — not just the probes — finishes inside the budget.
RESPONSE_OVERHEAD_SECONDS = 2.0
MIN_CHUNK_BUDGET_SECONDS = 1.0

_OUTCOMES = {PASSED, FAILED, UNVERIFIED}


@dataclass
class CheckReport:
    config_errors: list[ConfigError] = field(default_factory=list)
    results: list[CheckResult] = field(default_factory=list)


def _unverified(ep: ModelEndpoint, reason: str, detail: str) -> CheckResult:
    return CheckResult(ep.label, ep.model, redact_url(ep.base_url), UNVERIFIED, reason, detail)


def _from_service(ep: ModelEndpoint, item: Any) -> CheckResult:
    """Trust only the outcome fields of a service result; identity comes from our endpoint."""
    if (
        not isinstance(item, dict)
        or item.get("label") != ep.label
        or item.get("outcome") not in _OUTCOMES
        or not isinstance(item.get("reason"), str)
    ):
        return _unverified(ep, "checker_unavailable", "malformed result from model check service")
    return CheckResult(
        ep.label,
        ep.model,
        redact_url(ep.base_url),
        item["outcome"],
        item["reason"],
        redact(str(item.get("detail") or ""), [ep.api_key]),
    )


async def _check_chunk(
    chunk: list[ModelEndpoint], deadline: float, timeout: float
) -> list[CheckResult]:
    """Check one chunk: probes get ``deadline``; the call as a whole gets ``timeout``."""
    settings = config_mod.settings
    url = model_check_service_url()
    if not url:
        return await _check_in_process(chunk, deadline)

    payload = {
        "deadline_seconds": deadline,
        "endpoints": [ep.to_request() for ep in chunk],
        "approved_urls": list(approved_destinations()),  # the service enforces the policy too
    }
    # Verified TLS: this request carries provider keys (settings refuse an http:// URL).
    verify: str | bool = settings.model_check_ca_bundle or True
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, verify=verify
        ) as client:
            resp = await client.post(url.rstrip("/") + "/check", json=payload)
    except httpx.HTTPError as exc:
        logger.warning(
            "model check service unreachable (%s); %d endpoint(s) unverified",
            type(exc).__name__,
            len(chunk),
        )
        detail = f"model check service unreachable ({type(exc).__name__})"
        return [_unverified(ep, "checker_unavailable", detail) for ep in chunk]

    if resp.status_code == 503:
        logger.warning("model check service busy; %d endpoint(s) unverified", len(chunk))
        return [_unverified(ep, "checker_busy", "model check service is busy") for ep in chunk]
    if resp.status_code != 200:
        logger.warning("model check service returned HTTP %d", resp.status_code)
        detail = f"model check service returned HTTP {resp.status_code}"
        return [_unverified(ep, "checker_unavailable", detail) for ep in chunk]
    try:
        items = resp.json().get("results")
    except (ValueError, AttributeError):
        items = None
    if not isinstance(items, list):
        detail = "malformed response from model check service"
        return [_unverified(ep, "checker_unavailable", detail) for ep in chunk]
    if len(items) != len(chunk):
        logger.warning("model check service returned %d results for %d", len(items), len(chunk))
    return [_from_service(ep, items[i] if i < len(items) else None) for i, ep in enumerate(chunk)]


_SA_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


def _own_namespace() -> str:
    """The namespace this server runs in: POD_NAMESPACE (downward API), else the service
    account's namespace file. Empty outside Kubernetes."""
    namespace = os.environ.get("POD_NAMESPACE", "").strip()
    if namespace:
        return namespace
    try:
        with open(_SA_NAMESPACE_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def model_check_service_url() -> str:
    """The standalone check service's URL, with ``{namespace}`` replaced by this server's own.
    Its certificate names amortized-model-check.<namespace>.svc, and a kustomize ``namespace:``
    overlay can't rewrite a namespace inside a URL string."""
    url = config_mod.settings.model_check_url
    if "{namespace}" in url:
        namespace = _own_namespace()
        if not namespace:
            logger.warning("model_check_url uses {namespace}, but this server's is unknown")
        url = url.replace("{namespace}", namespace)
    return url


async def _check_in_process(chunk: list[ModelEndpoint], deadline: float) -> list[CheckResult]:
    """Probe from this process, under the checker's process-wide limit (shared by every
    validation the server runs at once) — refused, not queued, when it's full."""
    from amortized.model_check import checker

    if not checker.try_reserve(len(chunk)):
        logger.warning("model check busy; %d endpoint(s) unverified", len(chunk))
        return [_unverified(ep, "checker_busy", "too many model checks running") for ep in chunk]
    try:
        return await checker.check_endpoints(chunk, deadline, approved_destinations())
    finally:
        checker.release(len(chunk))


async def run_checks(endpoints: list[ModelEndpoint], budget: float) -> list[CheckResult]:
    """Check ``endpoints`` within one total, wall-clock ``budget``: chunks of CHUNK_SIZE, at most
    MAX_CONCURRENT_CHUNKS at a time, each given the budget remaining when it starts. Anything
    still running when the budget ends is cancelled and reported unverified."""
    loop = asyncio.get_running_loop()
    budget_end = loop.time() + budget
    chunks = [endpoints[i : i + CHUNK_SIZE] for i in range(0, len(endpoints), CHUNK_SIZE)]
    slots = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)

    async def run_chunk(index: int, chunk: list[ModelEndpoint]) -> list[CheckResult]:
        async with slots:
            remaining = budget_end - loop.time()
            # The first wave starts immediately and always gets the whole budget, however small
            # it's configured. Only a chunk that had to wait for a slot can find too little left.
            if index >= MAX_CONCURRENT_CHUNKS and remaining < MIN_CHUNK_BUDGET_SECONDS:
                detail = "model check budget ran out before this endpoint could be checked"
                return [_unverified(ep, "budget_exhausted", detail) for ep in chunk]
            overhead = min(RESPONSE_OVERHEAD_SECONDS, remaining / 4)
            return await _check_chunk(
                chunk, deadline=max(remaining - overhead, 0.1), timeout=max(remaining, 0.1)
            )

    tasks = [asyncio.create_task(run_chunk(i, chunk)) for i, chunk in enumerate(chunks)]
    try:
        done, _ = await asyncio.wait(tasks, timeout=max(budget_end - loop.time(), 0.0))
    finally:
        # Also on cancellation of this call (asyncio.wait doesn't cancel its tasks): nothing may
        # keep probing, or start a queued chunk, after run_checks has returned or raised.
        unfinished = [t for t in tasks if not t.done()]
        for task in unfinished:
            task.cancel()
        await asyncio.gather(*unfinished, return_exceptions=True)

    results: list[CheckResult] = []
    for task, chunk in zip(tasks, chunks, strict=True):
        if task in done and task.exception() is None:
            results.extend(task.result())
        elif task in done:
            logger.warning("model check failed unexpectedly (%s)", type(task.exception()).__name__)
            detail = "model check failed unexpectedly"
            results.extend(_unverified(ep, "checker_unavailable", detail) for ep in chunk)
        else:
            detail = "the model check did not finish within its time budget"
            results.extend(_unverified(ep, "budget_exhausted", detail) for ep in chunk)
    return results


async def check_job_models(job_type: str, config: dict[str, Any]) -> CheckReport:
    """Config errors and probe results for the LLM endpoints a job of ``job_type`` will call.

    A builder opts in by defining ``model_checks(config, env)``; job types without it have
    nothing to check. Config errors short-circuit the probe — they already block the job."""
    builder = get_builder(job_type)
    hook = getattr(builder, "model_checks", None) if builder is not None else None
    if hook is None:
        return CheckReport()
    endpoints, config_errors = hook(config, forwarded_env())
    if config_errors:
        return CheckReport(config_errors=list(config_errors))
    endpoints = dedup(list(endpoints))
    if not endpoints:
        return CheckReport()
    budget = config_mod.settings.model_check_deadline_seconds
    # Probes start at once, alongside the denial lookups: one endpoint's slow DNS must not eat
    # the budget another endpoint needs to give its answer.
    probes = asyncio.create_task(run_checks(endpoints, budget))
    try:
        denied = await _destination_denials(endpoints, budget)
    except BaseException:
        probes.cancel()
        raise
    if denied:
        probes.cancel()
        await asyncio.gather(probes, return_exceptions=True)
        return CheckReport(config_errors=denied)
    return CheckReport(results=await probes)


MAX_CONCURRENT_RESOLUTIONS = 16
_resolution_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


async def _destination_denials(endpoints: list[ModelEndpoint], budget: float) -> list[ConfigError]:
    """Policy denials don't depend on the endpoint answering, so they're decided here rather than
    only by the probe, which a busy or unavailable checker never runs. Lookups share the check's
    budget; one unresolved or still pending when it ends is left to the probe."""
    from amortized.model_check import destinations

    approved = approved_destinations()
    loop = asyncio.get_running_loop()
    slots = _resolution_slots.setdefault(loop, asyncio.Semaphore(MAX_CONCURRENT_RESOLUTIONS))

    async def decide(ep: ModelEndpoint) -> destinations.Decision:
        url = destinations.chat_request_url(ep.base_url, ep.provider_type)
        literal = destinations.literal_address(url)
        if literal is not None:  # no lookup needed, so no slot
            return destinations.decide(url, [literal], approved)
        async with slots:
            return await destinations.authorize(url, approved)

    tasks = [asyncio.create_task(decide(ep)) for ep in endpoints]
    try:
        done, _ = await asyncio.wait(tasks, timeout=budget)
    finally:
        unfinished = [t for t in tasks if not t.done()]
        for task in unfinished:
            task.cancel()
        await asyncio.gather(*unfinished, return_exceptions=True)
    errors: list[ConfigError] = []
    for ep, task in zip(endpoints, tasks, strict=True):
        if task not in done or task.cancelled() or task.exception() is not None:
            continue
        decision = task.result()
        if decision.verdict == destinations.DENIED:
            errors.append(
                ConfigError(
                    ep.label,
                    "destination_denied",
                    f"{ep.label} model {ep.model!r} at {redact_url(ep.base_url)}"
                    f" {decision.reason}. Use a public HTTPS endpoint, or ask an operator to"
                    " approve the service (AMORTIZED_APPROVED_MODEL_URLS)",
                )
            )
    return errors
