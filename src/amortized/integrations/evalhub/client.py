"""SDK transport with non-retrying submission and server-owned tenant mapping."""

import logging
from contextlib import asynccontextmanager
from typing import Any, cast

from evalhub import AsyncEvalHubClient
from evalhub.models import JobSubmissionRequest

import amortized.config as config_mod
from amortized.integrations.evalhub import runs_on_hub, server_limit

logger = logging.getLogger(__name__)


@asynccontextmanager
async def client(hub_url: str = "", tenant: str = "") -> Any:
    settings = config_mod.settings
    origin = hub_url or settings.evalhub_url
    if settings.evalhub_token_file and not origin.startswith("https://"):
        raise ValueError("Refusing to send Eval Hub credentials to an insecure saved origin")
    # Construct per operation: the SDK reads bearer-token files at construction,
    # so sharing a long-lived instance would freeze a rotating SA token.
    async with AsyncEvalHubClient(
        base_url=origin,
        tenant=tenant or settings.evalhub_tenant or settings.compute_namespace,
        auth_token_path=settings.evalhub_token_file or None,
        ca_bundle_path=settings.evalhub_ca_bundle or None,
        timeout=settings.evalhub_timeout,
        max_retries=0,
    ) as sdk:
        yield sdk


async def submit(payload: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    async with client(execution["hub_url"], execution["tenant"]) as sdk:
        result = await sdk.jobs.submit(JobSubmissionRequest.model_validate(payload))
        return cast(dict[str, Any], result.model_dump(mode="json", exclude_none=True))


async def observe(execution: dict[str, Any]) -> dict[str, Any]:
    async with client(execution["hub_url"], execution["tenant"]) as sdk:
        result = await sdk.jobs.get(execution["remote_id"])
        return cast(dict[str, Any], result.model_dump(mode="json", exclude_none=True))


async def cancel(execution: dict[str, Any]) -> None:
    async with client(execution["hub_url"], execution["tenant"]) as sdk:
        await sdk.jobs.cancel(execution["remote_id"], hard_delete=False)


async def reconcile(execution: dict[str, Any]) -> list[dict[str, Any]]:
    matches = []
    offset = 0
    async with client(execution["hub_url"], execution["tenant"]) as sdk:
        while True:
            # SDK jobs.list omits name/tags/offset. Use its authenticated request
            # transport to page the documented REST API, rather than a second client.
            response = await sdk._request_get(
                "/evaluations/jobs",
                params={"name": execution["correlation"], "limit": 100, "offset": offset},
            )
            data = response.json()
            items = data.get("items") or []
            for item in items:
                if (
                    item.get("name") == execution["correlation"]
                    and execution["correlation"] in (item.get("tags") or [])
                    and item.get("resource", {}).get("tenant", execution["tenant"])
                    == execution["tenant"]
                ):
                    matches.append(item)
            offset += len(items)
            if not items or offset >= data.get("total_count", offset):
                return matches


async def logs(execution: dict[str, Any], tail: int) -> list[str]:
    from evalhub.client import JobLogOptions

    async with client(execution["hub_url"], execution["tenant"]) as sdk:
        content = await sdk.jobs.get_logs(
            execution["remote_id"], options=JobLogOptions(tail_lines=tail)
        )
        return cast(list[str], content.splitlines()[-tail:])


async def validate_definition(config: dict[str, Any]) -> list[str]:
    """Read-only pre-confirmation catalog/capability checks."""
    definition = config.get("evaluation") or {}
    settings = config_mod.settings
    if not runs_on_hub(config):
        return (
            ["evaluation definitions require the configured Eval Hub engine"] if definition else []
        )
    if settings.resolved_default_backend != "kubernetes":
        return server_limit("Eval Hub evaluations", "they need the Kubernetes backend")
    if not settings.mlflow_tracking_uri:
        return server_limit("Eval Hub evaluations", "no MLflow artifact storage is configured")
    # Hub saves every evaluation into the MLflow workspace named after its job namespace.
    # Amortized sends a workspace only with the token, and an unset one falls back to the
    # server pod's namespace; a set one must equal the jobs namespace (config.py).
    if not (settings.mlflow_tracking_token_file and settings.mlflow_workspace):
        return server_limit(
            "Eval Hub evaluations",
            "its MLflow isn't set up with workspaces (Eval Hub saves into workspace "
            f"{settings.compute_namespace!r}; set AMORTIZED_MLFLOW_TRACKING_TOKEN_FILE and "
            "AMORTIZED_MLFLOW_WORKSPACE)",
        )
    if not settings.evalhub_staging_bucket:
        return server_limit("task evaluations", "no input staging bucket is configured")
    from urllib.parse import urlparse

    from amortized.jobs.base import JobBuildError
    from amortized.jobs.eval import _endpoint_spec

    if any(config.get(key) for key in ("endpoint_base", "endpoint_tuned")):
        return ["Submit one model per evaluation using endpoint, rather than legacy endpoint pairs"]
    for key in ("endpoint", "judge"):
        if not config.get(key):
            continue
        try:
            spec = _endpoint_spec(config, key)
        except JobBuildError as exc:
            return [str(exc)]
        if spec["provider_type"] != "openai":
            return [
                f"{key}: use an OpenAI-compatible gateway for this provider; "
                "Hub's credential proxy cannot inject native Anthropic x-api-key headers"
            ]
        parsed = urlparse(spec["base_url"])
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return [f"{key}: expected a routable HTTP(S) endpoint"]
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return [f"{key}: an endpoint URL can't carry credentials, a query or a fragment"]
        if parsed.hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
            return [f"{key}: loopback endpoints cannot be reached from Eval Hub workloads"]
    provider_id, benchmark_id = settings.evalhub_task_provider, "task"
    try:
        async with client() as sdk:
            provider = await sdk.providers.get(provider_id)
        if benchmark_id not in {b.id for b in provider.benchmarks}:
            return server_limit(
                "task evaluations",
                f"Eval Hub's provider {provider_id!r} has no benchmark {benchmark_id!r}",
            )
        runtime = (provider.runtime or {}).get("k8s") or {}
        # Hub 1.0.3 serializes this Go struct field as "Image" in JSON,
        # although provider YAML uses "image". Accept both wire formats.
        image = runtime.get("Image") or runtime.get("image", "")
        if "@sha256:" not in image:
            return server_limit(
                "task evaluations",
                f"Eval Hub's provider {provider_id!r} has no pinned adapter image digest",
            )
    except Exception:
        logger.warning("Could not read Eval Hub's task provider %r", provider_id, exc_info=True)
        return [
            f"Cannot verify the task evaluation against Eval Hub's provider {provider_id!r};"
            " retry validation"
        ]
    return []
