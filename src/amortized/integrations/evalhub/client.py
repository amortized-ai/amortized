"""SDK transport with non-retrying submission and server-owned tenant mapping."""

from contextlib import asynccontextmanager
from typing import Any, cast

from evalhub import AsyncEvalHubClient
from evalhub.models import JobSubmissionRequest

import amortized.config as config_mod
from amortized.integrations.evalhub import server_limit


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
    if not settings.evalhub_url:
        return (
            ["evaluation definitions require the configured Eval Hub engine"] if definition else []
        )
    if settings.resolved_default_backend != "kubernetes":
        return ["Eval Hub execution requires the Kubernetes preparation backend"]
    benchmark = definition.get("kind") == "benchmark"
    from amortized.evaluation.spec import PROVIDER_PARAMETERS

    if benchmark and definition["provider_id"] not in PROVIDER_PARAMETERS:
        return ["This benchmark provider needs an implemented parameter/capability contract"]
    if not settings.mlflow_tracking_uri:
        return ["Eval Hub evaluation requires configured MLflow artifact storage"]
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
    if not benchmark and not settings.evalhub_staging_bucket:
        return ["Eval Hub task evaluation requires a configured input staging bucket"]
    from urllib.parse import urlparse

    from amortized.jobs.eval import _endpoint_spec

    for key in ("endpoint", "endpoint_base", "endpoint_tuned", "judge"):
        if not config.get(key):
            continue
        spec = _endpoint_spec(config, key, "UNUSED", {})
        if spec["provider_type"] != "openai":
            return [
                f"{key}: use an OpenAI-compatible gateway for this provider; "
                "Hub's credential proxy cannot inject native Anthropic x-api-key headers"
            ]
        parsed = urlparse(spec["base_url"])
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return [f"{key}: expected a routable HTTP(S) endpoint"]
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return [f"{key}: credentials belong in api_key, not endpoint URLs"]
        if parsed.hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
            return [f"{key}: loopback endpoints cannot be reached from Eval Hub workloads"]
    if any(config.get(key) for key in ("endpoint_base", "endpoint_tuned")):
        return ["Submit one model per evaluation using endpoint, rather than legacy endpoint pairs"]
    if benchmark:
        if definition["provider_id"] not in settings.evalhub_providers:
            return [f"benchmark provider {definition['provider_id']!r} is not enabled"]
        # Catalog membership, compatibility and settings are checked when the
        # recipe is pinned (benchmark_recipe), from the same catalog read.
        return []
    provider_id, benchmark_id = settings.evalhub_task_provider, "task"
    try:
        async with client() as sdk:
            provider = await sdk.providers.get(provider_id)
        if benchmark_id not in {b.id for b in provider.benchmarks}:
            return [f"benchmark {benchmark_id!r} is not in provider {provider_id!r}"]
        runtime = (provider.runtime or {}).get("k8s") or {}
        # Hub 1.0.3 serializes this Go struct field as "Image" in JSON,
        # although provider YAML uses "image". Accept both wire formats.
        image = runtime.get("Image") or runtime.get("image", "")
        if "@sha256:" not in image:
            return [f"provider {provider_id!r} must use a pinned adapter image digest"]
    except Exception:
        return [
            "Cannot verify the selected benchmark against the Eval Hub catalog; retry validation"
        ]
    return []


async def benchmark_recipe(config: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve the exact recipe a benchmark job would run, or the reasons it cannot."""
    from amortized.integrations.evalhub import catalog

    definition = config.get("evaluation") or {}
    try:
        capability = catalog.Capability.from_model(
            {**config, "declared_capabilities": definition.get("declared_capabilities")}
        )
    except ValueError as exc:
        return None, [str(exc)]
    try:
        async with client() as sdk:
            entries = await catalog.load(sdk, [definition["provider_id"]])
    except Exception:
        return None, [
            "Cannot verify the selected benchmark against the Eval Hub catalog; retry validation"
        ]
    return catalog.resolve_recipe(entries, capability, definition, config)
