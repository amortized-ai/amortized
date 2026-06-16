"""MCP server using fastmcp with httpx ASGI transport.

Replaces the fastapi-mcp auto-generated server with hand-crafted tools
that have proper descriptions, annotations, and structured error handling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from fastmcp import FastMCP
from httpx import ASGITransport
from mcp.types import ToolAnnotations

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("amortized.mcp")

mcp = FastMCP(
    name="amortized",
    instructions=(
        "Amortized is an AI model customization runtime. "
        "Use these tools to submit training/SDG/eval/serve jobs, "
        "track progress, view logs and metrics, manage artifacts, "
        "run evaluations, configure compute backends, and discover "
        "recipes and capabilities."
    ),
)

_fastapi_app: FastAPI | None = None


async def _call(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
) -> Any:
    """Make an in-process HTTP call to the FastAPI app via ASGI transport.

    Translates HTTP 4xx/5xx into ``ValueError`` with a structured message
    so that fastmcp surfaces the error to the MCP client cleanly.
    """
    if _fastapi_app is None:
        raise RuntimeError("MCP server not initialised; call create_mcp_server first")

    transport = ASGITransport(app=_fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mcp") as client:
        response = await client.request(method, path, params=params, json=json_body)

    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:
            raise ValueError(f"HTTP {response.status_code}: {response.text}") from None

        msg = body.get("message", body.get("detail", f"HTTP {response.status_code}"))
        if isinstance(msg, list):
            msg = "; ".join(str(m) for m in msg)
        details = body.get("details", [])
        if details:
            detail_strs = [d.get("msg", str(d)) for d in details if isinstance(d, dict)]
            if detail_strs:
                msg = f"{msg} ({'; '.join(detail_strs)})"
        raise ValueError(msg)

    if response.status_code == 204:
        return {}
    return response.json()


async def _call_upload(
    path: str,
    file_path: str,
    *,
    data: dict[str, str] | None = None,
) -> Any:
    """Upload a file via multipart form to the FastAPI app."""
    if _fastapi_app is None:
        raise RuntimeError("MCP server not initialised; call create_mcp_server first")

    p = Path(file_path)
    if not p.exists():
        raise ValueError(f"File not found: {file_path}")

    transport = ASGITransport(app=_fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mcp") as client:
        with open(p, "rb") as f:
            files = {"file": (p.name, f, "application/octet-stream")}
            response = await client.post(path, files=files, data=data or {})

    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:
            raise ValueError(f"HTTP {response.status_code}: {response.text}") from None
        msg = body.get("message", body.get("detail", f"HTTP {response.status_code}"))
        if isinstance(msg, list):
            msg = "; ".join(str(m) for m in msg)
        raise ValueError(msg)

    return response.json()


# ---------------------------------------------------------------------------
# Tools -- job management
# ---------------------------------------------------------------------------

_SDG_EXAMPLE = (
    '{"model": "openai/gpt-4o-mini", "num_samples": 50, '
    '"strategy_params": {"generated_attributes": [{"id": "question", '
    '"instruction_messages": [{"role": "user", "content": '
    '"Generate a diverse question about science."}]}], '
    '"passthrough_attributes": ["question"]}}'
)


@mcp.tool(
    description=(
        "Submit a new job to the Amortized runtime. "
        "Supported types: training, sdg, eval, serve.\n\n"
        "For training jobs, config must include: algorithm, model_name_or_path, data_path.\n"
        "For SDG jobs, config must include: model, num_samples.\n\n"
        f"Example SDG config:\n{_SDG_EXAMPLE}"
    ),
    annotations=ToolAnnotations(
        destructiveHint=False,
        idempotentHint=False,
    ),
)
async def submit_job(
    type: str,
    config: dict[str, Any],
    backend: str = "local",
    gpus: int = 0,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Submit a training, sdg, eval, or serve job."""
    compute: dict[str, Any] = {"backend": backend, "gpus": gpus}
    if gpu_type:
        compute["gpu_type"] = gpu_type

    payload: dict[str, Any] = {
        "type": type,
        "config": config,
        "compute": compute,
        "dry_run": False,
    }
    result: dict[str, Any] = await _call("POST", "/api/v1/jobs", json_body=payload)
    return result


@mcp.tool(
    description=(
        "List jobs, optionally filtered by status and/or type. "
        "Status values: queued, running, succeeded, failed, cancelled. "
        "Type values: training, sdg, eval, serve."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_jobs(
    status: str | None = None,
    type: str | None = None,
) -> list[dict[str, Any]]:
    """List all jobs with optional status/type filters."""
    params: dict[str, str] = {}
    if status:
        params["status"] = status
    if type:
        params["type"] = type
    result: list[dict[str, Any]] = await _call("GET", "/api/v1/jobs", params=params or None)
    return result


@mcp.tool(
    description="Get full details for a single job by its ID.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_job(job_id: str) -> dict[str, Any]:
    """Retrieve details of a specific job."""
    result: dict[str, Any] = await _call("GET", f"/api/v1/jobs/{job_id}")
    return result


@mcp.tool(
    description="Cancel a running or queued job. This cannot be undone.",
    annotations=ToolAnnotations(destructiveHint=True),
)
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a job."""
    result: dict[str, Any] = await _call("DELETE", f"/api/v1/jobs/{job_id}")
    return result


@mcp.tool(
    description=(
        "Resume a failed job, optionally from a specific checkpoint. "
        "Only applicable to jobs in 'failed' status."
    ),
    annotations=ToolAnnotations(destructiveHint=False),
)
async def resume_job(
    job_id: str,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Resume a failed job."""
    payload: dict[str, Any] | None = None
    if checkpoint_id:
        payload = {"checkpoint_id": checkpoint_id}
    result: dict[str, Any] = await _call("POST", f"/api/v1/jobs/{job_id}/resume", json_body=payload)
    return result


@mcp.tool(
    description=(
        "Get log lines for a job as a JSON list. Returns the most recent log and progress events."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_job_logs(job_id: str) -> list[dict[str, Any]]:
    """Get job log lines (non-streaming)."""
    events: list[dict[str, Any]] = await _call(
        "GET",
        f"/api/v1/jobs/{job_id}/events",
        params={"types": "log,progress"},
    )
    logs: list[dict[str, Any]] = []
    for event in events:
        data = event.get("data", {})
        logs.append(
            {
                "timestamp": event.get("timestamp", ""),
                "type": event.get("type", "log"),
                "message": data.get("message", ""),
            }
        )
    return logs


def _summarise_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute latest values and trend summary from raw metric points."""
    if not metrics:
        return {"latest": {}, "trend": "no metrics recorded yet", "total_steps": 0}

    latest = metrics[-1]
    total_steps = len(metrics)

    losses = [m["loss"] for m in metrics if m.get("loss") is not None]
    trend_parts: list[str] = []
    if len(losses) >= 2:
        first, last = losses[0], losses[-1]
        change_pct = ((last - first) / abs(first)) * 100 if first != 0 else 0.0
        if change_pct < -1:
            direction = "decreasing"
        elif change_pct > 1:
            direction = "increasing"
        else:
            direction = "flat"
        trend_parts.append(f"loss {direction} ({first:.4f} -> {last:.4f}, {change_pct:+.1f}%)")
    elif len(losses) == 1:
        trend_parts.append(f"loss = {losses[0]:.4f} (single step)")

    if latest.get("epoch") is not None:
        trend_parts.append(f"epoch {latest['epoch']:.1f}")

    return {
        "latest": {k: v for k, v in latest.items() if v is not None},
        "trend": "; ".join(trend_parts) if trend_parts else "insufficient data",
        "total_steps": total_steps,
    }


@mcp.tool(
    description=(
        "Get training metrics for a job: latest values and a trend summary. "
        "Only available for training jobs."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_job_metrics(job_id: str) -> dict[str, Any]:
    """Get training metrics with trend summary."""
    raw: list[dict[str, Any]] = await _call("GET", f"/api/v1/jobs/{job_id}/metrics")
    return _summarise_metrics(raw)


@mcp.tool(
    description="Get structured evaluation results for a completed eval job.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_job_results(job_id: str) -> dict[str, Any]:
    """Get eval results for a job."""
    result: dict[str, Any] = await _call("GET", f"/api/v1/jobs/{job_id}/results")
    return result


# ---------------------------------------------------------------------------
# Tools -- artifacts
# ---------------------------------------------------------------------------


@mcp.tool(
    description="List artifacts, optionally filtered by type or producer job ID.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_artifacts(
    type: str | None = None,
    producer_job: str | None = None,
) -> list[dict[str, Any]]:
    """List all artifacts."""
    params: dict[str, str] = {}
    if type:
        params["type"] = type
    if producer_job:
        params["producer_job"] = producer_job
    result: list[dict[str, Any]] = await _call("GET", "/api/v1/artifacts", params=params or None)
    return result


@mcp.tool(
    description="Get details for a specific artifact by its ID.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_artifact(artifact_id: str) -> dict[str, Any]:
    """Get artifact details."""
    result: dict[str, Any] = await _call("GET", f"/api/v1/artifacts/{artifact_id}")
    return result


@mcp.tool(
    description=(
        "Preview the first few lines of a job artifact (e.g. JSONL dataset output). "
        "Returns content preview with metadata about size and format."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def preview_artifact(
    job_id: str,
    artifact_id: str,
    lines: int = 5,
) -> dict[str, Any]:
    """Preview artifact content."""
    result: dict[str, Any] = await _call(
        "GET",
        f"/api/v1/jobs/{job_id}/artifacts/{artifact_id}/preview",
        params={"lines": str(lines)},
    )
    return result


@mcp.tool(
    description=(
        "Upload a local file as an artifact. "
        "Provide the absolute path to a file on the server's filesystem."
    ),
    annotations=ToolAnnotations(destructiveHint=False),
)
async def upload_artifact(
    file_path: str,
    artifact_type: str = "dataset",
    name: str | None = None,
) -> dict[str, Any]:
    """Upload a file as an artifact."""
    data: dict[str, str] = {"artifact_type": artifact_type}
    if name:
        data["name"] = name
    result: dict[str, Any] = await _call_upload("/api/v1/artifacts/upload", file_path, data=data)
    return result


# ---------------------------------------------------------------------------
# Tools -- recipes & discovery
# ---------------------------------------------------------------------------


@mcp.tool(
    description="List available recipe templates with name, description, and type.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_recipes() -> list[dict[str, Any]]:
    """List recipe templates."""
    result: list[dict[str, Any]] = await _call("GET", "/api/v1/recipes")
    return result


@mcp.tool(
    description="Get the full definition for a specific recipe by name.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_recipe(name: str) -> dict[str, Any]:
    """Get a recipe definition."""
    result: dict[str, Any] = await _call("GET", f"/api/v1/recipes/{name}")
    return result


@mcp.tool(
    description=(
        "Submit a job from a recipe template. Provide the recipe name and optional "
        "overrides for config values. Set dry_run=true to validate without submitting."
    ),
    annotations=ToolAnnotations(
        destructiveHint=False,
        idempotentHint=False,
    ),
)
async def submit_recipe_job(
    recipe: str,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a job from a recipe."""
    payload: dict[str, Any] = {
        "recipe": recipe,
        "overrides": overrides or {},
        "dry_run": dry_run,
    }
    result: dict[str, Any] = await _call("POST", "/api/v1/jobs/recipe", json_body=payload)
    return result


@mcp.tool(
    description=(
        "List SDG (synthetic data generation) capabilities: available strategies, "
        "attribute types, data sources, judge templates, and environment types."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_sdg_capabilities() -> dict[str, Any]:
    """Get asynth capabilities."""
    result: dict[str, Any] = await _call("GET", "/api/v1/flows/capabilities")
    return result


@mcp.tool(
    description=(
        "Validate a job configuration without creating it. Returns validation "
        "result with errors and warnings."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def validate_config(
    type: str,
    config: dict[str, Any],
    backend: str = "local",
    gpus: int = 0,
) -> dict[str, Any]:
    """Validate a job config (dry run)."""
    payload: dict[str, Any] = {
        "type": type,
        "config": config,
        "compute": {"backend": backend, "gpus": gpus},
        "dry_run": True,
    }
    result: dict[str, Any] = await _call("POST", "/api/v1/jobs/validate", json_body=payload)
    return result


@mcp.tool(
    description=(
        "Get the JSON schema for a job type's configuration. "
        "Type values: training, sdg, eval, serve."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def get_job_type_schema(type: str) -> dict[str, Any]:
    """Get config schema for a job type."""
    result: dict[str, Any] = await _call("GET", f"/api/v1/job-types/{type}/schema")
    return result


# ---------------------------------------------------------------------------
# Tools -- judge & evaluators
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Judge data quality using a built-in template. Provide a template name "
        "(e.g. 'generic/safety', 'code/quality') and a list of data items to judge."
    ),
    annotations=ToolAnnotations(readOnlyHint=False),
)
async def judge_data(
    template: str,
    data: list[dict[str, Any]],
    model: str = "openai/gpt-4o-mini",
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Judge data with a template."""
    payload: dict[str, Any] = {
        "template": template,
        "data": data,
        "model": model,
    }
    if api_base:
        payload["api_base"] = api_base
    if api_key:
        payload["api_key"] = api_key
    result: dict[str, Any] = await _call("POST", "/api/v1/judge", json_body=payload)
    return result


@mcp.tool(
    description="List available judge templates (e.g. generic/safety, code/quality).",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_judge_templates() -> list[dict[str, Any]]:
    """List judge templates."""
    result: list[dict[str, Any]] = await _call("GET", "/api/v1/judge/templates")
    return result


@mcp.tool(
    description="List all evaluators (custom evaluation configurations).",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_evaluators() -> list[dict[str, Any]]:
    """List evaluators."""
    result: list[dict[str, Any]] = await _call("GET", "/api/v1/evaluators")
    return result


@mcp.tool(
    description=(
        "Run an evaluation using an existing evaluator. Provide the evaluator ID "
        "and dataset to evaluate. Optionally override model and inference params."
    ),
    annotations=ToolAnnotations(readOnlyHint=False),
)
async def run_evaluation(
    evaluator_id: str,
    dataset: list[dict[str, Any]],
    model_override: str | None = None,
    inference_params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an evaluation."""
    payload: dict[str, Any] = {
        "evaluator_id": evaluator_id,
        "dataset": dataset,
    }
    if model_override:
        payload["model_override"] = model_override
    if inference_params_override:
        payload["inference_params_override"] = inference_params_override
    result: dict[str, Any] = await _call("POST", "/api/v1/evaluations", json_body=payload)
    return result


# ---------------------------------------------------------------------------
# Tools -- admin & infrastructure
# ---------------------------------------------------------------------------


@mcp.tool(
    description="List all registered compute backends with their capabilities.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_compute_backends() -> list[dict[str, Any]]:
    """List compute backends."""
    result: list[dict[str, Any]] = await _call("GET", "/api/v1/compute")
    return result


@mcp.tool(
    description="Register a new SSH compute backend for running jobs on a remote machine.",
    annotations=ToolAnnotations(destructiveHint=False),
)
async def register_backend(
    name: str,
    host: str,
    type: str = "ssh",
    user: str | None = None,
    key_path: str | None = None,
    remote_base_dir: str = "~/amortized-jobs",
    container_runtime: str = "podman",
) -> dict[str, Any]:
    """Register a compute backend."""
    payload: dict[str, Any] = {
        "name": name,
        "type": type,
        "host": host,
        "remote_base_dir": remote_base_dir,
        "container_runtime": container_runtime,
    }
    if user:
        payload["user"] = user
    if key_path:
        payload["key_path"] = key_path
    result: dict[str, Any] = await _call("POST", "/api/v1/settings/backends", json_body=payload)
    return result


@mcp.tool(
    description="Test connectivity and GPU detection for a registered compute backend.",
    annotations=ToolAnnotations(readOnlyHint=False),
)
async def test_backend(name: str) -> dict[str, Any]:
    """Test a backend connection."""
    result: dict[str, Any] = await _call("POST", f"/api/v1/settings/backends/{name}/test")
    return result


@mcp.tool(
    description="Remove a registered compute backend. This cannot be undone.",
    annotations=ToolAnnotations(destructiveHint=True),
)
async def remove_backend(name: str) -> dict[str, Any]:
    """Remove a compute backend."""
    result: dict[str, Any] = await _call("DELETE", f"/api/v1/settings/backends/{name}")
    return result


@mcp.tool(
    description="Store an LLM provider API key for use by jobs.",
    annotations=ToolAnnotations(destructiveHint=False),
)
async def add_api_key(
    name: str,
    provider: str,
    key: str,
) -> dict[str, Any]:
    """Add an API key."""
    result: dict[str, Any] = await _call(
        "POST",
        "/api/v1/settings/api-keys",
        json_body={"name": name, "provider": provider, "key": key},
    )
    return result


@mcp.tool(
    description="List stored API keys (keys are redacted, showing only last 4 characters).",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def list_api_keys() -> list[dict[str, Any]]:
    """List stored API keys."""
    result: list[dict[str, Any]] = await _call("GET", "/api/v1/settings/api-keys")
    return result


@mcp.tool(
    description="Delete a stored API key. This cannot be undone.",
    annotations=ToolAnnotations(destructiveHint=True),
)
async def delete_api_key(key_id: str) -> dict[str, Any]:
    """Delete an API key."""
    result: dict[str, Any] = await _call("DELETE", f"/api/v1/settings/api-keys/{key_id}")
    return result


@mcp.tool(
    description=(
        "Estimate GPU VRAM required for LoRA fine-tuning a model. Returns estimated VRAM in GB."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def estimate_vram(
    model_name_or_path: str,
    lora_r: int = 16,
    batch_size: int = 2,
    max_length: int = 2048,
    load_in_4bit: bool = False,
) -> dict[str, Any]:
    """Estimate VRAM requirements."""
    result: dict[str, Any] = await _call(
        "POST",
        "/api/v1/estimate",
        json_body={
            "model_name_or_path": model_name_or_path,
            "lora_r": lora_r,
            "batch_size": batch_size,
            "max_length": max_length,
            "load_in_4bit": load_in_4bit,
        },
    )
    return result


@mcp.tool(
    description="Check runtime health and GPU availability.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def health_check() -> dict[str, Any]:
    """Health check."""
    result: dict[str, Any] = await _call("GET", "/api/v1/health")
    return result


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------


@mcp.resource("amortized://capabilities")
async def capabilities_resource() -> str:
    """Platform overview: job types, algorithms, judge templates, backends."""
    capabilities = await _call("GET", "/api/v1/flows/capabilities")
    backends = await _call("GET", "/api/v1/compute")
    templates = await _call("GET", "/api/v1/judge/templates")

    overview = {
        "job_types": ["training", "sdg", "eval", "serve"],
        "algorithms": ["sft", "dpo", "grpo", "kto"],
        "judge_templates": [t.get("name", t) for t in templates] if templates else [],
        "backends": [b.get("name", b) for b in backends] if backends else [],
        "sdg_capabilities": capabilities,
    }
    return json.dumps(overview, indent=2)


@mcp.resource("amortized://recipes")
async def recipes_resource() -> str:
    """Recipe catalog: name, description, and type for all recipes."""
    recipes = await _call("GET", "/api/v1/recipes")
    return json.dumps(recipes, indent=2)


@mcp.resource("amortized://recipes/{name}")
async def recipe_detail_resource(name: str) -> str:
    """Full configuration for a specific recipe."""
    recipe = await _call("GET", f"/api/v1/recipes/{name}")
    return json.dumps(recipe, indent=2)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def create_mcp_server(app: FastAPI) -> FastMCP:
    """Initialise the fastmcp server and mount it on the FastAPI app at ``/mcp``."""
    global _fastapi_app
    _fastapi_app = app
    app.mount("/mcp", mcp.http_app())
    return mcp
