"""MCP server using fastmcp with httpx ASGI transport.

Replaces the fastapi-mcp auto-generated server with hand-crafted tools
that have proper descriptions, annotations, and structured error handling.
"""

from __future__ import annotations

import logging
import os
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
        "track progress, view logs and metrics, and manage job lifecycle."
    ),
)

_fastapi_app: FastAPI | None = None


async def _call(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: Any = None,
) -> Any:
    """Make an in-process HTTP call to the FastAPI app via ASGI transport.

    Translates HTTP 4xx/5xx into ``ValueError`` with a structured message
    so that fastmcp surfaces the error to the MCP client cleanly.
    """
    if _fastapi_app is None:
        raise RuntimeError("MCP server not initialised; call create_mcp_server first")

    transport = ASGITransport(app=_fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mcp") as client:
        response = await client.request(method, path, params=params, json=json)

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


# ---------------------------------------------------------------------------
# Tools — job management
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
    result: dict[str, Any] = await _call("POST", "/api/v1/jobs", json=payload)
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
    result: dict[str, Any] = await _call("POST", f"/api/v1/jobs/{job_id}/resume", json=payload)
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
# Server lifecycle
# ---------------------------------------------------------------------------


def create_mcp_server(app: FastAPI) -> FastMCP:
    """Initialise the fastmcp server and mount it on the FastAPI app at ``/mcp``."""
    global _fastapi_app
    _fastapi_app = app
    app.mount("/mcp", mcp.http_app())
    return mcp


def create_mcp_server_legacy(app: FastAPI) -> Any:
    """Fall back to the fastapi-mcp auto-generated server."""
    from amortized.mcp.server_legacy import create_mcp_server as _legacy

    return _legacy(app)


def create_mcp_server_auto(app: FastAPI) -> Any:
    """Pick the MCP implementation based on the AMORTIZED_LEGACY_MCP env flag."""
    if os.environ.get("AMORTIZED_LEGACY_MCP") == "1":
        logger.info("Using legacy fastapi-mcp server (AMORTIZED_LEGACY_MCP=1)")
        return create_mcp_server_legacy(app)
    return create_mcp_server(app)
