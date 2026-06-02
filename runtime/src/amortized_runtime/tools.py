"""OpenAI function-calling tool definitions and server-side execution.

Each tool maps to a runtime API endpoint. The agent calls these tools
internally via httpx — the user never runs commands themselves.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("amortized_runtime.tools")

RUNTIME_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_sdg_flows",
            "description": "List available SDG (synthetic data generation) flows.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_sdg_job",
            "description": (
                "Submit a synthetic data generation job. "
                "Use propose_action instead if you want the user to confirm first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {
                        "type": "string",
                        "description": "SDG flow identifier",
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to input dataset",
                    },
                    "model": {
                        "type": "string",
                        "description": "Teacher model name (e.g. openai/gpt-4o)",
                    },
                    "api_base": {
                        "type": "string",
                        "description": "Teacher model API base URL",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Teacher model API key",
                    },
                },
                "required": ["flow_id", "dataset_path", "model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_training_job",
            "description": (
                "Submit a LoRA SFT training job. "
                "Use propose_action instead if you want the user to confirm first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "HuggingFace model ID or local path",
                    },
                    "data_path": {
                        "type": "string",
                        "description": "Path to training data (JSONL)",
                    },
                    "ckpt_output_dir": {
                        "type": "string",
                        "description": "Output directory for checkpoints",
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Learning rate (default: 2e-4)",
                    },
                    "num_epochs": {
                        "type": "integer",
                        "description": "Number of training epochs (default: 3)",
                    },
                    "lora_r": {
                        "type": "integer",
                        "description": "LoRA rank (default: 16)",
                    },
                    "lora_alpha": {
                        "type": "integer",
                        "description": "LoRA alpha scaling factor (default: 32)",
                    },
                    "load_in_4bit": {
                        "type": "boolean",
                        "description": "Enable QLoRA 4-bit quantization",
                    },
                    "micro_batch_size": {
                        "type": "integer",
                        "description": "Micro batch size (default: 2)",
                    },
                    "max_seq_len": {
                        "type": "integer",
                        "description": "Maximum sequence length (default: 2048)",
                    },
                },
                "required": ["model_path", "data_path", "ckpt_output_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_job_status",
            "description": "Check the status and details of a specific job.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to check",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_metrics",
            "description": "Get training metrics (loss, learning rate, epoch) for a job.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to get metrics for",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_jobs",
            "description": "List all jobs, optionally filtered by status or type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "running", "completed", "failed", "cancelled"],
                        "description": "Filter by job status",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["training", "sdg"],
                        "description": "Filter by job type",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_vram",
            "description": "Estimate GPU VRAM requirements for a training configuration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "HuggingFace model ID",
                    },
                    "lora_r": {
                        "type": "integer",
                        "description": "LoRA rank (default: 16)",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size (default: 2)",
                    },
                    "max_seq_len": {
                        "type": "integer",
                        "description": "Max sequence length (default: 2048)",
                    },
                    "load_in_4bit": {
                        "type": "boolean",
                        "description": "Use QLoRA 4-bit quantization",
                    },
                },
                "required": ["model_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact_preview",
            "description": (
                "Preview the contents of a job artifact (first few lines of "
                "generated data, metrics, etc). Use this to assess data quality, "
                "check training metrics, or show the user a sample of generated data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID",
                    },
                    "artifact_id": {
                        "type": "string",
                        "description": (
                            "Artifact ID (optional — if omitted, previews the main output file)"
                        ),
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to preview (default 5, max 50)",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_action",
            "description": (
                "Propose an action for the user to confirm before executing. "
                "Use this instead of directly calling submit_training_job or "
                "submit_sdg_job so the user can review and approve the configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["submit_training_job", "submit_sdg_job"],
                        "description": "The action to propose",
                    },
                    "config": {
                        "type": "object",
                        "description": "The configuration for the action",
                    },
                    "label": {
                        "type": "string",
                        "description": "Human-readable button label (e.g. 'Start Training')",
                    },
                },
                "required": ["action_type", "config", "label"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by calling the runtime API and return the result.

    The ``propose_action`` tool is special — it returns a sentinel dict
    that the caller inspects to build an SSE ``action`` event.
    """
    if name == "propose_action":
        return {
            "__proposed_action__": True,
            "action_type": arguments.get("action_type", ""),
            "config": arguments.get("config", {}),
            "label": arguments.get("label", "Confirm"),
        }

    try:
        async with httpx.AsyncClient(base_url=RUNTIME_BASE, timeout=60) as client:
            return await _call_api(client, name, arguments)
    except httpx.HTTPError as exc:
        logger.exception("Tool %s HTTP error", name)
        return {"error": f"API request failed: {exc}"}
    except Exception as exc:
        logger.exception("Tool %s unexpected error", name)
        return {"error": str(exc)}


async def _call_api(
    client: httpx.AsyncClient,
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool call to the correct runtime API endpoint."""
    if name == "list_sdg_flows":
        r = await client.get("/api/v1/flows")
        r.raise_for_status()
        data: dict[str, Any] = {"flows": r.json()}
        return data

    if name == "submit_sdg_job":
        r = await client.post("/api/v1/jobs/sdg", json=args)
        r.raise_for_status()
        result: dict[str, Any] = r.json()
        return result

    if name == "submit_training_job":
        r = await client.post("/api/v1/jobs/training", json=args)
        r.raise_for_status()
        result2: dict[str, Any] = r.json()
        return result2

    if name == "check_job_status":
        job_id = args["job_id"]
        r = await client.get(f"/api/v1/jobs/{job_id}")
        r.raise_for_status()
        result3: dict[str, Any] = r.json()
        return result3

    if name == "get_job_metrics":
        job_id = args["job_id"]
        r = await client.get(f"/api/v1/jobs/{job_id}/metrics")
        r.raise_for_status()
        metrics_data: dict[str, Any] = {"metrics": r.json()}
        return metrics_data

    if name == "list_jobs":
        params: dict[str, str] = {}
        if "status" in args:
            params["status"] = args["status"]
        if "type" in args:
            params["type"] = args["type"]
        r = await client.get("/api/v1/jobs", params=params)
        r.raise_for_status()
        jobs_data: dict[str, Any] = {"jobs": r.json()}
        return jobs_data

    if name == "estimate_vram":
        r = await client.post("/api/v1/estimate", json=args)
        r.raise_for_status()
        estimate_data: dict[str, Any] = r.json()
        return estimate_data

    if name == "read_artifact_preview":
        job_id = args["job_id"]
        artifact_id = args.get("artifact_id")
        preview_params: dict[str, Any] = {}
        if "lines" in args:
            preview_params["lines"] = args["lines"]
        if artifact_id:
            r = await client.get(
                f"/api/v1/jobs/{job_id}/artifacts/{artifact_id}/preview",
                params=preview_params,
            )
        else:
            # No artifact_id — list artifacts and preview the first one
            r = await client.get(f"/api/v1/jobs/{job_id}/artifacts")
            r.raise_for_status()
            artifacts = r.json()
            if not artifacts:
                return {"error": "No artifacts found for this job"}
            artifact_id = artifacts[0]["id"]
            r = await client.get(
                f"/api/v1/jobs/{job_id}/artifacts/{artifact_id}/preview",
                params=preview_params,
            )
        r.raise_for_status()
        preview_data: dict[str, Any] = r.json()
        return preview_data

    return {"error": f"Unknown tool: {name}"}


def tool_result_summary(name: str, result: dict[str, Any]) -> str:
    """Return a short human-readable summary of a tool result for SSE events."""
    if "error" in result:
        return f"Error: {result['error']}"

    if name == "list_sdg_flows":
        flows = result.get("flows", [])
        return f"Found {len(flows)} SDG flow(s)"

    if name == "list_jobs":
        jobs = result.get("jobs", [])
        return f"Found {len(jobs)} job(s)"

    if name in ("submit_training_job", "submit_sdg_job"):
        job_id = result.get("id", "unknown")
        return f"Job created: {job_id}"

    if name == "check_job_status":
        status = result.get("status", "unknown")
        return f"Job status: {status}"

    if name == "get_job_metrics":
        metrics = result.get("metrics", [])
        return f"Got {len(metrics)} metric point(s)"

    if name == "estimate_vram":
        vram = result.get("estimated_vram_gb", "?")
        return f"Estimated VRAM: {vram} GB"

    if name == "read_artifact_preview":
        fmt = result.get("format", "unknown")
        filename = result.get("filename", "unknown")
        if result.get("type") == "binary":
            size = result.get("size", 0)
            return f"Binary file: {filename} ({fmt}, {size} bytes)"
        line_count = len(result.get("lines", []))
        return f"Preview: {filename} ({line_count} lines)"

    return json.dumps(result)[:120]
