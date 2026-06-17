"""Tool execution — calls core/ functions directly via Repository.

Each tool maps to a core domain function. The agent calls these tools
internally — the user never runs commands themselves.

NOTE: MCP tools for external AI agents are now auto-generated from the
FastAPI OpenAPI spec via fastapi-mcp (see amortized/mcp/server.py).
These hand-written definitions remain for the internal chat agent's
function-calling interface.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from amortized.agent.schemas import TOOL_REGISTRY, TOOLS

logger = logging.getLogger("amortized.agent.tools")


# Re-export for backward compatibility
__all__ = ["TOOLS", "TOOL_REGISTRY", "execute_tool", "tool_result_summary"]


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    repo: Any,
) -> dict[str, Any]:
    """Execute a tool by calling core/ functions directly.

    Sentinel tools (``propose_action``, ``present_options``) return
    special dicts that the caller inspects to build SSE events.
    """
    if name == "propose_action":
        return {
            "__proposed_action__": True,
            "action_type": arguments.get("action_type", ""),
            "config": arguments.get("config", {}),
            "label": arguments.get("label", "Confirm"),
        }

    if name == "present_options":
        return {
            "__present_options__": True,
            "prompt": arguments.get("prompt", ""),
            "options": arguments.get("options", []),
        }

    try:
        return await _dispatch(repo, name, arguments)
    except Exception as exc:
        logger.exception("Tool %s error", name)
        return {"error": str(exc)}


async def _dispatch(
    repo: Any,
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool call to the correct core function."""
    if name == "list_sdg_flows":
        from amortized.api.flows import _discover_pipelines

        pipelines = _discover_pipelines()
        return {"flows": [p.model_dump() for p in pipelines]}

    if name == "submit_sdg_job":
        from amortized.core.jobs import create_job
        from amortized.models import JobType

        config = {k: v for k, v in args.items() if v is not None}
        row = await create_job(repo, job_type=JobType.sdg, config=config)
        return row

    if name == "submit_training_job":
        from amortized.core.jobs import create_job
        from amortized.models import JobType

        config = {k: v for k, v in args.items() if v is not None}
        output_dir = config.pop("output_dir", None)
        row = await create_job(
            repo,
            job_type=JobType.training,
            config=config,
            output_dir=output_dir,
        )
        return row

    if name == "check_job_status":
        from amortized.core.jobs import get_job

        job_id = args["job_id"]
        job_row = await get_job(repo, job_id)
        if job_row is None:
            return {"error": f"Job {job_id} not found"}
        return job_row

    if name == "get_job_metrics":
        from amortized.core.jobs import get_job

        job_id = args["job_id"]
        job_row = await get_job(repo, job_id)
        if job_row is None:
            return {"error": f"Job {job_id} not found"}
        output_dir = job_row.get("output_dir")
        if not output_dir:
            return {"metrics": []}
        metrics_path = Path(output_dir) / "training_metrics.jsonl"
        if not metrics_path.exists():
            return {"metrics": []}
        metrics: list[dict[str, Any]] = []
        for line in metrics_path.read_text().strip().splitlines():
            if line.strip():
                try:
                    metrics.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
        return {"metrics": metrics}

    if name == "list_jobs":
        from amortized.core.jobs import list_jobs
        from amortized.models import JobStatus, JobType

        status = JobStatus(args["status"]) if "status" in args else None
        job_type = JobType(args["type"]) if "type" in args else None
        rows = await list_jobs(repo, status=status, job_type=job_type)
        return {"jobs": rows}

    if name == "estimate_vram":
        from amortized.api.estimate import _estimate_vram
        from amortized.models import MemoryEstimateRequest

        req = MemoryEstimateRequest(
            model_name_or_path=args["model_name_or_path"],
            lora_r=args.get("lora_r", 16),
            batch_size=args.get("batch_size", 2),
            max_length=args.get("max_length", 2048),
            load_in_4bit=args.get("load_in_4bit", False),
        )
        vram = _estimate_vram(req)
        return {
            "model_name_or_path": req.model_name_or_path,
            "lora_r": req.lora_r,
            "batch_size": req.batch_size,
            "max_length": req.max_length,
            "estimated_vram_gb": vram,
            "load_in_4bit": req.load_in_4bit,
        }

    if name == "create_dataset":
        import amortized.config as _config_mod

        filename = Path(args["filename"]).name
        if not filename.endswith(".jsonl"):
            filename += ".jsonl"
        configured = _config_mod.settings.datasets_dir
        base = _config_mod.settings.data_dir / "datasets"
        datasets_dir = configured if configured is not None else base
        datasets_dir.mkdir(parents=True, exist_ok=True)
        file_path = datasets_dir / filename
        rows_data: list[dict[str, Any]] = args.get("rows", [])
        if not rows_data:
            return {"error": "rows must not be empty"}
        with open(file_path, "w") as f:
            for row in rows_data:
                f.write(json.dumps(row) + "\n")
        columns = list(rows_data[0].keys()) if rows_data else []
        return {
            "path": str(file_path),
            "rows_written": len(rows_data),
            "columns": columns,
        }

    if name == "preview_dataset":
        path = Path(args["path"])
        if not path.exists():
            return {"error": f"Dataset not found: {args['path']}"}
        max_rows = min(args.get("rows", 3), 10)
        parsed_rows: list[dict[str, Any]] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                line = line.strip()
                if line:
                    try:
                        parsed_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        columns = list(parsed_rows[0].keys()) if parsed_rows else []
        return {
            "path": str(path),
            "rows": parsed_rows,
            "columns": columns,
            "total_rows_previewed": len(parsed_rows),
        }

    if name == "convert_dataset":
        import amortized.config as _config_mod
        from amortized.api.datasets import _detect_format, _row_to_messages

        source = Path(args["source_path"])
        if not source.exists():
            return {"error": f"Source dataset not found: {args['source_path']}"}
        all_rows: list[dict[str, Any]] = []
        with open(source) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))
        if not all_rows:
            return {"error": "Source dataset is empty"}
        first_row = all_rows[0]
        if "messages" in first_row:
            converted = all_rows
        else:
            fmt = _detect_format(first_row, None)
            if fmt is None:
                return {"error": f"Cannot detect input format. Columns: {list(first_row.keys())}"}
            user_col, asst_col = fmt
            converted = [_row_to_messages(r, user_col, asst_col) for r in all_rows]
        output_filename = Path(args["output_filename"]).name
        if not output_filename.endswith(".jsonl"):
            output_filename += ".jsonl"
        configured = _config_mod.settings.datasets_dir
        base = _config_mod.settings.data_dir / "datasets"
        datasets_dir = configured if configured is not None else base
        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_path = datasets_dir / output_filename
        with open(output_path, "w") as f:
            for r in converted:
                f.write(json.dumps(r) + "\n")
        return {
            "path": str(output_path),
            "rows_converted": len(converted),
            "sample_row": converted[0],
        }

    if name == "judge_data":
        from amortized.core.artifacts import list_artifacts
        from amortized.core.judge_templates import load_judge_template

        job_id = args["job_id"]
        sample_size = args.get("sample_size", 10)
        artifacts = await list_artifacts(repo, job_id)
        if not artifacts:
            return {"error": "No artifacts found for this job"}
        artifact_id = artifacts[0]["id"]
        judge_artifact = artifacts[0]
        file_path = Path(judge_artifact["path"])
        if not file_path.exists():
            return {"error": "Artifact file not found on disk"}
        data_rows: list[dict[str, Any]] = []
        with open(file_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                line = line.rstrip("\n")
                if line:
                    try:
                        data_rows.append(json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        continue
        if not data_rows:
            return {"error": "No parseable data rows in artifact"}
        try:
            from asynth import JudgeConfig, LiteLLMInferenceConfig, create_judge
        except ImportError:
            return {"error": "asynth is not installed — judge functionality unavailable"}
        template_data = load_judge_template(args["template"])
        judge_config = JudgeConfig(**template_data)
        inference_config = LiteLLMInferenceConfig(model=args["model"])
        j = create_judge(judge_config, inference_config=inference_config)
        judge_results: Any = j.judge(data_rows)
        serialized: list[dict[str, Any]] = []
        for r in judge_results:
            if hasattr(r, "model_dump"):
                serialized.append(r.model_dump())
            elif isinstance(r, dict):
                serialized.append(r)
            else:
                serialized.append({"raw": str(r)})
        passed = sum(1 for r in serialized if r.get("passed", False))
        return {
            "results": serialized,
            "summary": {
                "total": len(serialized),
                "passed": passed,
                "failed": len(serialized) - passed,
                "pass_rate": passed / len(serialized) if serialized else 0,
            },
        }

    if name == "list_judge_templates":
        from amortized.core.judge_templates import list_judge_templates

        templates = list_judge_templates()
        return {"templates": [{"name": t} for t in templates]}

    if name == "list_api_keys":
        keys = await repo.list_api_keys()
        return {"keys": keys}

    if name == "add_api_key":
        key_row: dict[str, Any] = await repo.create_api_key(
            key_id=str(uuid.uuid4()),
            name=args["provider"],
            provider=args["provider"],
            key_value=args["key"],
            created_at=datetime.now(UTC).isoformat(),
        )
        return key_row

    if name == "read_artifact_preview":
        from amortized.core.artifacts import get_artifact, list_artifacts

        job_id = args["job_id"]
        artifact_id = args.get("artifact_id")
        max_lines = min(args.get("lines", 5), 50)
        max_lines = max(1, max_lines)

        if not artifact_id:
            artifacts = await list_artifacts(repo, job_id)
            if not artifacts:
                return {"error": "No artifacts found for this job"}
            artifact_id = artifacts[0]["id"]

        artifact: dict[str, Any] | None = await get_artifact(repo, artifact_id)
        if artifact is None:
            return {"error": f"Artifact {artifact_id} not found"}

        file_path = Path(artifact["path"])
        if not file_path.exists():
            return {"error": "Artifact file not found on disk"}

        binary_exts = {".safetensors", ".bin", ".model", ".pt", ".gguf"}
        if file_path.suffix.lower() in binary_exts:
            return {
                "type": "binary",
                "format": file_path.suffix.lstrip("."),
                "size": file_path.stat().st_size,
                "filename": file_path.name,
            }

        preview_lines: list[str] = []
        with open(file_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                preview_lines.append(line.rstrip("\n"))

        return {
            "type": "text",
            "format": file_path.suffix.lstrip("."),
            "filename": file_path.name,
            "lines": preview_lines,
            "total_size": file_path.stat().st_size,
        }

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

    if name == "create_dataset":
        path = result.get("path", "unknown")
        rows_written = result.get("rows_written", 0)
        return f"Created dataset: {path} ({rows_written} rows)"

    if name == "convert_dataset":
        path = result.get("path", "unknown")
        count = result.get("rows_converted", 0)
        return f"Converted dataset: {path} ({count} rows)"

    if name == "preview_dataset":
        path = result.get("path", "unknown")
        count = result.get("total_rows_previewed", 0)
        return f"Preview: {path} ({count} rows)"

    if name == "judge_data":
        scores = result.get("results", [])
        return f"Judged {len(scores)} row(s)"

    if name == "list_judge_templates":
        templates = result.get("templates", [])
        return f"Found {len(templates)} judge template(s)"

    if name == "list_api_keys":
        keys = result.get("keys", [])
        if not keys:
            return "No API keys configured"
        providers = [k.get("provider", "?") for k in keys]
        return f"Keys configured for: {', '.join(providers)}"

    if name == "add_api_key":
        provider = result.get("provider", "unknown")
        return f"API key stored for provider '{provider}'"

    if name == "read_artifact_preview":
        fmt = result.get("format", "unknown")
        filename = result.get("filename", "unknown")
        if result.get("type") == "binary":
            size = result.get("size", 0)
            return f"Binary file: {filename} ({fmt}, {size} bytes)"
        line_count = len(result.get("lines", []))
        return f"Preview: {filename} ({line_count} lines)"

    return json.dumps(result)[:120]
