"""Claude Code CLI agent for guiding users through model customization workflows."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from amortized_runtime.config import settings

logger = logging.getLogger("amortized_runtime.agent")

# Project root where .claude/ skills and CLAUDE.md live
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CONTEXT_PREAMBLE = """\
You are the Amortized Studio assistant — an AI concierge embedded in a web \
dashboard that helps users optimize their AI agent workflows.

## YOUR ROLE
You handle ALL technical work on behalf of the user. The user interacts with \
you through a chat interface in their browser — they do NOT have terminal \
access and cannot run commands.

When the user wants to:
- **Train a model**: YOU submit the training job by calling the API with \
Bash (curl -X POST http://localhost:8000/api/v1/jobs/training ...)
- **Generate data**: YOU submit the SDG job by calling the API
- **Check status**: YOU call the API and report the results in plain language
- **Estimate VRAM**: YOU call the API and explain the results

## CRITICAL RULES
- NEVER ask the user to run commands, scripts, or code
- NEVER reference project files, directories, or scripts
- NEVER show code blocks with shell commands for the user to execute
- Always submit jobs and check status YOURSELF using curl to the runtime API
- Present results in user-friendly language with markdown formatting
- Guide users to the Studio UI pages (Jobs, Flows, Settings) when relevant

## API ENDPOINTS
- POST http://localhost:8000/api/v1/jobs/training — Submit training job
- POST http://localhost:8000/api/v1/jobs/sdg — Submit SDG job
- GET http://localhost:8000/api/v1/jobs — List jobs
- GET http://localhost:8000/api/v1/jobs/{id} — Job details
- GET http://localhost:8000/api/v1/jobs/{id}/metrics — Training metrics
- GET http://localhost:8000/api/v1/flows — Available SDG flows
- POST http://localhost:8000/api/v1/estimate — VRAM estimation
- DELETE http://localhost:8000/api/v1/jobs/{id} — Cancel job

## AVAILABLE MODELS
- Training: Qwen/Qwen2.5-1.5B-Instruct (small, fast), use QLoRA \
(load_in_4bit=true) for 7B+ models
- SDG teacher: Use the model configured in the job (e.g., openai/gpt-5-mini)

## THE AMORTIZATION WORKFLOW
1. Understand the user's task
2. Generate training data using SDG (you submit the job)
3. Fine-tune a small model on that data (you submit the job)
4. Help evaluate results\
"""


def _build_context(history: list[dict[str, str]] | None = None) -> str:
    """Build the append-system-prompt context string from conversation history."""
    parts: list[str] = [CONTEXT_PREAMBLE]

    if history:
        parts.append("\n\n## Conversation History")
        for entry in history:
            role = entry["role"].capitalize()
            parts.append(f"\n{role}: {entry['content']}")

    return "\n".join(parts)


def _build_cmd(
    message: str,
    context: str,
    output_format: str = "json",
    verbose: bool = False,
) -> list[str]:
    """Build the claude CLI command."""
    cmd = [
        settings.claude_command,
        "-p",
        "--output-format",
        output_format,
        "--append-system-prompt",
        context,
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--max-turns",
        str(settings.claude_max_turns),
        "--model",
        settings.claude_model,
        "--allowedTools",
        "Bash",
    ]
    if verbose:
        cmd.append("--verbose")
    if output_format == "stream-json":
        cmd.append("--include-partial-messages")
    cmd.append(message)
    return cmd


async def process_message(
    message: str,
    history: list[dict[str, str]] | None = None,
    project_dir: str | None = None,
) -> str:
    """Send a message to Claude Code CLI and return the response text."""
    context = _build_context(history)
    cmd = _build_cmd(message, context, output_format="json")
    cwd = project_dir or str(PROJECT_ROOT)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
    except FileNotFoundError:
        logger.error("claude CLI not found at %r", settings.claude_command)
        return (
            "The Claude CLI is not installed or not in PATH. "
            "Please install Claude Code and try again."
        )
    except TimeoutError:
        logger.error("claude CLI timed out after 300s")
        proc.kill()
        await proc.wait()
        return "The request timed out. Please try again with a simpler question."

    if proc.returncode != 0:
        logger.error(
            "claude CLI exited with code %d: %s",
            proc.returncode,
            stderr.decode() if stderr else "",
        )
        return "Sorry, something went wrong processing your request. Please try again."

    try:
        data: dict[str, Any] = json.loads(stdout.decode())
        return str(data.get("result", ""))
    except json.JSONDecodeError:
        output = stdout.decode() if stdout else ""
        logger.error("Failed to parse claude CLI JSON output: %s", output[:500])
        return "Sorry, I couldn't parse the response. Please try again."


async def stream_message(
    message: str,
    history: list[dict[str, str]] | None = None,
    project_dir: str | None = None,
) -> asyncio.subprocess.Process:
    """Spawn claude CLI with stream-json output and return the async Process.

    The caller reads proc.stdout line by line using ``async for line in proc.stdout``.
    """
    context = _build_context(history)
    cmd = _build_cmd(message, context, output_format="stream-json", verbose=True)
    cwd = project_dir or str(PROJECT_ROOT)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    return proc
