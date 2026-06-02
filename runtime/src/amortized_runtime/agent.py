"""Claude Code CLI agent for guiding users through model customization workflows."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from amortized_runtime.config import settings

logger = logging.getLogger("amortized_runtime.agent")

# Project root where .claude/ skills and CLAUDE.md live
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CONTEXT_PREAMBLE = """\
You are the Amortized assistant — an AI guide that helps users optimize their \
AI agent workflows by replacing expensive frontier model calls with smaller, \
customized models.

You have access to the Amortized runtime API at http://localhost:8000. \
Use the skills in .claude/skills/ to interact with the API when the user \
asks you to submit jobs, check status, estimate VRAM, list flows, or list jobs.

Be concise and helpful. Use markdown formatting.\
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
    ]
    if verbose:
        cmd.append("--verbose")
    cmd.append(message)
    return cmd


def process_message(
    message: str,
    history: list[dict[str, str]] | None = None,
    project_dir: str | None = None,
) -> str:
    """Send a message to Claude Code CLI and return the response text."""
    context = _build_context(history)
    cmd = _build_cmd(message, context, output_format="json")
    cwd = project_dir or str(PROJECT_ROOT)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300,
        )
    except FileNotFoundError:
        logger.error("claude CLI not found at %r", settings.claude_command)
        return (
            "The Claude CLI is not installed or not in PATH. "
            "Please install Claude Code and try again."
        )
    except subprocess.TimeoutExpired:
        logger.error("claude CLI timed out after 300s")
        return "The request timed out. Please try again with a simpler question."

    if result.returncode != 0:
        logger.error("claude CLI exited with code %d: %s", result.returncode, result.stderr)
        return "Sorry, something went wrong processing your request. Please try again."

    try:
        data: dict[str, Any] = json.loads(result.stdout)
        return str(data.get("result", ""))
    except json.JSONDecodeError:
        logger.error("Failed to parse claude CLI JSON output: %s", result.stdout[:500])
        return "Sorry, I couldn't parse the response. Please try again."


def stream_message(
    message: str,
    history: list[dict[str, str]] | None = None,
    project_dir: str | None = None,
) -> subprocess.Popen[str]:
    """Spawn claude CLI with stream-json output and return the Popen handle.

    The caller reads proc.stdout line by line, parsing JSON events.
    """
    context = _build_context(history)
    cmd = _build_cmd(message, context, output_format="stream-json", verbose=True)
    cwd = project_dir or str(PROJECT_ROOT)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )
    return proc
