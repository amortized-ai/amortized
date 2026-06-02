"""Tests for the Claude Code CLI agent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amortized_runtime.agent import (
    CONTEXT_PREAMBLE,
    _build_cmd,
    _build_context,
    process_message,
    stream_message,
)


class TestBuildContext:
    """Verify context building for the append-system-prompt."""

    def test_no_history(self) -> None:
        context = _build_context()
        assert CONTEXT_PREAMBLE in context
        assert "Conversation History" not in context

    def test_with_history(self) -> None:
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]
        context = _build_context(history)
        assert CONTEXT_PREAMBLE in context
        assert "Conversation History" in context
        assert "User: hi" in context
        assert "Assistant: hello there" in context

    def test_empty_history(self) -> None:
        context = _build_context([])
        assert CONTEXT_PREAMBLE in context
        assert "Conversation History" not in context


class TestBuildCmd:
    """Verify CLI command construction."""

    @patch("amortized_runtime.agent.settings")
    def test_json_format(self, mock_settings: MagicMock) -> None:
        mock_settings.claude_command = "claude"
        mock_settings.claude_model = "sonnet"
        mock_settings.claude_max_turns = 1

        cmd = _build_cmd("hello", "context", output_format="json")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "json"
        assert "--dangerously-skip-permissions" in cmd
        assert "--no-session-persistence" in cmd
        assert "--max-turns" in cmd
        assert "--model" in cmd
        assert "--verbose" not in cmd
        assert cmd[-1] == "hello"

    @patch("amortized_runtime.agent.settings")
    def test_stream_json_verbose(self, mock_settings: MagicMock) -> None:
        mock_settings.claude_command = "claude"
        mock_settings.claude_model = "sonnet"
        mock_settings.claude_max_turns = 1

        cmd = _build_cmd("hello", "context", output_format="stream-json", verbose=True)
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"
        assert "--verbose" in cmd
        assert "--include-partial-messages" in cmd

    @patch("amortized_runtime.agent.settings")
    def test_json_format_no_partial_messages(self, mock_settings: MagicMock) -> None:
        mock_settings.claude_command = "claude"
        mock_settings.claude_model = "sonnet"
        mock_settings.claude_max_turns = 1

        cmd = _build_cmd("hello", "context", output_format="json")
        assert "--include-partial-messages" not in cmd

    @patch("amortized_runtime.agent.settings")
    def test_custom_model(self, mock_settings: MagicMock) -> None:
        mock_settings.claude_command = "/usr/bin/claude"
        mock_settings.claude_model = "opus"
        mock_settings.claude_max_turns = 3

        cmd = _build_cmd("test", "ctx")
        assert cmd[0] == "/usr/bin/claude"
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "opus"
        turns_idx = cmd.index("--max-turns")
        assert cmd[turns_idx + 1] == "3"


def _mock_async_process(
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
) -> AsyncMock:
    """Create a mock asyncio subprocess process."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.wait = AsyncMock(return_value=returncode)
    proc.kill = MagicMock()
    return proc


class TestProcessMessage:
    """Test process_message with mocked asyncio subprocess."""

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_successful_response(self, mock_create: AsyncMock) -> None:
        mock_proc = _mock_async_process(
            stdout=json.dumps({"result": "I can help you fine-tune a model."}).encode(),
        )
        mock_create.return_value = mock_proc

        result = await process_message("help me fine-tune")
        assert result == "I can help you fine-tune a model."
        mock_create.assert_called_once()

        call_args = mock_create.call_args[0]
        assert call_args[0] == "claude"
        assert "-p" in call_args
        assert "--output-format" in call_args
        assert "help me fine-tune" in call_args

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_claude_not_found(self, mock_create: AsyncMock) -> None:
        mock_create.side_effect = FileNotFoundError()
        result = await process_message("hello")
        assert "not installed" in result

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_timeout(self, mock_create: AsyncMock) -> None:
        mock_proc = _mock_async_process()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_create.return_value = mock_proc

        result = await process_message("hello")
        assert "timed out" in result
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_nonzero_exit(self, mock_create: AsyncMock) -> None:
        mock_proc = _mock_async_process(returncode=1, stderr=b"error")
        mock_create.return_value = mock_proc

        result = await process_message("hello")
        assert "went wrong" in result

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_invalid_json(self, mock_create: AsyncMock) -> None:
        mock_proc = _mock_async_process(stdout=b"not json")
        mock_create.return_value = mock_proc

        result = await process_message("hello")
        assert "couldn't parse" in result

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_passes_history_in_context(self, mock_create: AsyncMock) -> None:
        mock_proc = _mock_async_process(
            stdout=json.dumps({"result": "Sure, here's the status."}).encode(),
        )
        mock_create.return_value = mock_proc

        history = [
            {"role": "user", "content": "start a training job"},
            {"role": "assistant", "content": "I started the job."},
        ]
        result = await process_message("what's the status?", history=history)
        assert result == "Sure, here's the status."

        call_args = mock_create.call_args[0]
        prompt_idx = list(call_args).index("--append-system-prompt")
        context = call_args[prompt_idx + 1]
        assert "start a training job" in context
        assert "I started the job." in context

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_custom_project_dir(self, mock_create: AsyncMock) -> None:
        mock_proc = _mock_async_process(
            stdout=json.dumps({"result": "ok"}).encode(),
        )
        mock_create.return_value = mock_proc

        await process_message("hello", project_dir="/tmp/myproject")
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["cwd"] == "/tmp/myproject"


class TestStreamMessage:
    """Test stream_message spawns async subprocess correctly."""

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_returns_process(self, mock_create: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_create.return_value = mock_proc

        result = await stream_message("hello")
        assert result is mock_proc
        mock_create.assert_called_once()

        call_args = mock_create.call_args[0]
        assert call_args[0] == "claude"
        assert "-p" in call_args
        assert "--output-format" in call_args
        idx = list(call_args).index("--output-format")
        assert call_args[idx + 1] == "stream-json"
        assert "--verbose" in call_args
        assert "--include-partial-messages" in call_args
        assert "hello" in call_args

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.asyncio.create_subprocess_exec")
    async def test_stream_with_history(self, mock_create: AsyncMock) -> None:
        mock_proc = AsyncMock()
        mock_create.return_value = mock_proc

        history = [{"role": "user", "content": "prior message"}]
        await stream_message("follow up", history=history)

        call_args = mock_create.call_args[0]
        prompt_idx = list(call_args).index("--append-system-prompt")
        context = call_args[prompt_idx + 1]
        assert "prior message" in context


class TestContextPreamble:
    """Verify the context preamble contains required information."""

    def test_contains_identity(self) -> None:
        assert "Amortized assistant" in CONTEXT_PREAMBLE

    def test_contains_api_reference(self) -> None:
        assert "localhost:8000" in CONTEXT_PREAMBLE

    def test_contains_skills_reference(self) -> None:
        assert ".claude/skills/" in CONTEXT_PREAMBLE
