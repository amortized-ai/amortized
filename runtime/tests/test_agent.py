"""Tests for the Claude Code CLI agent."""

import json
import subprocess
from unittest.mock import MagicMock, patch

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


class TestProcessMessage:
    """Test process_message with mocked subprocess."""

    @patch("amortized_runtime.agent.subprocess.run")
    def test_successful_response(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"result": "I can help you fine-tune a model."}),
            stderr="",
        )
        result = process_message("help me fine-tune")
        assert result == "I can help you fine-tune a model."
        mock_run.assert_called_once()

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "help me fine-tune" in cmd

    @patch("amortized_runtime.agent.subprocess.run")
    def test_claude_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        result = process_message("hello")
        assert "not installed" in result

    @patch("amortized_runtime.agent.subprocess.run")
    def test_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        result = process_message("hello")
        assert "timed out" in result

    @patch("amortized_runtime.agent.subprocess.run")
    def test_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        result = process_message("hello")
        assert "went wrong" in result

    @patch("amortized_runtime.agent.subprocess.run")
    def test_invalid_json(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr=""
        )
        result = process_message("hello")
        assert "couldn't parse" in result

    @patch("amortized_runtime.agent.subprocess.run")
    def test_passes_history_in_context(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"result": "Sure, here's the status."}),
            stderr="",
        )
        history = [
            {"role": "user", "content": "start a training job"},
            {"role": "assistant", "content": "I started the job."},
        ]
        result = process_message("what's the status?", history=history)
        assert result == "Sure, here's the status."

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        # The append-system-prompt should contain history
        prompt_idx = cmd.index("--append-system-prompt")
        context = cmd[prompt_idx + 1]
        assert "start a training job" in context
        assert "I started the job." in context

    @patch("amortized_runtime.agent.subprocess.run")
    def test_custom_project_dir(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"result": "ok"}),
            stderr="",
        )
        process_message("hello", project_dir="/tmp/myproject")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == "/tmp/myproject"


class TestStreamMessage:
    """Test stream_message spawns subprocess correctly."""

    @patch("amortized_runtime.agent.subprocess.Popen")
    def test_returns_popen(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        result = stream_message("hello")
        assert result is mock_proc
        mock_popen.assert_called_once()

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"
        assert "--verbose" in cmd
        assert "hello" in cmd

    @patch("amortized_runtime.agent.subprocess.Popen")
    def test_stream_with_history(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        history = [{"role": "user", "content": "prior message"}]
        stream_message("follow up", history=history)

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        prompt_idx = cmd.index("--append-system-prompt")
        context = cmd[prompt_idx + 1]
        assert "prior message" in context


class TestContextPreamble:
    """Verify the context preamble contains required information."""

    def test_contains_identity(self) -> None:
        assert "Amortized assistant" in CONTEXT_PREAMBLE

    def test_contains_api_reference(self) -> None:
        assert "localhost:8000" in CONTEXT_PREAMBLE

    def test_contains_skills_reference(self) -> None:
        assert ".claude/skills/" in CONTEXT_PREAMBLE
