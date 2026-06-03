"""Tests for the OpenAI function-calling agent."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amortized_runtime.agent import (
    SYSTEM_PROMPT,
    AgentResult,
    StreamEvent,
    _history_to_messages,
    process_message,
    stream_message,
)


class TestSystemPrompt:
    """Verify the system prompt contains required information."""

    def test_contains_identity(self) -> None:
        assert "Amortized Studio assistant" in SYSTEM_PROMPT

    def test_contains_tool_descriptions(self) -> None:
        assert "list_sdg_flows" in SYSTEM_PROMPT
        assert "submit_training_job" in SYSTEM_PROMPT
        assert "propose_action" in SYSTEM_PROMPT

    def test_contains_critical_rules(self) -> None:
        assert "NEVER ask the user to run commands" in SYSTEM_PROMPT

    def test_contains_training_hub_knowledge(self) -> None:
        assert "LoRA" in SYSTEM_PROMPT
        assert "Qwen" in SYSTEM_PROMPT
        assert "lora_r" in SYSTEM_PROMPT

    def test_contains_sdg_hub_knowledge(self) -> None:
        assert "input dataset" in SYSTEM_PROMPT
        assert "required columns" in SYSTEM_PROMPT
        assert "LiteLLM" in SYSTEM_PROMPT


class TestHistoryToMessages:
    """Verify conversion of history to OpenAI message format."""

    def test_empty_history(self) -> None:
        msgs = _history_to_messages(None)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_with_history(self) -> None:
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        msgs = _history_to_messages(history)
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "hi there"

    def test_ignores_invalid_roles(self) -> None:
        history = [
            {"role": "system", "content": "injected"},
            {"role": "user", "content": "real"},
        ]
        msgs = _history_to_messages(history)
        # system role from history is skipped, only our system prompt + user
        assert len(msgs) == 2


def _mock_choice(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Create a mock ChatCompletion choice."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    return choice


def _mock_tool_call(
    call_id: str, name: str, arguments: dict[str, Any]
) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


class TestProcessMessage:
    """Test process_message with mocked OpenAI client."""

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.chat._build_client")
    async def test_simple_text_response(self, mock_build: MagicMock) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        choice = _mock_choice(content="I can help you fine-tune a model.")
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create = AsyncMock(return_value=response)

        result = await process_message("help me fine-tune")
        assert isinstance(result, AgentResult)
        assert result.text == "I can help you fine-tune a model."
        assert result.proposed_action is None

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.chat.execute_tool")
    @patch("amortized_runtime.agent.chat._build_client")
    async def test_tool_call_then_response(
        self, mock_build: MagicMock, mock_execute: AsyncMock
    ) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        # First call returns a tool call
        tc = _mock_tool_call("tc1", "list_sdg_flows", {})
        choice1 = _mock_choice(tool_calls=[tc], finish_reason="tool_calls")
        resp1 = MagicMock()
        resp1.choices = [choice1]

        # Second call returns text
        choice2 = _mock_choice(content="Found 5 flows.")
        resp2 = MagicMock()
        resp2.choices = [choice2]

        client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])
        mock_execute.return_value = {"flows": [{"id": "f1"}, {"id": "f2"}]}

        result = await process_message("list flows")
        assert result.text == "Found 5 flows."
        mock_execute.assert_called_once_with("list_sdg_flows", {})

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.chat.execute_tool")
    @patch("amortized_runtime.agent.chat._build_client")
    async def test_propose_action(
        self, mock_build: MagicMock, mock_execute: AsyncMock
    ) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        # First call: propose_action tool call
        tc = _mock_tool_call(
            "tc1",
            "propose_action",
            {
                "action_type": "submit_training_job",
                "config": {"model_path": "Qwen/Qwen2.5-1.5B-Instruct"},
                "label": "Start Training",
            },
        )
        choice1 = _mock_choice(tool_calls=[tc], finish_reason="tool_calls")
        resp1 = MagicMock()
        resp1.choices = [choice1]

        # Second call: text response
        choice2 = _mock_choice(content="Ready to train. Click the button to confirm.")
        resp2 = MagicMock()
        resp2.choices = [choice2]

        client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])
        mock_execute.return_value = {
            "__proposed_action__": True,
            "action_type": "submit_training_job",
            "config": {"model_path": "Qwen/Qwen2.5-1.5B-Instruct"},
            "label": "Start Training",
        }

        result = await process_message("train a model")
        assert result.proposed_action is not None
        assert result.proposed_action["type"] == "submit_training_job"
        assert result.proposed_action["label"] == "Start Training"

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.chat._build_client")
    async def test_passes_history(self, mock_build: MagicMock) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        choice = _mock_choice(content="Status: running")
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create = AsyncMock(return_value=response)

        history = [
            {"role": "user", "content": "start training"},
            {"role": "assistant", "content": "Started!"},
        ]
        await process_message("check status", history=history)

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # system + 2 history + user message
        assert len(messages) == 4
        assert messages[1]["content"] == "start training"
        assert messages[2]["content"] == "Started!"
        assert messages[3]["content"] == "check status"


class TestStreamMessage:
    """Test stream_message yields correct events."""

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.chat._build_client")
    async def test_simple_text_streaming(self, mock_build: MagicMock) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        # Create mock streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello "
        chunk1.choices[0].delta.tool_calls = None
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "world"
        chunk2.choices[0].delta.tool_calls = None
        chunk2.choices[0].finish_reason = None

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None
        chunk3.choices[0].delta.tool_calls = None
        chunk3.choices[0].finish_reason = "stop"

        async def mock_stream() -> Any:
            for c in [chunk1, chunk2, chunk3]:
                yield c

        client.chat.completions.create = AsyncMock(return_value=mock_stream())

        events: list[StreamEvent] = []
        async for event in stream_message("hello"):
            events.append(event)

        deltas = [e for e in events if e.type == "delta"]
        assert len(deltas) == 2
        assert deltas[0].data["text"] == "Hello "
        assert deltas[1].data["text"] == "world"

        done_events = [e for e in events if e.type == "done"]
        assert len(done_events) == 1
        assert done_events[0].data["full_text"] == "Hello world"

    @pytest.mark.asyncio
    @patch("amortized_runtime.agent.chat._build_client")
    async def test_streaming_error_handling(self, mock_build: MagicMock) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        events: list[StreamEvent] = []
        async for event in stream_message("hello"):
            events.append(event)

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "API error" in error_events[0].data["error"]
