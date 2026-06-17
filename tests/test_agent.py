"""Tests for the OpenAI function-calling agent."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amortized.agent import (
    SYSTEM_PROMPT,
    AgentResult,
    _history_to_messages,
    process_message,
    stream_message,
)
from amortized.agent.protocol import EventType, StreamEvent
from amortized.agent.schemas import (
    TOOL_REGISTRY,
    TOOLS,
    PresentOptionsInput,
    SubmitSdgJobInput,
    ToolDef,
)
from amortized.agent.tools import execute_tool, tool_result_summary


class TestSystemPrompt:
    """Verify the system prompt contains required information."""

    def test_contains_identity(self) -> None:
        assert "Amortized Studio assistant" in SYSTEM_PROMPT

    def test_contains_tool_descriptions(self) -> None:
        assert "list_sdg_flows" in SYSTEM_PROMPT
        assert "submit_training_job" in SYSTEM_PROMPT
        assert "propose_action" in SYSTEM_PROMPT

    def test_contains_present_options(self) -> None:
        assert "present_options" in SYSTEM_PROMPT

    def test_contains_critical_rules(self) -> None:
        assert "NEVER ask the user to run commands" in SYSTEM_PROMPT

    def test_contains_training_hub_knowledge(self) -> None:
        assert "LoRA" in SYSTEM_PROMPT
        assert "Qwen" in SYSTEM_PROMPT
        assert "lora_r" in SYSTEM_PROMPT

    def test_contains_asynth_knowledge(self) -> None:
        assert "asynth" in SYSTEM_PROMPT
        assert "strategy_params" in SYSTEM_PROMPT
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


class TestProtocol:
    """Test SSE protocol types."""

    def test_event_type_values(self) -> None:
        assert EventType.metadata == "metadata"
        assert EventType.thinking == "thinking"
        assert EventType.tool_result == "tool_result"
        assert EventType.delta == "delta"
        assert EventType.action == "action"
        assert EventType.options == "options"
        assert EventType.done == "done"
        assert EventType.error == "error"

    def test_stream_event_model(self) -> None:
        event = StreamEvent(type=EventType.delta, data={"text": "hello"})
        assert event.type == EventType.delta
        assert event.data == {"text": "hello"}

    def test_stream_event_serialization(self) -> None:
        event = StreamEvent(type=EventType.done, data={"full_text": "result"})
        d = event.model_dump()
        assert d["type"] == "done"
        assert d["data"]["full_text"] == "result"


class TestSchemas:
    """Test Pydantic tool schemas."""

    def test_tool_registry_count(self) -> None:
        assert len(TOOL_REGISTRY) == 17

    def test_tools_list_matches_registry(self) -> None:
        assert len(TOOLS) == len(TOOL_REGISTRY)

    def test_tool_def_to_openai_schema(self) -> None:
        td = ToolDef("test_tool", "A test tool", SubmitSdgJobInput)
        schema = td.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert "properties" in schema["function"]["parameters"]
        assert "model" in schema["function"]["parameters"]["properties"]

    def test_tool_def_no_input_model(self) -> None:
        td = ToolDef("empty_tool", "No params")
        schema = td.to_openai_schema()
        assert schema["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_present_options_input(self) -> None:
        inp = PresentOptionsInput(
            prompt="Choose a model",
            options=[{"label": "Qwen 1.5B"}, {"label": "Llama 7B"}],
        )
        assert inp.prompt == "Choose a model"
        assert len(inp.options) == 2

    def test_all_tools_have_names(self) -> None:
        names = {t.name for t in TOOL_REGISTRY}
        assert "list_sdg_flows" in names
        assert "submit_sdg_job" in names
        assert "submit_training_job" in names
        assert "propose_action" in names
        assert "present_options" in names
        assert "list_api_keys" in names

    def test_no_duplicate_tool_names(self) -> None:
        names = [t.name for t in TOOL_REGISTRY]
        assert len(names) == len(set(names))


class TestToolExecution:
    """Test tool execution dispatching."""

    @pytest.mark.asyncio
    async def test_propose_action_sentinel(self) -> None:
        result = await execute_tool(
            "propose_action",
            {
                "action_type": "submit_training_job",
                "config": {"model": "test"},
                "label": "Go",
            },
            repo=None,
        )
        assert result["__proposed_action__"] is True
        assert result["action_type"] == "submit_training_job"

    @pytest.mark.asyncio
    async def test_present_options_sentinel(self) -> None:
        result = await execute_tool(
            "present_options",
            {
                "prompt": "Pick a model",
                "options": [{"label": "Qwen"}, {"label": "Llama"}],
            },
            repo=None,
        )
        assert result["__present_options__"] is True
        assert result["prompt"] == "Pick a model"
        assert len(result["options"]) == 2

    @pytest.mark.asyncio
    async def test_unknown_tool(self) -> None:
        result = await execute_tool("nonexistent_tool", {}, repo=None)
        assert "error" in result
        assert "Unknown tool" in result["error"]


class TestToolResultSummary:
    """Test tool_result_summary function."""

    def test_error_result(self) -> None:
        assert tool_result_summary("any", {"error": "boom"}) == "Error: boom"

    def test_list_jobs(self) -> None:
        s = tool_result_summary("list_jobs", {"jobs": [{"id": "1"}, {"id": "2"}]})
        assert s == "Found 2 job(s)"

    def test_submit_job(self) -> None:
        s = tool_result_summary("submit_training_job", {"id": "abc123"})
        assert s == "Job created: abc123"

    def test_estimate_vram(self) -> None:
        s = tool_result_summary("estimate_vram", {"estimated_vram_gb": 3.5})
        assert s == "Estimated VRAM: 3.5 GB"


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


def _mock_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


class TestProcessMessage:
    """Test process_message with mocked OpenAI client."""

    @pytest.mark.asyncio
    @patch("amortized.agent.chat._build_client")
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
    @patch("amortized.agent.chat.execute_tool")
    @patch("amortized.agent.chat._build_client")
    async def test_tool_call_then_response(
        self, mock_build: MagicMock, mock_execute: AsyncMock
    ) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        tc = _mock_tool_call("tc1", "list_sdg_flows", {})
        choice1 = _mock_choice(tool_calls=[tc], finish_reason="tool_calls")
        resp1 = MagicMock()
        resp1.choices = [choice1]

        choice2 = _mock_choice(content="Found 5 flows.")
        resp2 = MagicMock()
        resp2.choices = [choice2]

        client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])
        mock_execute.return_value = {"flows": [{"id": "f1"}, {"id": "f2"}]}

        result = await process_message("list flows")
        assert result.text == "Found 5 flows."
        mock_execute.assert_called_once_with("list_sdg_flows", {}, None)

    @pytest.mark.asyncio
    @patch("amortized.agent.chat.execute_tool")
    @patch("amortized.agent.chat._build_client")
    async def test_propose_action(self, mock_build: MagicMock, mock_execute: AsyncMock) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        tc = _mock_tool_call(
            "tc1",
            "propose_action",
            {
                "action_type": "submit_training_job",
                "config": {"model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"},
                "label": "Start Training",
            },
        )
        choice1 = _mock_choice(tool_calls=[tc], finish_reason="tool_calls")
        resp1 = MagicMock()
        resp1.choices = [choice1]

        choice2 = _mock_choice(content="Ready to train. Click the button to confirm.")
        resp2 = MagicMock()
        resp2.choices = [choice2]

        client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])
        mock_execute.return_value = {
            "__proposed_action__": True,
            "action_type": "submit_training_job",
            "config": {"model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"},
            "label": "Start Training",
        }

        result = await process_message("train a model")
        assert result.proposed_action is not None
        assert result.proposed_action["type"] == "submit_training_job"
        assert result.proposed_action["label"] == "Start Training"

    @pytest.mark.asyncio
    @patch("amortized.agent.chat.execute_tool")
    @patch("amortized.agent.chat._build_client")
    async def test_present_options(self, mock_build: MagicMock, mock_execute: AsyncMock) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        tc = _mock_tool_call(
            "tc1",
            "present_options",
            {
                "prompt": "Choose a model",
                "options": [{"label": "Qwen 1.5B"}, {"label": "Llama 7B"}],
            },
        )
        choice1 = _mock_choice(tool_calls=[tc], finish_reason="tool_calls")
        resp1 = MagicMock()
        resp1.choices = [choice1]

        choice2 = _mock_choice(content="Let me know which model you prefer.")
        resp2 = MagicMock()
        resp2.choices = [choice2]

        client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])
        mock_execute.return_value = {
            "__present_options__": True,
            "prompt": "Choose a model",
            "options": [{"label": "Qwen 1.5B"}, {"label": "Llama 7B"}],
        }

        result = await process_message("which model should I use?")
        assert result.presented_options is not None
        assert result.presented_options["prompt"] == "Choose a model"
        assert len(result.presented_options["options"]) == 2

    @pytest.mark.asyncio
    @patch("amortized.agent.chat._build_client")
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
    @patch("amortized.agent.chat._build_client")
    async def test_simple_text_streaming(self, mock_build: MagicMock) -> None:
        client = AsyncMock()
        mock_build.return_value = client

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

        deltas = [e for e in events if e.type == EventType.delta]
        assert len(deltas) == 2
        assert deltas[0].data["text"] == "Hello "
        assert deltas[1].data["text"] == "world"

        done_events = [e for e in events if e.type == EventType.done]
        assert len(done_events) == 1
        assert done_events[0].data["full_text"] == "Hello world"

    @pytest.mark.asyncio
    @patch("amortized.agent.chat._build_client")
    async def test_streaming_error_handling(self, mock_build: MagicMock) -> None:
        client = AsyncMock()
        mock_build.return_value = client

        client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        events: list[StreamEvent] = []
        async for event in stream_message("hello"):
            events.append(event)

        error_events = [e for e in events if e.type == EventType.error]
        assert len(error_events) == 1
        assert "API error" in error_events[0].data["error"]
