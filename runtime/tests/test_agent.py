"""Tests for the Claude-powered agent."""

from unittest.mock import MagicMock, patch

from amortized_runtime.agent import SYSTEM_PROMPT, _build_messages, process_message


class TestSystemPrompt:
    """Verify system prompt contains required knowledge."""

    def test_contains_training_hub_info(self) -> None:
        assert "lora_sft" in SYSTEM_PROMPT
        assert "LoRAEstimator" in SYSTEM_PROMPT
        assert "QLoRAEstimator" in SYSTEM_PROMPT

    def test_contains_sdg_hub_info(self) -> None:
        assert "FlowRegistry" in SYSTEM_PROMPT
        assert "Flow.from_yaml" in SYSTEM_PROMPT
        assert "set_model_config" in SYSTEM_PROMPT

    def test_contains_api_endpoints(self) -> None:
        assert "POST /api/v1/jobs/training" in SYSTEM_PROMPT
        assert "POST /api/v1/jobs/sdg" in SYSTEM_PROMPT
        assert "GET /api/v1/jobs" in SYSTEM_PROMPT
        assert "GET /api/v1/flows" in SYSTEM_PROMPT
        assert "POST /api/v1/estimate" in SYSTEM_PROMPT

    def test_contains_amortization_workflow(self) -> None:
        assert "Generate data" in SYSTEM_PROMPT or "Generate" in SYSTEM_PROMPT
        assert "Fine-tune" in SYSTEM_PROMPT
        assert "Deploy" in SYSTEM_PROMPT

    def test_contains_default_model(self) -> None:
        assert "Qwen/Qwen2.5-1.5B-Instruct" in SYSTEM_PROMPT


class TestBuildMessages:
    """Verify conversation history is passed correctly."""

    def test_empty_history(self) -> None:
        messages = _build_messages([], "hello")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_with_history(self) -> None:
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]
        messages = _build_messages(history, "how are you")
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "hi"}
        assert messages[1] == {"role": "assistant", "content": "hello there"}
        assert messages[2] == {"role": "user", "content": "how are you"}

    def test_preserves_order(self) -> None:
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response2"},
        ]
        messages = _build_messages(history, "third")
        assert len(messages) == 5
        assert messages[-1]["content"] == "third"


class TestProcessMessage:
    """Test the process_message function with mocked Anthropic client."""

    @patch("amortized_runtime.agent.settings")
    def test_no_api_key_returns_error(self, mock_settings: MagicMock) -> None:
        mock_settings.anthropic_api_key = ""
        result = process_message("hello")
        assert "API key" in result
        assert "AMORTIZED_ANTHROPIC_API_KEY" in result

    @patch("amortized_runtime.agent.get_client")
    @patch("amortized_runtime.agent.settings")
    def test_calls_claude_api(
        self, mock_settings: MagicMock, mock_get_client: MagicMock
    ) -> None:
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.anthropic_model = "claude-sonnet-4-20250514"

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "I can help you fine-tune a model."

        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = process_message("help me fine-tune")
        assert result == "I can help you fine-tune a model."

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["system"] == SYSTEM_PROMPT
        assert call_kwargs["messages"] == [{"role": "user", "content": "help me fine-tune"}]

    @patch("amortized_runtime.agent.get_client")
    @patch("amortized_runtime.agent.settings")
    def test_passes_conversation_history(
        self, mock_settings: MagicMock, mock_get_client: MagicMock
    ) -> None:
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.anthropic_model = "claude-sonnet-4-20250514"

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Sure, here's the status."

        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        history = [
            {"role": "user", "content": "start a training job"},
            {"role": "assistant", "content": "I started the job."},
        ]
        result = process_message("what's the status?", history=history)
        assert result == "Sure, here's the status."

        call_kwargs = mock_client.messages.create.call_args[1]
        assert len(call_kwargs["messages"]) == 3
        assert call_kwargs["messages"][0]["content"] == "start a training job"
        assert call_kwargs["messages"][2]["content"] == "what's the status?"


class TestStreamMessage:
    """Test the stream_message function."""

    @patch("amortized_runtime.agent.settings")
    def test_no_api_key_returns_none(self, mock_settings: MagicMock) -> None:
        from amortized_runtime.agent import stream_message

        mock_settings.anthropic_api_key = ""
        result = stream_message("hello")
        assert result is None

    @patch("amortized_runtime.agent.get_client")
    @patch("amortized_runtime.agent.settings")
    def test_returns_stream_context(
        self, mock_settings: MagicMock, mock_get_client: MagicMock
    ) -> None:
        from amortized_runtime.agent import stream_message

        mock_settings.anthropic_api_key = "test-key"
        mock_settings.anthropic_model = "claude-sonnet-4-20250514"

        mock_stream = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream
        mock_get_client.return_value = mock_client

        result = stream_message("hello")
        assert result is not None
        mock_client.messages.stream.assert_called_once()
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["system"] == SYSTEM_PROMPT
