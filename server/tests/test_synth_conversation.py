"""Tests for the conversation synthesis pipeline."""

from __future__ import annotations

from amortized_synth.pipelines.conversation import ConversationPipeline
from amortized_synth.types import Conversation, Turn


class TestConversationPipeline:
    def test_build_initial_prompt(self) -> None:
        pipeline = ConversationPipeline()
        conv = Conversation(id="test_0", seed={"topic": "python", "persona": "a student"})
        prompt = pipeline.build_prompt(conv, turn_num=0)

        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert "python" in prompt[1]["content"]
        assert "student" in prompt[1]["content"]

    def test_build_assistant_prompt(self) -> None:
        pipeline = ConversationPipeline()
        conv = Conversation(
            id="test_0",
            turns=[Turn(role="user", content="What is Python?")],
        )
        prompt = pipeline.build_prompt(conv, turn_num=1)

        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert prompt[1]["content"] == "What is Python?"

    def test_build_user_followup_prompt(self) -> None:
        pipeline = ConversationPipeline()
        conv = Conversation(
            id="test_0",
            turns=[
                Turn(role="user", content="What is Python?"),
                Turn(role="assistant", content="Python is a programming language."),
            ],
        )
        prompt = pipeline.build_prompt(conv, turn_num=2)

        assert prompt[0]["role"] == "system"
        assert "USER:" in prompt[1]["content"]
        assert "ASSISTANT:" in prompt[1]["content"]

    def test_process_response_turn_0(self) -> None:
        pipeline = ConversationPipeline()
        conv = Conversation(id="test_0", seed={"topic": "python"})
        pipeline.process_response(conv, "What is Python?", turn_num=0)

        assert len(conv.turns) == 1
        assert conv.turns[0].role == "user"
        assert conv.turns[0].content == "What is Python?"

    def test_process_response_assistant(self) -> None:
        pipeline = ConversationPipeline()
        conv = Conversation(
            id="test_0",
            turns=[Turn(role="user", content="What is Python?")],
        )
        pipeline.process_response(conv, "Python is great.", turn_num=1)

        assert len(conv.turns) == 2
        assert conv.turns[1].role == "assistant"

    def test_format_output(self) -> None:
        pipeline = ConversationPipeline()
        conv = Conversation(
            id="test_0",
            seed={"topic": "python"},
            turns=[
                Turn(role="user", content="Q"),
                Turn(role="assistant", content="A"),
            ],
        )
        output = pipeline.format_output(conv)

        assert output["id"] == "test_0"
        assert len(output["messages"]) == 3  # system + user + assistant
        assert output["messages"][0]["role"] == "system"

    def test_custom_system_prompt(self) -> None:
        pipeline = ConversationPipeline(system_prompt="You are a pirate.")
        conv = Conversation(
            id="test_0",
            turns=[Turn(role="user", content="Hello")],
        )
        prompt = pipeline.build_prompt(conv, turn_num=1)

        assert prompt[0]["content"] == "You are a pirate."

    def test_config_schema(self) -> None:
        schema = ConversationPipeline.config_schema()
        assert "system_prompt" in schema
        assert "user_simulator_prompt" in schema
