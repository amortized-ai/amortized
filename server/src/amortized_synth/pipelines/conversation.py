"""Multi-turn conversation synthesis pipeline."""

from __future__ import annotations

from typing import Any

from amortized_synth.pipelines.base import BasePipeline
from amortized_synth.types import Conversation, Turn

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Respond clearly and concisely."
)

_DEFAULT_USER_SIMULATOR_PROMPT = (
    "You are simulating a user in a conversation with an AI assistant. "
    "Generate a natural follow-up message based on the conversation so far. "
    "Be specific and ask relevant follow-up questions or provide additional context. "
    "Respond with ONLY the user's message, no meta-commentary."
)


class ConversationPipeline(BasePipeline):
    """Multi-turn conversation synthesis.

    Turn 0: generates the user's first message from the seed topic.
    Odd turns: assistant responses (LLM call with assistant system prompt).
    Even turns: simulated user follow-ups (LLM call with user simulator prompt).
    """

    name = "conversation"
    description = "Multi-turn conversation synthesis with simulated user and assistant"
    supports_multi_turn = True

    def __init__(
        self,
        system_prompt: str | None = None,
        user_simulator_prompt: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._user_simulator_prompt = user_simulator_prompt or _DEFAULT_USER_SIMULATOR_PROMPT

    def build_prompt(self, conversation: Conversation, turn_num: int) -> list[dict[str, str]]:
        if turn_num == 0:
            return self._build_initial_user_prompt(conversation)
        if turn_num % 2 == 1:
            return self._build_assistant_prompt(conversation)
        return self._build_user_followup_prompt(conversation)

    def process_response(
        self, conversation: Conversation, response: str, turn_num: int
    ) -> None:
        if turn_num == 0:
            conversation.turns.append(Turn(role="user", content=response))
        elif turn_num % 2 == 1:
            conversation.turns.append(Turn(role="assistant", content=response))
        else:
            conversation.turns.append(Turn(role="user", content=response))

    def format_output(self, conversation: Conversation) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self._system_prompt},
            *[{"role": t.role, "content": t.content} for t in conversation.turns],
        ]
        return {
            "id": conversation.id,
            "messages": messages,
            "metadata": {
                "seed": conversation.seed,
                "attributes": conversation.attributes,
            },
        }

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "system_prompt": {"type": "string", "description": "System prompt for the assistant"},
            "user_simulator_prompt": {
                "type": "string",
                "description": "System prompt for simulated user",
            },
        }

    def _build_initial_user_prompt(self, conversation: Conversation) -> list[dict[str, str]]:
        topic = conversation.seed.get("topic", "general knowledge")
        persona = conversation.seed.get("persona", "a curious user")
        constraints = conversation.seed.get("constraints", "")
        constraint_text = f"\nConstraints: {constraints}" if constraints else ""

        prompt = (
            f"You are simulating {persona}. "
            f"Write an opening message to an AI assistant about: {topic}. "
            f"Be specific and natural.{constraint_text}\n\n"
            f"Respond with ONLY the user's message, no meta-commentary."
        )
        return [
            {"role": "system", "content": self._user_simulator_prompt},
            {"role": "user", "content": prompt},
        ]

    def _build_assistant_prompt(self, conversation: Conversation) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt}]
        for turn in conversation.turns:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def _build_user_followup_prompt(self, conversation: Conversation) -> list[dict[str, str]]:
        history = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in conversation.turns
        )
        prompt = (
            f"Continue this conversation as the user. Here is the history:\n\n"
            f"{history}\n\nRespond with ONLY the user's next message."
        )
        return [
            {"role": "system", "content": self._user_simulator_prompt},
            {"role": "user", "content": prompt},
        ]
