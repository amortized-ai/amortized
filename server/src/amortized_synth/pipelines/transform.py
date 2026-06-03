"""Data transformation/augmentation pipeline."""

from __future__ import annotations

from typing import Any

from amortized_synth.pipelines.base import BasePipeline
from amortized_synth.types import Conversation, Turn

_DEFAULT_TRANSFORM_PROMPT = (
    "Transform the following text according to the instructions. "
    "Output ONLY the transformed text, nothing else."
)


class TransformPipeline(BasePipeline):
    """Takes existing data and transforms it (rephrase, translate, augment).

    Maps input rows to output rows via LLM calls. Single-turn, no conversation state.
    """

    name = "transform"
    description = "Data transformation and augmentation via LLM"
    supports_multi_turn = False

    def __init__(self, system_prompt: str | None = None, **_kwargs: Any) -> None:
        self._system_prompt = system_prompt or _DEFAULT_TRANSFORM_PROMPT

    def build_prompt(self, conversation: Conversation, turn_num: int) -> list[dict[str, str]]:
        instruction = conversation.seed.get("instruction", "Rephrase the following text.")
        text = conversation.seed.get("text", conversation.seed.get("input", ""))
        user_msg = f"{instruction}\n\n{text}" if text else instruction
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def process_response(
        self, conversation: Conversation, response: str, turn_num: int
    ) -> None:
        conversation.turns.append(Turn(role="assistant", content=response))
        conversation.status = "completed"

    def format_output(self, conversation: Conversation) -> dict[str, Any]:
        return {
            "id": conversation.id,
            "input": conversation.seed.get("text", conversation.seed.get("input", "")),
            "instruction": conversation.seed.get("instruction", ""),
            "output": conversation.turns[0].content if conversation.turns else "",
        }

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "system_prompt": {
                "type": "string",
                "description": "System prompt for the transformation",
            },
        }
