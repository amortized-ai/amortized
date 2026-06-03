"""Attribute-guided single-turn generation pipeline."""

from __future__ import annotations

from typing import Any

from amortized_synth.pipelines.base import BasePipeline
from amortized_synth.types import Conversation, Turn


class AttributePipeline(BasePipeline):
    """Single-turn generation guided by attribute constraints.

    Generates one sample per seed, applying attribute constraints via the system prompt.
    No multi-turn loop — marks conversation as completed after the first response.
    """

    name = "attribute"
    description = "Attribute-guided single-turn generation"
    supports_multi_turn = False

    def __init__(self, attributes: dict[str, Any] | None = None, **_kwargs: Any) -> None:
        self._attributes = attributes or {}

    def build_prompt(self, conversation: Conversation, turn_num: int) -> list[dict[str, str]]:
        attrs = {**self._attributes, **conversation.attributes}
        attr_text = (
            "\n".join(f"- {k}: {v}" for k, v in attrs.items()) if attrs else "None specified"
        )

        seed_text = conversation.seed.get("instruction", conversation.seed.get("topic", ""))
        system = (
            f"Generate a response following these attribute constraints:\n{attr_text}\n\n"
            f"Respond directly with the generated content only."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": seed_text},
        ]

    def process_response(
        self, conversation: Conversation, response: str, turn_num: int
    ) -> None:
        conversation.turns.append(Turn(role="assistant", content=response))
        conversation.status = "completed"

    def format_output(self, conversation: Conversation) -> dict[str, Any]:
        instruction = conversation.seed.get("instruction", conversation.seed.get("topic", ""))
        output = conversation.turns[0].content if conversation.turns else ""
        return {
            "id": conversation.id,
            "instruction": instruction,
            "output": output,
            "attributes": {**self._attributes, **conversation.attributes},
        }

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "attributes": {
                "type": "object",
                "description": "Attribute constraints (e.g. style, domain, difficulty)",
            },
        }
