"""Base pipeline interface for Amortized Synth."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from amortized_synth.types import Conversation


class BasePipeline(ABC):
    """A synthesis pipeline defines how to build prompts and process responses."""

    name: str = ""
    description: str = ""
    supports_multi_turn: bool = False

    @abstractmethod
    def build_prompt(self, conversation: Conversation, turn_num: int) -> list[dict[str, str]]:
        """Build the LLM messages for the next turn of this conversation."""

    @abstractmethod
    def process_response(
        self, conversation: Conversation, response: str, turn_num: int
    ) -> None:
        """Process the LLM response. Mutate conversation in place."""

    @abstractmethod
    def format_output(self, conversation: Conversation) -> dict[str, Any]:
        """Convert a completed conversation to the output format."""

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        """Return a JSON schema describing pipeline-specific configuration."""
        return {}
