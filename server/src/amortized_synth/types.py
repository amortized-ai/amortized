"""Core types for Amortized Synth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """A single turn in a conversation."""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A multi-turn conversation being synthesized."""

    id: str
    turns: list[Turn] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    seed: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    retries: int = 0


@dataclass
class SynthStats:
    """Statistics from a synthesis run."""

    total_requested: int
    total_completed: int
    total_failed: int
    total_tokens_used: int
    total_turns_generated: int
    elapsed_seconds: float


@dataclass
class SynthResult:
    """Result of a synthesis run."""

    conversations: list[Conversation]
    stats: SynthStats
