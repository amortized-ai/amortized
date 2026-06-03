"""LiteLLM inference client with async batching and concurrency control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import litellm


@dataclass
class ModelConfig:
    """Configuration for a model endpoint."""

    model: str
    api_base: str | None = None
    api_key: str | None = None
    max_concurrent: int = 16
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class InferenceClient:
    """Batched LiteLLM client with concurrency control."""

    config: ModelConfig
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    total_tokens: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        """Single completion with concurrency limiting."""
        async with self._semaphore:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if self.config.api_base is not None:
                kwargs["api_base"] = self.config.api_base
            if self.config.api_key is not None:
                kwargs["api_key"] = self.config.api_key

            response = await litellm.acompletion(**kwargs)
            usage = getattr(response, "usage", None)
            if usage:
                self.total_tokens += getattr(usage, "total_tokens", 0)
            content: str | None = response.choices[0].message.content
            return content or ""

    async def complete_batch(
        self, message_batches: list[list[dict[str, Any]]]
    ) -> list[str | None]:
        """Concurrent completions for a batch. Returns None for failed items."""
        tasks = [self.complete(msgs) for msgs in message_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if isinstance(r, str) else None for r in results]
