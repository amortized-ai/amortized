"""Core synthesis engine with batched turn-by-turn generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

from amortized_synth.inference import InferenceClient
from amortized_synth.pipelines.base import BasePipeline
from amortized_synth.types import Conversation, SynthResult, SynthStats, Turn


class SynthEngine:
    """Core synthesis engine with batched turn-by-turn generation.

    The engine manages the generation loop:
    1. Initialize conversations from seed data
    2. Loop turn-by-turn: batch all conversations needing the next turn
    3. Handle stragglers: retry failed conversations
    4. Checkpoint periodically for resume
    """

    def __init__(self, client: InferenceClient, pipeline: BasePipeline) -> None:
        self.client = client
        self.pipeline = pipeline

    async def run(
        self,
        seeds: list[dict[str, Any]],
        *,
        max_turns: int = 5,
        max_retries: int = 2,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 50,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> SynthResult:
        """Run the synthesis loop."""
        start_time = time.monotonic()

        conversations = self._init_conversations(seeds)

        if checkpoint_dir and (checkpoint_dir / "state.json").exists():
            conversations = self._load_checkpoint(checkpoint_dir)

        completed = sum(1 for c in conversations if c.status == "completed")
        total = len(conversations)

        for turn_num in range(max_turns):
            active = [c for c in conversations if c.status == "in_progress"]
            if not active:
                break

            prompts = [self.pipeline.build_prompt(c, turn_num) for c in active]
            responses = await self.client.complete_batch(prompts)

            for conv, response in zip(active, responses, strict=True):
                if response is None:
                    conv.retries += 1
                    if conv.retries >= max_retries:
                        conv.status = "failed"
                    continue
                self.pipeline.process_response(conv, response, turn_num)

            newly_completed = sum(1 for c in conversations if c.status == "completed") - completed
            completed += newly_completed

            if on_progress:
                on_progress(completed, total)

            if checkpoint_dir and completed > 0 and completed % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, conversations)

        for c in conversations:
            if c.status == "in_progress":
                c.status = "completed"

        elapsed = time.monotonic() - start_time

        if checkpoint_dir:
            self._save_checkpoint(checkpoint_dir, conversations)

        return SynthResult(
            conversations=conversations,
            stats=self._compute_stats(conversations, elapsed),
        )

    def _init_conversations(self, seeds: list[dict[str, Any]]) -> list[Conversation]:
        return [
            Conversation(
                id=f"conv_{i}",
                seed=seed,
                attributes=seed.get("attributes", {}),
                status="in_progress",
            )
            for i, seed in enumerate(seeds)
        ]

    def _compute_stats(
        self, conversations: list[Conversation], elapsed: float
    ) -> SynthStats:
        return SynthStats(
            total_requested=len(conversations),
            total_completed=sum(1 for c in conversations if c.status == "completed"),
            total_failed=sum(1 for c in conversations if c.status == "failed"),
            total_tokens_used=self.client.total_tokens,
            total_turns_generated=sum(len(c.turns) for c in conversations),
            elapsed_seconds=round(elapsed, 2),
        )

    def _save_checkpoint(
        self, checkpoint_dir: Path, conversations: list[Conversation]
    ) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "conversations": [
                {
                    "id": c.id,
                    "turns": [
                        {"role": t.role, "content": t.content, "metadata": t.metadata}
                        for t in c.turns
                    ],
                    "attributes": c.attributes,
                    "seed": c.seed,
                    "status": c.status,
                    "retries": c.retries,
                }
                for c in conversations
            ]
        }
        with open(checkpoint_dir / "state.json", "w") as f:
            json.dump(state, f)

    def _load_checkpoint(self, checkpoint_dir: Path) -> list[Conversation]:
        with open(checkpoint_dir / "state.json") as f:
            state = json.load(f)
        conversations: list[Conversation] = []
        for cdata in state["conversations"]:
            conv = Conversation(
                id=cdata["id"],
                turns=[
                    Turn(
                        role=t["role"],
                        content=t["content"],
                        metadata=t.get("metadata", {}),
                    )
                    for t in cdata["turns"]
                ],
                attributes=cdata.get("attributes", {}),
                seed=cdata.get("seed", {}),
                status=cdata["status"],
                retries=cdata.get("retries", 0),
            )
            conversations.append(conv)
        return conversations
