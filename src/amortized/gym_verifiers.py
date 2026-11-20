"""NeMo Gym verifiers the gym image ships — the reward pool for ``reward_type: gym``.

The list comes from the image itself: containers/gym/verifiers.txt picks the Gym resources
servers to install, the image build writes a manifest of each one's description and request
fields, and scripts/sync_gym_verifiers.sh copies that manifest to gym_verifiers.json here
(a test keeps it in sync with verifiers.txt). Nothing here is per-verifier code.

A training row for ``gym_verifier: <name>`` is the question plus that verifier's request
fields, passed to its ``/verify`` unchanged; the verifier's own schema is the format check
(check_gym_data.py runs it on the first rows before training starts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GymVerifier:
    name: str
    description: str
    domain: str
    fields: tuple[str, ...]
    """Request fields the verifier reads from a row, besides the question."""
    required: tuple[str, ...]
    """The subset its request schema requires (checked when the job is submitted)."""
    system_prompt: str | None = None
    """Default rollout system prompt when the job sets none (the answer format it parses)."""


# Answer-format instructions for verifiers whose parsing a bare question would not reveal.
_SYSTEM_PROMPTS = {
    "mcqa": (
        "Answer the multiple-choice question. Give only the letter of the correct option"
        " inside \\boxed{}, e.g. \\boxed{B}."
    ),
}


def _load() -> dict[str, GymVerifier]:
    manifest = json.loads((Path(__file__).with_name("gym_verifiers.json")).read_text())
    return {
        entry["name"]: GymVerifier(
            name=entry["name"],
            description=entry.get("description", ""),
            domain=entry.get("domain", ""),
            fields=tuple(entry.get("fields", ())),
            required=tuple(entry.get("required", ())),
            system_prompt=_SYSTEM_PROMPTS.get(entry["name"]),
        )
        for entry in manifest
    }


GYM_VERIFIERS: dict[str, GymVerifier] = _load()
