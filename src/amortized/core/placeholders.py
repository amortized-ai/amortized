"""Canonical placeholder tokens used in starter templates.

Starter templates (agents/*/skills/**/reference-payload.json) ship these tokens
for the agent to resolve via its tools (list_documents, list_models) or for
lineage chaining (parent_job_id). Keeping the exact strings here gives the agent
side a single authority and lets tests guard against spelling drift.

Mirror of studio/src/features/recipes/lib/placeholders.ts — keep the two in sync.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

PLACEHOLDER_DOCUMENT_ID = "<doc-id-from-list_documents>"
PLACEHOLDER_MODEL = "<from-list_models>"
PLACEHOLDER_PARENT_JOB_ID = "<SDG_JOB_ID>"

PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {PLACEHOLDER_DOCUMENT_ID, PLACEHOLDER_MODEL, PLACEHOLDER_PARENT_JOB_ID}
)


def is_placeholder(value: str) -> bool:
    return value in PLACEHOLDER_TOKENS


def find_unresolved_placeholders(config: object) -> list[str]:
    """Distinct placeholder tokens still present anywhere in a config subtree.

    A token is an agent-resolution hint, never a valid value; a non-empty result
    means the config was submitted (e.g. by the agent) without resolving one.
    """
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, str):
            if value in PLACEHOLDER_TOKENS:
                found.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(config)
    return sorted(found)


# Prompt-store references (#541). A reviewed prompt is persisted once and cited
# from an SDG config by an unguessable id, so the config stays small at emission.
# The token is the WHOLE value of a prompt field (a column's system_prompt, a
# processor's system message) — it is a plain Jinja-safe string, never embedded
# in surrounding text — so resolution is a full-value swap, mirroring how
# document-id/model placeholders occupy an entire field.
PROMPT_ID_PREFIX = "prm_"
_PROMPT_REF_RE = re.compile(r"^<prompt:(prm_[A-Za-z0-9]+)>$")


def make_prompt_ref(prompt_id: str) -> str:
    return f"<prompt:{prompt_id}>"


def is_prompt_ref(value: object) -> bool:
    return isinstance(value, str) and _PROMPT_REF_RE.match(value) is not None


def prompt_ref_id(value: object) -> str | None:
    """The prompt id a value references, or None if the value is not a ref token."""
    if not isinstance(value, str):
        return None
    match = _PROMPT_REF_RE.match(value)
    return match.group(1) if match else None


def find_prompt_refs(config: object) -> list[str]:
    """Distinct prompt ids referenced anywhere in a config subtree (sorted)."""
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, str):
            ref = prompt_ref_id(value)
            if ref is not None:
                found.add(ref)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(config)
    return sorted(found)


def resolve_prompt_refs(config: Any, resolver: Callable[[str], str | None]) -> Any:
    """Return a copy of `config` with every `<prompt:...>` ref replaced by its text.

    Pure: `config` is not mutated, so a persisted ref-form config stays ref-form
    while the rendered copy carries full text. `resolver(id)` returns the stored
    text (or None); an unresolved ref is left as-is for the caller to reject —
    callers fail closed by checking find_prompt_refs/resolver first.
    """

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            ref = prompt_ref_id(value)
            if ref is not None:
                text = resolver(ref)
                return text if text is not None else value
            return value
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(item) for item in value]
        return value

    return walk(config)
