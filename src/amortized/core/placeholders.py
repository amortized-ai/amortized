"""Canonical placeholder tokens used in starter templates.

Starter templates (agents/*/skills/**/reference-payload.json) ship these tokens
for the agent to resolve via its tools (list_documents, list_models) or for
lineage chaining (parent_job_id). Keeping the exact strings here gives the agent
side a single authority and lets tests guard against spelling drift.

Mirror of studio/src/features/recipes/lib/placeholders.ts — keep the two in sync.
"""

from __future__ import annotations

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
