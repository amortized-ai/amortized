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
