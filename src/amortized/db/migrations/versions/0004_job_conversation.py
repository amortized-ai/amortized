"""Job -> conversation mapping for push-based continuation

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-07

- jobs.conversation_id: the agent-proxy session id (the _sessions key, which the
  frontend sends as X-Conversation-Id on create) that dispatched this job. Lets a
  backend watcher drive the next-step turn when the job finishes even if the chat
  tab is closed. Empty = no watcher, falls back to the frontend job-monitor card.
- jobs.continuation_claimed_at: the atomic single-fire guard shared by the watcher
  and the frontend card, so the job-complete continuation is driven exactly once no
  matter which actor observes the terminal state first.

Both columns are additive and default to '' — code predating them ignores them.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS conversation_id TEXT NOT NULL DEFAULT ''")
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS continuation_claimed_at TEXT NOT NULL DEFAULT ''"
    )


def downgrade() -> None:
    # No-op: additive columns, harmless if rolled back (CI forbids destructive
    # schema changes and code predating them ignores them entirely).
    pass
