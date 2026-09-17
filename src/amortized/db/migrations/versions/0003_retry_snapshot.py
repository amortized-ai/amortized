"""request_config snapshot and retry lineage for eval jobs

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-17

- jobs.request_config: the pre-dispatch config, written at job creation
  and never touched by the worker (which overwrites jobs.config with
  resolved/injected fields). retry_job clones from this snapshot so a
  re-run reproduces the original request verbatim — rubric text included,
  which keeps scores in the same comparison group.
- jobs.retry_of: lineage column linking a retried job to its original.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS request_config JSONB")
    op.execute("ALTER TABLE jobs ALTER COLUMN request_config SET DEFAULT '{}'::jsonb")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS retry_of TEXT NOT NULL DEFAULT ''")
    # Snapshot for existing rows: best effort — later rows carry the
    # worker-resolved config (retry strips the known runtime-injected
    # keys for those).
    op.execute("UPDATE jobs SET request_config = config WHERE request_config IS NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS retry_of")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS request_config")
