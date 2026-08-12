"""add index on parent_job_id for lineage queries

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-10
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_parent_job_id ON jobs(parent_job_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_jobs_parent_job_id")
