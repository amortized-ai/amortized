"""use proper PostgreSQL types for timestamps and config

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-06
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE jobs
            ALTER COLUMN config TYPE JSONB USING config::jsonb,
            ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at::timestamptz,
            ALTER COLUMN started_at TYPE TIMESTAMPTZ
                USING NULLIF(started_at, '')::timestamptz,
            ALTER COLUMN completed_at TYPE TIMESTAMPTZ
                USING NULLIF(completed_at, '')::timestamptz
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE jobs
            ALTER COLUMN config TYPE TEXT USING config::text,
            ALTER COLUMN created_at TYPE TEXT USING created_at::text,
            ALTER COLUMN started_at TYPE TEXT
                USING COALESCE(started_at::text, ''),
            ALTER COLUMN completed_at TYPE TEXT
                USING COALESCE(completed_at::text, '')
        """
    )
