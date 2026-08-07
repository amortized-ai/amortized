"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-05
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            type            TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'queued',
            config          TEXT NOT NULL DEFAULT '{}',
            recipe          TEXT DEFAULT '',
            user_id         TEXT DEFAULT '',
            k8s_job_name    TEXT DEFAULT '',
            k8s_namespace   TEXT DEFAULT '',
            mlflow_run_id   TEXT DEFAULT '',
            mlflow_experiment TEXT DEFAULT '',
            parent_job_id   TEXT DEFAULT '',
            error           TEXT DEFAULT '',
            created_at      TEXT NOT NULL,
            started_at      TEXT DEFAULT '',
            completed_at    TEXT DEFAULT '',
            backend_handle  TEXT DEFAULT ''
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_jobs_user_id")
    op.execute("DROP INDEX IF EXISTS idx_jobs_created_at")
    op.execute("DROP INDEX IF EXISTS idx_jobs_type")
    op.execute("DROP INDEX IF EXISTS idx_jobs_status")
    op.execute("DROP TABLE IF EXISTS jobs")
