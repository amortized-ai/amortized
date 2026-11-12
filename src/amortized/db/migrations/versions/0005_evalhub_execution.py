"""Durable Eval Hub handoff and disposable execution projection."""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS execution_owner TEXT NOT NULL DEFAULT 'native'"
    )
    op.execute("""
        CREATE TABLE IF NOT EXISTS evalhub_executions (
            job_id TEXT PRIMARY KEY,
            tenant TEXT NOT NULL,
            hub_url TEXT NOT NULL,
            correlation TEXT NOT NULL UNIQUE,
            state TEXT NOT NULL DEFAULT 'preparing',
            remote_id TEXT,
            request JSONB,
            inputs JSONB NOT NULL DEFAULT '{}',
            snapshot JSONB NOT NULL DEFAULT '{}',
            cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
            sync_error TEXT NOT NULL DEFAULT '',
            observed_at TIMESTAMPTZ,
            serving_handle TEXT NOT NULL DEFAULT '',
            secret_name TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (hub_url, tenant, remote_id)
        )
    """)


def downgrade() -> None:
    # Additive migration; historical mappings must survive an application rollback.
    pass
