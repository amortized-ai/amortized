"""model-liveness check report on jobs

Adds `jobs.model_check_report`: the result of checking a job's LLM endpoints (eval judge,
SDG teacher models, ...) when it was created. An endpoint that couldn't be verified (e.g. the
check service was unavailable) is recorded as `unverified` so the warning survives a refresh
and shows in job details.

Additive with a non-null default: existing rows get an empty report, and an older image that
never writes the column keeps working.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS model_check_report JSONB"
        " NOT NULL DEFAULT '[]'::jsonb"
    )


def downgrade() -> None:
    # No-op: CI forbids destructive schema changes, and the column is additive and harmless
    # if this migration is rolled back (code predating it ignores it).
    pass
