"""pin the legacy "classifier" training alias to embedding_sft

The "classifier" algorithm alias used to resolve to embedding_sft (the embedding
router); it now resolves to classifier_sft (a classification head). A training
job stored before the switch with algorithm "classifier" (retried, cloned, or
still queued across the upgrade) would otherwise silently train a head instead
of the router it was created for. Rewrite those stored configs to the concrete
algorithm they meant. New jobs store the resolved name (core.jobs.create_job),
so no stored config carries the bare alias after this.

Data-only; no schema change. Downgrade is a no-op: embedding_sft is what the
old code resolved "classifier" to anyway.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for column in ("config", "request_config"):
        op.execute(
            f"""
            UPDATE jobs
            SET {column} = jsonb_set({column}, '{{algorithm}}', '"embedding_sft"')
            WHERE type = 'training'
              AND {column} IS NOT NULL
              AND jsonb_typeof({column}) = 'object'
              AND btrim({column}->>'algorithm') = 'classifier'
            """
        )


def downgrade() -> None:
    pass
