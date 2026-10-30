"""prompt store for reference-token SDG configs (#541)

Adds a `prompts` table so a reviewed prompt (shown via show_prompt) is persisted
once and referenced by an unguessable id from SDG configs as a `<prompt:prm_...>`
token, instead of being pasted inline. Keeping configs small at EMISSION is what
fixes the config_json corruption a weak driver hits when it re-serializes a
~20KB inline config; the full text is re-materialized only at the container
boundary (jobs/sdg.py build()).

Schema-only and additive: no backfill. Existing inline configs keep working (ref
resolution is a no-op when a config carries no refs), so this is harmless to
users still on an older image (they gain an empty table they never read).

- content_sha256 is UNIQUE: identical prompt text dedups to one row, so
  show_prompt is idempotent and re-showing the same prompt reuses its id.
- user_id starts empty (the MCP path is service-to-service with no auth header);
  ownership is stamped later by the proxy from the session's messages.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS prompts (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL DEFAULT '',
            conversation_id TEXT NOT NULL DEFAULT '',
            role            TEXT NOT NULL DEFAULT '',
            label           TEXT NOT NULL DEFAULT '',
            purpose         TEXT NOT NULL DEFAULT '',
            content         TEXT NOT NULL,
            content_sha256  TEXT NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS prompts_content_sha256_key"
        " ON prompts (content_sha256)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS prompts_user_id_idx ON prompts (user_id)")


def downgrade() -> None:
    # No-op: CI forbids destructive schema changes, and the table is additive and
    # harmless if this migration is rolled back (code predating it ignores it).
    pass
