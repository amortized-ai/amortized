"""strip API keys from stored job configs

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-12

Eval jobs used to accept an inline endpoint/judge api_key. The worker scrubbed it
from jobs.config at dispatch, but jobs.request_config (the retry snapshot) and jobs
that were never dispatched kept it in plaintext. Job configs no longer accept keys
(a model that needs one comes from a provider added under Settings); this removes
every api_key already stored in either column, and the user:password@ and query
parts of any stored base_url.
"""

import json
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _strip_url_credentials(url: str) -> str:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return ""  # can't be cleaned safely, and could never have been called
    if not (parts.username or parts.password or parts.query):
        return url
    host = parts.hostname or ""
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _strip_api_keys(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = str(key).lower().replace("-", "_")
            if name in ("api_key", "apikey"):
                continue
            if name == "base_url" and isinstance(item, str):
                out[key] = _strip_url_credentials(item)
            else:
                out[key] = _strip_api_keys(item)
        return out
    if isinstance(value, list):
        return [_strip_api_keys(item) for item in value]
    return value


def _load(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str) else raw


def upgrade() -> None:
    conn = op.get_bind()
    # Case-insensitive, and LIKE's "_" matches any character, so this covers every key
    # spelling _strip_api_keys normalizes (api_key, API-KEY, apiKey, BASE_URL, base-url).
    rows = conn.execute(
        sa.text(
            "SELECT id, config, request_config FROM jobs"
            " WHERE config::text ILIKE '%api%key%' OR request_config::text ILIKE '%api%key%'"
            " OR config::text ILIKE '%base_url%' OR request_config::text ILIKE '%base_url%'"
        )
    ).fetchall()
    for job_id, raw_config, raw_request_config in rows:
        config = _load(raw_config)
        request_config = _load(raw_request_config)
        stripped_config = _strip_api_keys(config)
        stripped_request_config = _strip_api_keys(request_config)
        if stripped_config == config and stripped_request_config == request_config:
            continue
        conn.execute(
            sa.text(
                "UPDATE jobs SET config = CAST(:config AS jsonb),"
                " request_config = CAST(:request_config AS jsonb) WHERE id = :id"
            ),
            {
                "id": job_id,
                "config": json.dumps(stripped_config),
                "request_config": (
                    None if stripped_request_config is None else json.dumps(stripped_request_config)
                ),
            },
        )


def downgrade() -> None:
    # No-op: the removed keys cannot (and must not) be restored.
    pass
