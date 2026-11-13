"""Data migrations, run against the real test database (alembic up to a revision, seed, upgrade)."""

from __future__ import annotations

import json
import os
import subprocess

import asyncpg
import pytest
from conftest import TEST_DATABASE_URL


def _alembic(target: str) -> None:
    env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
    subprocess.run(["alembic", "upgrade", target], capture_output=True, env=env, check=True)


@pytest.mark.asyncio
async def test_0005_pins_legacy_classifier_alias_to_embedding_sft() -> None:
    # "classifier" meant embedding_sft before it was repointed to classifier_sft; stored
    # pre-upgrade training configs (and their retry snapshots) must keep that meaning.
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    try:
        await conn.execute("DROP TABLE IF EXISTS alembic_version")
        await conn.execute("DROP TABLE IF EXISTS jobs")
        _alembic("0004")
        rows = {
            "legacy": ("training", {"algorithm": "classifier", "model_name_or_path": "m"},
                       {"algorithm": " classifier "}),
            "router": ("training", {"algorithm": "embedding"}, {}),
            "llm": ("training", {"algorithm": "osft"}, None),
            "eval": ("eval", {"algorithm": "classifier"}, {}),
        }
        for job_id, (job_type, config, snapshot) in rows.items():
            await conn.execute(
                "INSERT INTO jobs (id, type, status, config, request_config, created_at)"
                " VALUES ($1, $2, 'failed', $3::jsonb, $4::jsonb, now())",
                job_id, job_type, json.dumps(config),
                None if snapshot is None else json.dumps(snapshot),
            )
        _alembic("0005")

        got = {}
        for r in await conn.fetch("SELECT id, config::text, request_config::text FROM jobs"):
            snapshot = r["request_config"] and json.loads(r["request_config"])
            got[r["id"]] = (json.loads(r["config"]), snapshot)
        assert got["legacy"][0] == {"algorithm": "embedding_sft", "model_name_or_path": "m"}
        assert got["legacy"][1] == {"algorithm": "embedding_sft"}
        assert got["router"][0] == {"algorithm": "embedding"}  # unchanged meaning, untouched
        assert got["llm"] == ({"algorithm": "osft"}, None)
        assert got["eval"][0] == {"algorithm": "classifier"}  # only training rows are rewritten
    finally:
        # Leave the schema at head (other tests assume the jobs table exists), minus our rows.
        await conn.execute("DELETE FROM jobs WHERE id = ANY($1::text[])", list(rows))
        await conn.close()
        _alembic("head")
