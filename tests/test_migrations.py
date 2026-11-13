"""Tests for alembic migrations."""

import os
import subprocess

import asyncpg
import pytest
from conftest import TEST_DATABASE_URL


@pytest.fixture(autouse=True)
async def _clean_db():
    """Drop all tables before each test for a clean slate."""
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    await conn.execute("DROP TABLE IF EXISTS alembic_version")
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.close()
    yield
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    await conn.execute("DROP TABLE IF EXISTS alembic_version")
    await conn.execute("DROP TABLE IF EXISTS jobs")
    await conn.close()


class TestAlembicMigrations:
    def test_upgrade_creates_schema(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"alembic upgrade failed: {result.stderr}"

    @pytest.mark.asyncio
    async def test_upgrade_creates_jobs_table(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)

        conn = await asyncpg.connect(TEST_DATABASE_URL)
        row = await conn.fetchrow(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'jobs'"
        )
        await conn.close()
        assert row is not None

    @pytest.mark.asyncio
    async def test_downgrade_drops_jobs_table(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)
        subprocess.run(["alembic", "downgrade", "base"], capture_output=True, env=env, check=True)

        conn = await asyncpg.connect(TEST_DATABASE_URL)
        row = await conn.fetchrow(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'jobs'"
        )
        await conn.close()
        assert row is None

    def test_current_shows_head_after_upgrade(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)
        result = subprocess.run(["alembic", "current"], capture_output=True, text=True, env=env)
        # Dynamically find the latest migration revision
        versions_dir = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "src",
            "amortized",
            "db",
            "migrations",
            "versions",
        )
        revisions = sorted(
            f for f in os.listdir(versions_dir) if f.endswith(".py") and not f.startswith("_")
        )
        latest_rev = revisions[-1].split("_")[0]  # e.g. "0002" from "0002_proper_pg_types.py"
        assert latest_rev in result.stdout

    def test_upgrade_is_idempotent(self) -> None:
        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        result1 = subprocess.run(
            ["alembic", "upgrade", "head"], capture_output=True, text=True, env=env
        )
        assert result1.returncode == 0
        result2 = subprocess.run(
            ["alembic", "upgrade", "head"], capture_output=True, text=True, env=env
        )
        assert result2.returncode == 0

    @pytest.mark.asyncio
    async def test_0005_strips_stored_api_keys(self) -> None:
        import json

        env = {**os.environ, "AMORTIZED_DATABASE_URL": TEST_DATABASE_URL}
        subprocess.run(["alembic", "upgrade", "0004"], capture_output=True, env=env, check=True)

        leaked = {
            "endpoint": {"base_url": "http://vllm:8000/v1", "model": "m", "api_key": "sk-a"},
            "judge": {"base_url": "openai", "model": "gpt-5", "api_key": "sk-b"},
            "endpoint_tuned": {"base_url": "https://u:sk-c@v.example.com:8443/v1?token=sk-d"},
            "rubric": [{"name": "r", "description": "d"}],
        }
        clean = {"algorithm": "sft", "data_run_id": "run-1"}
        url_only = {"judge": {"BASE_URL": "https://u:sk-e@v.example.com/v1?token=sk-f"}}
        bad_port = {"judge": {"base_url": "https://u:sk-g@v.example.com:99999/v1"}}
        conn = await asyncpg.connect(TEST_DATABASE_URL)
        await conn.execute(
            """INSERT INTO jobs (id, type, status, config, request_config, created_at) VALUES
               ('leaky', 'eval', 'failed', $1::jsonb, $2::jsonb, now()),
               ('clean', 'training', 'succeeded', $3::jsonb, $3::jsonb, now()),
               ('url-only', 'eval', 'failed', $4::jsonb, $4::jsonb, now()),
               ('bad-port', 'eval', 'failed', $5::jsonb, $5::jsonb, now())""",
            json.dumps({**leaked, "endpoint": {"base_url": "http://vllm:8000/v1", "model": "m"}}),
            json.dumps(leaked),
            json.dumps(clean),
            json.dumps(url_only),
            json.dumps(bad_port),
        )
        await conn.close()

        subprocess.run(["alembic", "upgrade", "head"], capture_output=True, env=env, check=True)

        conn = await asyncpg.connect(TEST_DATABASE_URL)
        rows = {
            r["id"]: (json.loads(r["config"]), json.loads(r["request_config"]))
            for r in await conn.fetch("SELECT id, config, request_config FROM jobs")
        }
        await conn.close()
        config, request_config = rows["leaky"]
        assert "sk-" not in json.dumps(config) + json.dumps(request_config)
        assert request_config["judge"] == {"base_url": "openai", "model": "gpt-5"}
        assert request_config["endpoint_tuned"] == {"base_url": "https://v.example.com:8443/v1"}
        assert request_config["rubric"] == leaked["rubric"]
        assert rows["clean"] == (clean, clean)
        assert rows["url-only"][1] == {"judge": {"BASE_URL": "https://v.example.com/v1"}}
        assert rows["bad-port"][1] == {"judge": {"base_url": ""}}
        assert "sk-" not in json.dumps(rows["url-only"]) + json.dumps(rows["bad-port"])
