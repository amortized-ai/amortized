"""Prompt-reference lifecycle for SDG configs (#541).

show_prompt stores the prompt and returns a <prompt:...> ref; the config carries
the ref (small at emission); validate resolves it for the card; create stores it
ref-form and claims ownership; build() materializes the full text for the container
only, leaving the persisted config ref-form. Covers the corruption-avoidance path
end to end. DB-backed — runs where the test Postgres is migrated to head.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import httpx
import pytest
import yaml

from amortized.api.jobs import _refify_and_own_prompts
from amortized.core.placeholders import find_prompt_refs, is_prompt_ref, make_prompt_ref
from amortized.db.connection import get_pool
from amortized.db.repository import Repository
from amortized.jobs import sdg as sdg_builder
from amortized.main import app

TEST_DATABASE_URL = os.environ.get(
    "AMORTIZED_TEST_DATABASE_URL",
    "postgresql://amortized:amortized@localhost:5432/amortized_test",
)


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    import amortized.config as config_mod
    import amortized.db.connection as db_conn_mod

    os.environ["AMORTIZED_DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["AMORTIZED_DATA_DIR"] = str(tmp_path)
    new_settings = config_mod.Settings()
    config_mod.settings = new_settings
    db_conn_mod.settings = new_settings


@pytest.fixture
async def client() -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        from amortized.db import init_db

        await init_db()
        yield c  # type: ignore[misc]


def _unique(text: str) -> str:
    # Keep content_sha256 distinct per test run so dedup across runs doesn't surprise us.
    return f"{text} // {uuid.uuid4()}"


async def _store(text: str, **kw) -> str:
    async with get_pool().acquire() as conn:
        row = await Repository(conn).create_prompt(
            prompt_id=f"prm_{uuid.uuid4().hex}",
            content=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            created_at="2026-01-01T00:00:00+00:00",
            **kw,
        )
    return str(row["id"])


def _llm_text_column(system_prompt: str) -> dict:
    return {
        "column_type": "llm-text",
        "name": "output",
        "model_alias": "teacher",
        "system_prompt": system_prompt,
        "prompt": "Process this: {{ content }}",
    }


def _model_config() -> dict:
    return {"alias": "teacher", "model": "gpt-4o", "provider": "openrouter"}


# --- show_prompt persists + returns a ref -----------------------------------


@pytest.mark.asyncio
async def test_show_prompt_persists_and_returns_ref(client: httpx.AsyncClient) -> None:
    text = _unique("You are an assessor. Score 1-5.")
    resp = await client.post(
        "/api/v1/ui/show_prompt", json={"title": "Assessor", "prompt": text}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prompt_id"].startswith("prm_")
    assert body["prompt_ref"] == make_prompt_ref(body["prompt_id"])
    assert body["prompt_id"] in body["agent_instruction"]
    # The row is retrievable and holds the exact text.
    async with get_pool().acquire() as conn:
        assert await Repository(conn).get_prompt_content(body["prompt_id"]) == text


@pytest.mark.asyncio
async def test_show_prompt_dedups_identical_text(client: httpx.AsyncClient) -> None:
    text = _unique("identical")
    a = (await client.post("/api/v1/ui/show_prompt", json={"prompt": text})).json()
    b = (await client.post("/api/v1/ui/show_prompt", json={"prompt": text})).json()
    assert a["prompt_id"] == b["prompt_id"]


# --- refify (recipe form path + ownership) ----------------------------------


@pytest.mark.asyncio
async def test_refify_converts_inline_prompts_and_owns(client: httpx.AsyncClient) -> None:
    sys_text = _unique("inline assessor prompt")
    config = {
        "columns": [{"name": "output", "system_prompt": sys_text}],
        "processors": [
            {
                "name": "sft",
                "template": {
                    "messages": [
                        {"role": "system", "content": sys_text},
                        {"role": "assistant", "content": "{{ output }}"},
                    ]
                },
            }
        ],
    }
    async with get_pool().acquire() as conn:
        await _refify_and_own_prompts(Repository(conn), config, user_id="alice")
    col_ref = config["columns"][0]["system_prompt"]
    proc_ref = config["processors"][0]["template"]["messages"][0]["content"]
    assert is_prompt_ref(col_ref) and is_prompt_ref(proc_ref)
    # Identical text dedups to ONE row → both fields carry the SAME ref.
    assert col_ref == proc_ref
    async with get_pool().acquire() as conn:
        repo = Repository(conn)
        assert await repo.get_prompt_content(find_prompt_refs(config)[0]) == sys_text
        assert {p["id"] for p in await repo.list_prompts("alice")} >= set(find_prompt_refs(config))


@pytest.mark.asyncio
async def test_refify_passes_through_refs_and_stamps_owner(client: httpx.AsyncClient) -> None:
    pid = await _store(_unique("already a ref"))  # ownerless (agent path)
    config = {"columns": [{"name": "output", "system_prompt": make_prompt_ref(pid)}]}
    async with get_pool().acquire() as conn:
        await _refify_and_own_prompts(Repository(conn), config, user_id="bob")
    # Ref is untouched (not re-inlined), and ownership is now claimed.
    assert config["columns"][0]["system_prompt"] == make_prompt_ref(pid)
    async with get_pool().acquire() as conn:
        assert (await Repository(conn).get_prompt(pid))["user_id"] == "bob"


# --- validate resolves for the card; 422 on a dangling ref ------------------


@pytest.mark.asyncio
async def test_validate_resolves_ref_for_card(client: httpx.AsyncClient) -> None:
    text = _unique("You assess tickets.")
    pid = await _store(text)
    config = {
        "columns": [_llm_text_column(make_prompt_ref(pid))],
        "model_configs": [_model_config()],
        "processors": [
            {
                "name": "sft",
                "template": {
                    "messages": [
                        {"role": "system", "content": make_prompt_ref(pid)},
                        {"role": "user", "content": "{{ content }}"},
                        {"role": "assistant", "content": "{{ output }}"},
                    ]
                },
            }
        ],
        "mode": "preview",
    }
    resp = await client.post("/api/v1/jobs/sdg/validate", json=config)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Card shows the resolved text...
    assert any(p["text"] == text for p in body["prompts"])
    # ...but the config stays ref-form (what create will store).
    assert body["config"]["columns"][0]["system_prompt"] == make_prompt_ref(pid)


@pytest.mark.asyncio
async def test_validate_422_on_missing_ref(client: httpx.AsyncClient) -> None:
    missing = f"prm_{uuid.uuid4().hex}"  # guaranteed absent in the shared test DB
    config = {
        "columns": [_llm_text_column(make_prompt_ref(missing))],
        "model_configs": [_model_config()],
        "mode": "preview",
    }
    resp = await client.post("/api/v1/jobs/sdg/validate", json=config)
    assert resp.status_code == 422
    assert missing in resp.text


# --- create stores ref-form and the create guard rejects dangling refs ------


@pytest.mark.asyncio
async def test_create_stores_refform_and_owns(client: httpx.AsyncClient) -> None:
    text = _unique("You assess tickets at create.")
    pid = await _store(text)
    config = {
        "columns": [_llm_text_column(make_prompt_ref(pid))],
        "model_configs": [_model_config()],
        "mode": "preview",
    }
    resp = await client.post(
        "/api/v1/jobs/sdg", json=config, headers={"X-Forwarded-User": "carol"}
    )
    assert resp.status_code == 201, resp.text
    stored = resp.json()["config"]
    assert stored["columns"][0]["system_prompt"] == make_prompt_ref(pid)
    async with get_pool().acquire() as conn:
        assert (await Repository(conn).get_prompt(pid))["user_id"] == "carol"


@pytest.mark.asyncio
async def test_create_422_on_missing_ref(client: httpx.AsyncClient) -> None:
    ghost = f"prm_{uuid.uuid4().hex}"
    config = {
        "columns": [_llm_text_column(make_prompt_ref(ghost))],
        "model_configs": [_model_config()],
        "mode": "preview",
    }
    resp = await client.post(
        "/api/v1/jobs/sdg", json=config, headers={"X-Forwarded-User": "carol"}
    )
    assert resp.status_code == 422
    assert ghost in resp.text


# --- build() materializes text for the container, keeps persisted refs -------


@pytest.mark.asyncio
async def test_build_materializes_text_keeps_persisted_refform(client: httpx.AsyncClient) -> None:
    text = _unique("FULL ASSESSOR TEXT")
    pid = await _store(text)
    config = {
        "columns": [
            {"name": "output", "model_alias": "teacher", "system_prompt": make_prompt_ref(pid)}
        ],
        "model_configs": [{"alias": "teacher", "model": "m", "provider": "p"}],
        "mode": "preview",
        "num_records": 5,
    }
    result = await sdg_builder.build({"id": "job-ref"}, config, {})
    rendered = yaml.safe_load(result.config_files["config.yaml"])["data_designer"]
    # The container YAML carries the full text...
    assert rendered["columns"][0]["system_prompt"] == text
    # ...while the persisted config keeps the ref (clone-safe, small).
    assert result.resolved_config["columns"][0]["system_prompt"] == make_prompt_ref(pid)


@pytest.mark.asyncio
async def test_build_fails_closed_on_missing_ref(client: httpx.AsyncClient) -> None:
    ghost = f"prm_{uuid.uuid4().hex}"  # guaranteed absent in the shared test DB
    config = {
        "columns": [
            {"name": "output", "model_alias": "teacher", "system_prompt": make_prompt_ref(ghost)}
        ],
        "model_configs": [{"alias": "teacher", "model": "m", "provider": "p"}],
        "mode": "preview",
    }
    with pytest.raises(sdg_builder.SDGBuildError, match=ghost):
        await sdg_builder.build({"id": "job-missing"}, config, {})
