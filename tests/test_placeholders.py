"""Starter templates must use the canonical placeholder tokens.

Guards against spelling drift (e.g. the historical "<from list_models>" vs
"<from-list_models>") between the templates and core.placeholders.
"""

from __future__ import annotations

from typing import Any

import pytest

from amortized.core.jobs import InvalidJobStateError, create_job
from amortized.core.placeholders import (
    PLACEHOLDER_DOCUMENT_ID,
    PLACEHOLDER_MODEL,
    PLACEHOLDER_PARENT_JOB_ID,
    find_prompt_refs,
    find_unresolved_placeholders,
    is_prompt_ref,
    make_prompt_ref,
    prompt_ref_id,
    resolve_prompt_refs,
)
from amortized.core.recipes import list_starter_templates
from amortized.models import JobType


def _configs_by_name() -> dict[str, dict[str, Any]]:
    return {t["name"]: t["config"] for t in list_starter_templates()}


def _required_by_name(name: str) -> list[str]:
    return {t["name"]: t.get("required", []) for t in list_starter_templates()}[name]


def test_knowledge_ingestion_uses_canonical_tokens() -> None:
    cfg = _configs_by_name()["Knowledge Ingestion"]
    assert cfg["document_ids"] == [PLACEHOLDER_DOCUMENT_ID]
    assert cfg["model_configs"][0]["model"] == PLACEHOLDER_MODEL
    assert cfg["model_configs"][0]["provider"] == PLACEHOLDER_MODEL


def test_knowledge_ingestion_declares_document_ids_required() -> None:
    assert _required_by_name("Knowledge Ingestion") == ["document_ids"]
    # Recipes that don't consume a document seed must not require it.
    assert _required_by_name("Task Distillation") == []


def test_training_templates_require_model_and_dataset() -> None:
    assert _required_by_name("Embedding Classifier") == ["model_name_or_path", "parent_job_id"]
    assert _required_by_name("Knowledge Ingestion OSFT") == ["model_name_or_path", "parent_job_id"]


def test_template_model_refs_use_canonical_token() -> None:
    for cfg in _configs_by_name().values():
        for mc in cfg.get("model_configs", []):
            for key in ("model", "provider"):
                value = mc.get(key, "")
                if isinstance(value, str) and value.startswith("<"):
                    assert value == PLACEHOLDER_MODEL


def test_template_parent_job_id_is_canonical_or_empty() -> None:
    for cfg in _configs_by_name().values():
        parent_job_id = cfg.get("parent_job_id", "")
        assert parent_job_id in ("", PLACEHOLDER_PARENT_JOB_ID)


def test_find_unresolved_placeholders_walks_nested_structures() -> None:
    config = {
        "document_ids": [PLACEHOLDER_DOCUMENT_ID],
        "model_configs": [{"model": PLACEHOLDER_MODEL, "provider": PLACEHOLDER_MODEL}],
        "num_records": 100,
    }
    assert find_unresolved_placeholders(config) == sorted(
        {PLACEHOLDER_DOCUMENT_ID, PLACEHOLDER_MODEL}
    )
    assert find_unresolved_placeholders({"document_ids": ["abc123"]}) == []


def test_prompt_ref_roundtrip() -> None:
    ref = make_prompt_ref("prm_abc123")
    assert ref == "<prompt:prm_abc123>"
    assert is_prompt_ref(ref)
    assert prompt_ref_id(ref) == "prm_abc123"
    # Only a full-value token is a ref — not embedded text, not a non-prm id.
    assert not is_prompt_ref("here is <prompt:prm_abc123> inline")
    assert prompt_ref_id("here is <prompt:prm_abc123> inline") is None
    assert not is_prompt_ref("<prompt:other>")
    assert not is_prompt_ref("plain text")
    assert prompt_ref_id(123) is None


def test_find_prompt_refs_walks_config() -> None:
    config = {
        "columns": [
            {"name": "ticket", "system_prompt": make_prompt_ref("prm_a")},
            {"name": "assess", "system_prompt": make_prompt_ref("prm_b")},
            {"name": "plain", "system_prompt": "inline text"},
        ],
        "processors": [
            {"template": {"messages": [{"role": "system", "content": make_prompt_ref("prm_a")}]}}
        ],
    }
    assert find_prompt_refs(config) == ["prm_a", "prm_b"]
    assert find_prompt_refs({"columns": [{"system_prompt": "inline"}]}) == []


def test_resolve_prompt_refs_is_pure_and_substitutes() -> None:
    config = {
        "columns": [{"name": "t", "system_prompt": make_prompt_ref("prm_a")}],
        "num_records": 10,
    }
    resolved = resolve_prompt_refs(config, {"prm_a": "FULL TEXT"}.get)
    assert resolved["columns"][0]["system_prompt"] == "FULL TEXT"
    assert resolved["num_records"] == 10
    # Source config is not mutated — the persisted ref-form survives.
    assert config["columns"][0]["system_prompt"] == make_prompt_ref("prm_a")


def test_resolve_prompt_refs_leaves_unknown_refs() -> None:
    config = {"columns": [{"system_prompt": make_prompt_ref("prm_missing")}]}
    resolved = resolve_prompt_refs(config, {}.get)
    # An unresolved ref is left verbatim for the caller to reject (fail-closed).
    assert resolved["columns"][0]["system_prompt"] == make_prompt_ref("prm_missing")


@pytest.mark.asyncio
async def test_create_job_rejects_unresolved_placeholders() -> None:
    # The rejection happens before the repository is touched, so a dummy repo is fine.
    with pytest.raises(InvalidJobStateError, match="placeholder"):
        await create_job(
            object(),  # type: ignore[arg-type]
            job_type=JobType.sdg,
            config={"model_configs": [{"model": PLACEHOLDER_MODEL}]},
        )
