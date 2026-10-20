"""Starter templates must use the canonical placeholder tokens.

Guards against spelling drift (e.g. the historical "<from list_models>" vs
"<from-list_models>") between the templates and core.placeholders.
"""

from __future__ import annotations

from typing import Any

from amortized.core.placeholders import (
    PLACEHOLDER_DOCUMENT_ID,
    PLACEHOLDER_MODEL,
    PLACEHOLDER_PARENT_JOB_ID,
)
from amortized.core.recipes import list_starter_templates


def _configs_by_name() -> dict[str, dict[str, Any]]:
    return {t["name"]: t["config"] for t in list_starter_templates()}


def test_knowledge_ingestion_uses_canonical_tokens() -> None:
    cfg = _configs_by_name()["Knowledge Ingestion"]
    assert cfg["document_ids"] == [PLACEHOLDER_DOCUMENT_ID]
    assert cfg["model_configs"][0]["model"] == PLACEHOLDER_MODEL
    assert cfg["model_configs"][0]["provider"] == PLACEHOLDER_MODEL


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
