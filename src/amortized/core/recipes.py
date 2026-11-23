"""Starter template loading from agents/*/skills reference payloads."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import amortized.config as _config_mod

logger = logging.getLogger("amortized.core.recipes")

_RECIPES_DIR = Path(__file__).resolve().parent.parent.parent.parent


def _get_base_dir() -> Path:
    if _config_mod.settings.recipes_dir is not None:
        return _config_mod.settings.recipes_dir
    return _RECIPES_DIR


def _template(config: dict[str, Any], job_type: str, use_case: str) -> dict[str, Any]:
    meta = config.pop("_meta", {})
    return {
        "name": meta.get("name", use_case),
        "type": job_type,
        "use_case": use_case,
        "description": meta.get("description", ""),
        # Per-recipe required fields (e.g. Knowledge Ingestion needs document_ids
        # — optional on the shared SDG schema, so declared on the recipe here).
        "required": meta.get("required", []),
        "config": config,
    }


def list_starter_templates() -> list[dict[str, Any]]:
    agents_dir = _get_base_dir() / "agents"
    if not agents_dir.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for agent_dir in sorted(agents_dir.iterdir()):
        skills_dir = agent_dir / "skills"
        if not skills_dir.is_dir():
            continue
        job_type = agent_dir.name
        for path in sorted(skills_dir.rglob("reference-payload.json")):
            rel = path.relative_to(skills_dir)
            use_case = "-".join(rel.parent.parts)
            if not use_case:
                continue

            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                logger.warning("Skipping invalid template: %s", path)
                continue

            # One recipe per file, or a list of recipes for a skill with several use cases;
            # each list entry's `_meta.id` is appended to the folder's use case.
            if isinstance(payload, list):
                for config in payload:
                    recipe_id = (
                        config.get("_meta", {}).get("id") if isinstance(config, dict) else None
                    )
                    if not recipe_id:
                        logger.warning("Skipping template without _meta.id in %s", path)
                        continue
                    results.append(_template(config, job_type, f"{use_case}-{recipe_id}"))
            elif isinstance(payload, dict):
                results.append(_template(payload, job_type, use_case))
            else:
                logger.warning("Skipping invalid template: %s", path)
    return results
