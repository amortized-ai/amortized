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
                config = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping invalid template: %s", path)
                continue

            meta = config.pop("_meta", {})
            name = meta.get("name", use_case)
            description = meta.get("description", "")

            results.append(
                {
                    "name": name,
                    "type": job_type,
                    "use_case": use_case,
                    "description": description,
                    "config": config,
                }
            )
    return results
