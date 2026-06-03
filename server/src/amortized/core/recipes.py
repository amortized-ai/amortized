"""Recipe loading and composition — zero HTTP imports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("amortized.core.recipes")

_RECIPES_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "recipes"


class RecipeNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Recipe not found: {name}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_recipe(name: str, *, recipes_dir: Path | None = None) -> dict[str, Any]:
    base_dir = recipes_dir or _RECIPES_DIR
    path = base_dir / f"{name}.yaml"
    if not path.is_file():
        raise RecipeNotFoundError(name)

    raw: dict[str, Any] = yaml.safe_load(path.read_text())

    if "extends" in raw:
        parent_name: str = raw.pop("extends")
        parent = load_recipe(parent_name, recipes_dir=base_dir)
        raw = _deep_merge(parent, raw)

    return raw


def list_recipes(*, recipes_dir: Path | None = None) -> list[dict[str, Any]]:
    base_dir = recipes_dir or _RECIPES_DIR
    results: list[dict[str, Any]] = []
    if not base_dir.is_dir():
        return results

    for path in sorted(base_dir.rglob("*.yaml")):
        rel = path.relative_to(base_dir).with_suffix("")
        name = str(rel)
        try:
            raw: dict[str, Any] = yaml.safe_load(path.read_text())
        except Exception:
            logger.warning("Skipping invalid recipe: %s", name)
            continue
        results.append({
            "name": name,
            "description": raw.get("description", ""),
            "type": raw.get("type", ""),
        })
    return results


def apply_overrides(recipe: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(recipe)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    return result
