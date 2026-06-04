"""Recipe loading and composition — zero HTTP imports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

import amortized.config as _config_mod

logger = logging.getLogger("amortized.core.recipes")

_RECIPES_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "recipes"


class RecipeNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Recipe not found: {name}")


class CircularRecipeError(Exception):
    def __init__(self, chain: list[str], name: str) -> None:
        self.chain = chain
        self.name = name
        super().__init__(f"Circular recipe extends: {' -> '.join(chain)} -> {name}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_recipes_dir(recipes_dir: Path | None = None) -> Path:
    if recipes_dir is not None:
        return recipes_dir
    if _config_mod.settings.recipes_dir is not None:
        return _config_mod.settings.recipes_dir
    return _RECIPES_DIR


def load_recipe(
    name: str,
    *,
    recipes_dir: Path | None = None,
    _chain: set[str] | None = None,
) -> dict[str, Any]:
    if _chain is None:
        _chain = set()
    if name in _chain:
        raise CircularRecipeError(list(_chain), name)
    _chain.add(name)

    base_dir = _get_recipes_dir(recipes_dir)
    path = base_dir / f"{name}.yaml"
    if not path.is_file():
        raise RecipeNotFoundError(name)

    raw: dict[str, Any] = yaml.safe_load(path.read_text())

    if "extends" in raw:
        parent_name: str = raw.pop("extends")
        parent = load_recipe(parent_name, recipes_dir=base_dir, _chain=_chain)
        raw = _deep_merge(parent, raw)

    job_type = raw.get("type")
    if job_type:
        _validate_merged_config(job_type, raw.get("config", {}))

    return raw


def _validate_merged_config(job_type: str, config: dict[str, Any]) -> None:
    """Validate merged recipe config against the job type's JSON Schema."""
    from amortized.core.job_types import UnknownJobTypeError, validate_config

    try:
        errors = validate_config(job_type, config)
    except UnknownJobTypeError:
        return
    if errors:
        logger.warning(
            "Recipe config validation warnings for type '%s': %s",
            job_type,
            "; ".join(errors),
        )


def list_recipes(*, recipes_dir: Path | None = None) -> list[dict[str, Any]]:
    base_dir = _get_recipes_dir(recipes_dir)
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
