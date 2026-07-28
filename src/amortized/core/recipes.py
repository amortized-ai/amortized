"""Recipe loading and composition — zero HTTP imports."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

import amortized.config as _config_mod

logger = logging.getLogger("amortized.core.recipes")

_RECIPES_DIR = Path(__file__).resolve().parent.parent.parent.parent


class RecipeNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Recipe not found: {name}")


class CircularRecipeError(Exception):
    def __init__(self, chain: list[str], name: str) -> None:
        self.chain = chain
        self.name = name
        super().__init__(f"Circular recipe extends: {' -> '.join(chain)} -> {name}")


class ProtectedRecipeError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Cannot delete built-in recipe: {name}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_recipes_dir(recipes_dir: Path | None = None) -> Path:
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

    base_dir = get_recipes_dir(recipes_dir)
    path = base_dir / f"{name}.yaml"
    if not path.is_file():
        raise RecipeNotFoundError(name)

    raw: dict[str, Any] = yaml.safe_load(path.read_text())

    if "extends" in raw:
        parent_name: str = raw.pop("extends")
        parent = load_recipe(parent_name, recipes_dir=base_dir, _chain=_chain)
        raw = _deep_merge(parent, raw)

    return raw


def list_recipes(*, recipes_dir: Path | None = None) -> list[dict[str, Any]]:
    base_dir = get_recipes_dir(recipes_dir)
    results: list[dict[str, Any]] = []

    scan_dirs = [base_dir / "templates"]
    if recipes_dir is not None:
        scan_dirs = [base_dir]

    for scan_dir in scan_dirs:
        if not scan_dir.is_dir():
            continue
        for path in sorted(scan_dir.rglob("*.yaml")):
            rel = path.relative_to(base_dir).with_suffix("")
            name = str(rel)
            try:
                raw: dict[str, Any] = yaml.safe_load(path.read_text())
            except Exception:
                logger.warning("Skipping invalid recipe: %s", name)
                continue
            results.append(
                {
                    "name": name,
                    "description": raw.get("description", ""),
                    "type": raw.get("type", ""),
                }
            )
    return results


_RECIPE_META_KEYS = frozenset({"type", "description", "extends", "name"})


def apply_overrides(recipe: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(recipe)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        if keys[0] in _RECIPE_META_KEYS:
            continue
        if keys[0] != "config":
            if not (value or value == 0 or value is False):
                continue
            keys = ["config", *keys]
        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    return result


def flatten_recipe_to_config(recipe: dict[str, Any]) -> dict[str, Any]:
    """Merge top-level recipe overrides into the config sub-dict."""
    config: dict[str, Any] = dict(recipe.get("config", {}))
    _meta_keys = frozenset({"type", "description", "extends", "config", "name"})
    for key, value in recipe.items():
        if key not in _meta_keys and (value or value == 0 or value is False):
            config[key] = value
    if "teacher_model" in config and "model" not in config:
        config["model"] = config.pop("teacher_model")
    return config


def delete_recipe(name: str, *, recipes_dir: Path | None = None) -> None:
    base_dir = get_recipes_dir(recipes_dir)

    if ".." in name.split("/"):
        raise ValueError("Invalid recipe name")

    path = base_dir / f"{name}.yaml"
    resolved = path.resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise ValueError("Invalid recipe name")

    if not name.startswith("templates/custom/"):
        raise ProtectedRecipeError(name)

    if not path.is_file():
        raise RecipeNotFoundError(name)

    path.unlink()
    logger.info("Deleted recipe %s", name)
