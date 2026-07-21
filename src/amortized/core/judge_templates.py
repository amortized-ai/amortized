"""Judge template loading — reads YAML configs from templates/eval/."""

import logging
from pathlib import Path
from typing import Any

import yaml

import amortized.config as config_mod

logger = logging.getLogger("amortized.core.judge_templates")

_SOURCE_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "eval"


def _get_templates_dir() -> Path:
    if config_mod.settings.recipes_dir:
        configured = config_mod.settings.recipes_dir / "templates" / "eval"
        if configured.is_dir():
            return configured
    return _SOURCE_TEMPLATES_DIR


def load_judge_template(name: str) -> dict[str, Any]:
    """Load a judge template YAML by name.

    Args:
        name: Template name like "safety" or "code-quality"
    """
    path = _get_templates_dir() / f"{name}.yaml"
    if not path.exists():
        available = list_judge_templates()
        raise FileNotFoundError(f"Judge template '{name}' not found. Available: {available}")
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def list_judge_templates() -> list[str]:
    """List available judge template names."""
    templates_dir = _get_templates_dir()
    if not templates_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(templates_dir)).removesuffix(".yaml")
        for p in templates_dir.rglob("*.yaml")
    )
