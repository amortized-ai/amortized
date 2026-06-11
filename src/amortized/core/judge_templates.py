"""Judge template loading — reads YAML configs from templates/eval/."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("amortized.core.judge_templates")

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "eval"


def load_judge_template(name: str) -> dict[str, Any]:
    """Load a judge template YAML by name.

    Args:
        name: Template name like "safety" or "code-quality"
    """
    path = _TEMPLATES_DIR / f"{name}.yaml"
    if not path.exists():
        available = list_judge_templates()
        raise FileNotFoundError(f"Judge template '{name}' not found. Available: {available}")
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def list_judge_templates() -> list[str]:
    """List available judge template names."""
    if not _TEMPLATES_DIR.is_dir():
        return []
    return sorted(
        str(p.relative_to(_TEMPLATES_DIR)).removesuffix(".yaml")
        for p in _TEMPLATES_DIR.rglob("*.yaml")
    )
