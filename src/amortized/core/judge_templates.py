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


def translate_template_to_judge_config(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate raw template YAML into a JudgeConfig-compatible dict.

    Returns (judge_config_dict, inference_defaults) where judge_config_dict
    is suitable for JudgeConfig.from_dict() and inference_defaults contains
    model/temperature from the template (to be overridden by request params).
    """
    if "rule_judge_params" in data:
        return data, {}

    cfg = data.get("config", data)
    judge_section = cfg.get("judge", {})

    judge_params: dict[str, Any] = {}
    if judge_section.get("prompt"):
        judge_params["prompt_template"] = judge_section["prompt"]
    for key in ("system_instruction", "judgment_type", "response_format", "include_explanation"):
        if cfg.get(key) is not None:
            judge_params[key] = cfg[key]

    inference_defaults: dict[str, Any] = {}
    if judge_section.get("model"):
        inference_defaults["model"] = judge_section["model"]
    if cfg.get("temperature") is not None:
        inference_defaults["temperature"] = cfg["temperature"]

    return {"judge_params": judge_params}, inference_defaults


def list_judge_templates() -> list[str]:
    """List available judge template names."""
    if not _TEMPLATES_DIR.is_dir():
        return []
    return sorted(
        str(p.relative_to(_TEMPLATES_DIR)).removesuffix(".yaml")
        for p in _TEMPLATES_DIR.rglob("*.yaml")
    )
