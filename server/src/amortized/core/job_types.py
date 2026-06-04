"""Job type registry — maps type strings to schemas and validation. Zero HTTP imports."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import jsonschema

logger = logging.getLogger("amortized.core.job_types")

_SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"

_REGISTRY: dict[str, dict[str, Any]] = {
    "training": {
        "description": "LoRA SFT fine-tuning job",
        "schema_file": "training.json",
    },
    "sdg": {
        "description": "Synthetic data generation job",
        "schema_file": "sdg.json",
    },
    "inference": {
        "description": "vLLM batch inference job",
        "schema_file": "inference.json",
    },
    "eval": {
        "description": "LLM-as-judge evaluation job",
        "schema_file": "eval.json",
    },
}

_schema_cache: dict[str, dict[str, Any]] = {}


def _load_schema(schema_file: str) -> dict[str, Any]:
    if schema_file not in _schema_cache:
        path = _SCHEMAS_DIR / schema_file
        _schema_cache[schema_file] = json.loads(path.read_text())
    return _schema_cache[schema_file]


def get_schema(job_type: str) -> dict[str, Any]:
    entry = _REGISTRY.get(job_type)
    if entry is None:
        raise UnknownJobTypeError(job_type)
    return _load_schema(entry["schema_file"])


def validate_config(job_type: str, config: dict[str, Any]) -> list[str]:
    schema = get_schema(job_type)
    validator = jsonschema.Draft202012Validator(schema)
    return [e.message for e in validator.iter_errors(config)]


def list_job_types() -> list[dict[str, str]]:
    return [
        {"type": job_type, "description": entry["description"]}
        for job_type, entry in _REGISTRY.items()
    ]


# --- Semantic (pre-flight) validators per job type ---

_TYPICAL_LORA_R = {4, 8, 16, 32, 64, 128}


async def _validate_training(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    lora_r = config.get("lora_r")
    if lora_r is not None and lora_r not in _TYPICAL_LORA_R:
        errors.append(f"lora_r={lora_r} is unusual; typical values: 4, 8, 16, 32, 64, 128")
    lora_alpha = config.get("lora_alpha")
    if lora_alpha is not None and lora_r is not None and lora_alpha < lora_r:
        errors.append(f"lora_alpha={lora_alpha} < lora_r={lora_r}; alpha is typically >= rank")
    return errors


async def _validate_sdg(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    model = config.get("model", "")
    if model and "/" not in model:
        errors.append(f"model='{model}' should use provider/name format (e.g. openai/gpt-4o)")
    return errors


async def _validate_inference(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tp = config.get("tensor_parallel_size")
    if tp is not None and (tp & (tp - 1)) != 0:
        errors.append(f"tensor_parallel_size={tp} must be a power of 2")
    return errors


async def _validate_eval(config: dict[str, Any]) -> list[str]:
    return []


_SemanticValidator = Callable[[dict[str, Any]], Coroutine[Any, Any, list[str]]]

_SEMANTIC_VALIDATORS: dict[str, _SemanticValidator] = {
    "training": _validate_training,
    "sdg": _validate_sdg,
    "inference": _validate_inference,
    "eval": _validate_eval,
}


async def validate_semantic(job_type: str, config: dict[str, Any]) -> list[str]:
    """Run semantic pre-flight checks for a given job type."""
    validator = _SEMANTIC_VALIDATORS.get(job_type)
    if validator is None:
        return []
    return await validator(config)


class UnknownJobTypeError(Exception):
    def __init__(self, job_type: str) -> None:
        self.job_type = job_type
        super().__init__(f"Unknown job type: {job_type}")
