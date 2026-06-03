"""Job type registry — maps type strings to schemas and validation. Zero HTTP imports."""

from __future__ import annotations

import json
import logging
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


class UnknownJobTypeError(Exception):
    def __init__(self, job_type: str) -> None:
        self.job_type = job_type
        super().__init__(f"Unknown job type: {job_type}")
