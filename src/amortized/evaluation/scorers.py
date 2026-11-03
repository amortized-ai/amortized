"""Implemented deterministic scorers with explicit invalid-output accounting."""

import json
import math
from typing import Any

from jsonschema import Draft202012Validator

from amortized.evaluation.spec import DeterministicScorer


def _extract(value: str, field: str) -> Any:
    if not field:
        return value
    parsed = json.loads(value)
    for component in field.split("."):
        parsed = parsed[int(component)] if isinstance(parsed, list) else parsed[component]
    return parsed


def score(scorer: DeterministicScorer, output: str, reference: str) -> float:
    candidate = _extract(output, scorer.response_field)
    if scorer.id == "json_schema":
        value = json.loads(candidate) if isinstance(candidate, str) else candidate
        return float(Draft202012Validator(scorer.schema_).is_valid(value))
    expected = _extract(reference, scorer.reference_field)
    if scorer.id == "exact_match":
        return float(candidate == expected)
    if isinstance(candidate, bool) or isinstance(expected, bool):
        raise ValueError("Boolean values are not numeric scores")
    actual, target = float(candidate), float(expected)
    if not math.isfinite(actual) or not math.isfinite(target):
        raise ValueError("Numeric scoring requires finite values")
    error = abs(actual - target)
    if not math.isfinite(error):
        raise ValueError("Numeric error overflowed")
    return error if scorer.id == "numeric_mae" else float(error <= scorer.tolerance)


def metric_metadata(scorer: DeterministicScorer) -> dict[str, Any]:
    numeric = scorer.id == "numeric_mae"
    return {
        "scorer_id": scorer.id,
        "scorer_version": scorer.version,
        "unit": "reference units" if numeric else "fraction",
        "direction": "lower" if numeric else "higher",
        "range": None if numeric else [0, 1],
        "invalid_output_policy": "exclude_with_count" if numeric else "zero",
    }


def apply_scorers(
    rows: list[dict[str, Any]], scorers: list[DeterministicScorer], results: dict[str, Any]
) -> None:
    results.setdefault("scores", {})
    results.setdefault("scores_n", {})
    results.setdefault("metric_metadata", {})
    for scorer in scorers:
        values: list[float] = []
        invalid = 0
        for row in rows:
            value: float | None
            try:
                if row.get("error"):
                    raise ValueError("Generation failed")
                value = score(scorer, row["output"], row["reference"])
            except (ValueError, TypeError, KeyError, IndexError):
                invalid += 1
                value = None if scorer.id == "numeric_mae" else 0.0
            row.setdefault("scores", {})[scorer.name] = value
            if value is not None:
                values.append(value)
        results["scores"][scorer.name] = (
            math.fsum(value / len(values) for value in values) if values else None
        )
        results["scores_n"][scorer.name] = len(values)
        results["metric_metadata"][scorer.name] = {
            **metric_metadata(scorer),
            "num_samples": len(values),
            "num_invalid": invalid,
            "num_total": len(rows),
        }
