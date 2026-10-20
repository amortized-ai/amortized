"""`config_json` single-string escape hatch on the validate request models.

GLM-5.3-Flash's native tool-call format (<arg_key>/<arg_value>) is mis-parsed for
configs ≳5KB: the nested arguments get merged into the first field, so validate_sdg_job
never parses and the agent spins (observed: 27 failed attempts, then it fabricated a
completion off a stale job). A SINGLE string argument round-trips intact, so the agent
can pass the entire config as one `config_json` string. When set, config_json fully
replaces the flat arguments. Applies to SDG, training, and eval (same provider bug).
"""

import json
from typing import Any

import pytest

from amortized.models import EvalJobRequest, SDGJobRequest, TrainingJobRequest

_SDG: dict[str, Any] = {
    "columns": [
        {
            "column_type": "sampler",
            "name": "domain",
            "sampler_type": "category",
            "params": {"values": ["A", "B"]},
        },
        {
            "column_type": "llm-text",
            "name": "rfe",
            "model_alias": "teacher",
            "prompt": "Make a ticket about {{ domain }}.",
        },
    ],
    "model_configs": [{"alias": "teacher", "model": "openai/gpt-oss-120b", "provider": "gateway"}],
    "processors": [
        {
            "processor_type": "schema_transform",
            "name": "sft",
            "template": {
                "messages": [
                    {"role": "user", "content": "{{ rfe }}"},
                    {"role": "assistant", "content": "ok"},
                ]
            },
        }
    ],
    "num_records": 10,
    "mode": "preview",
    "topic": "RFE",
}

_TRAIN: dict[str, Any] = {
    "algorithm": "sft",
    "model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "parent_job_id": "sdg-abc",
}

_EVAL: dict[str, Any] = {
    "rubric": [{"name": "acc", "description": "is it correct"}],
    "endpoint": {"base_url": "http://e", "model": "m"},
    "judge": {"base_url": "http://j", "model": "gpt-oss"},
    "parent_job_id": "sdg-xyz",
}


@pytest.mark.parametrize(
    ("model", "cfg"),
    [(SDGJobRequest, _SDG), (TrainingJobRequest, _TRAIN), (EvalJobRequest, _EVAL)],
)
def test_config_json_equals_flat(model: Any, cfg: dict[str, Any]) -> None:
    """Passing the whole config as a single config_json string yields the identical model
    as passing the fields flat — for every validate request type."""
    flat = model(**cfg)
    via = model(config_json=json.dumps(cfg))
    assert flat.model_dump(exclude_none=True) == via.model_dump(exclude_none=True)
    # the escape-hatch field never leaks into the stored config
    assert via.config_json is None
    assert "config_json" not in via.model_dump(exclude_none=True)


def test_config_json_overrides_stray_flat_args() -> None:
    """A weak model that sends config_json AND leftover flat args: config_json wins wholesale,
    so the flat args can't half-populate a corrupted config."""
    m = SDGJobRequest(config_json=json.dumps(_SDG), num_records=999, topic="WRONG")
    assert m.num_records == 10
    assert m.topic == "RFE"


def test_config_json_default_absent() -> None:
    """Normal flat calls are unaffected: config_json defaults to None and is dropped."""
    m = SDGJobRequest(**_SDG)
    assert m.config_json is None
    assert "config_json" not in m.model_dump(exclude_none=True)


def test_bad_config_json_errors_clearly() -> None:
    """A non-JSON / non-object config_json raises a message that names config_json, so the
    agent fixes the call instead of chasing a phantom (the real failure mode)."""
    with pytest.raises(ValueError, match="config_json"):
        SDGJobRequest(config_json="not json {")
    with pytest.raises(ValueError, match="config_json"):
        SDGJobRequest(config_json="[1, 2, 3]")  # valid JSON, but not an object


def test_config_json_still_runs_normal_validation() -> None:
    """Expansion feeds the config into the model's own validators — a config that is invalid
    on its own merits still fails (the escape hatch doesn't bypass validation)."""
    bad = {**_SDG, "columns": [{"column_type": "llm-text", "name": "x", "model_alias": "nope"}]}
    with pytest.raises(ValueError, match=r"model_configs|model_alias"):
        SDGJobRequest(config_json=json.dumps(bad))


@pytest.mark.parametrize(
    "model", [SDGJobRequest, TrainingJobRequest, EvalJobRequest]
)
def test_no_required_fields_in_schema(model: Any) -> None:
    """The MCP layer validates a tool call against the model's JSON schema BEFORE the
    config_json expansion runs. If any flat field is schema-required, a config_json-only
    call is rejected there with e.g. "'columns' is a required property" and never reaches
    expansion (the live bug the first deploy hit). So no field may be schema-required."""
    assert model.model_json_schema().get("required", []) == []


@pytest.mark.parametrize(
    ("model", "cfg"),
    [(SDGJobRequest, _SDG), (TrainingJobRequest, _TRAIN), (EvalJobRequest, _EVAL)],
)
def test_config_json_only_without_flat_fields(model: Any, cfg: dict[str, Any]) -> None:
    """A config_json-only call (no flat columns/algorithm/etc.) validates — this is exactly
    the shape the agent sends and the shape the MCP schema must accept."""
    m = model(config_json=json.dumps(cfg))
    assert m.model_dump(exclude_none=True) == model(**cfg).model_dump(exclude_none=True)


def test_empty_call_still_rejected() -> None:
    """Dropping schema-required does not make an empty config valid: the after-expansion
    checks still fire. SDG's points the agent at config_json; training is rejected by its
    inherited model-source check first (still rejected, just a different message)."""
    with pytest.raises(ValueError, match="config_json"):
        SDGJobRequest()
    with pytest.raises(ValueError):
        TrainingJobRequest()
