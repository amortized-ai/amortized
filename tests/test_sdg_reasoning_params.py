"""Reasoning-model teacher params are normalized in the generated SDG config."""

import yaml

from amortized.jobs import sdg


async def test_reasoning_teacher_drops_temperature() -> None:
    config = {
        "model_configs": [
            {
                "alias": "teacher",
                "model": "gpt-5.6-sol",
                "inference_parameters": {"temperature": 0.7, "max_tokens": 512},
            },
            {
                "alias": "standard",
                "model": "gpt-4o",
                "inference_parameters": {"temperature": 0.7, "max_tokens": 512},
            },
        ],
        "columns": [],
    }
    config_files: dict[str, str] = {}
    await sdg.build({"id": "test-job", "config": config}, config, config_files)

    dd = yaml.safe_load(config_files["config.yaml"])
    model_configs = dd["data_designer"]["model_configs"]
    teacher = next(mc for mc in model_configs if mc["model"] == "gpt-5.6-sol")
    standard = next(mc for mc in model_configs if mc["model"] == "gpt-4o")

    # Reasoning teacher: non-default temperature dropped (litellm rejects it);
    # max_tokens kept — litellm maps it to max_completion_tokens downstream.
    assert "temperature" not in teacher["inference_parameters"]
    assert teacher["inference_parameters"]["max_tokens"] == 512
    # Non-reasoning model: params untouched.
    assert standard["inference_parameters"]["temperature"] == 0.7
    assert standard["inference_parameters"]["max_tokens"] == 512
