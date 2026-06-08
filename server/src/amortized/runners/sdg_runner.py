"""SDG job runner — invoked as a subprocess by the worker.

Receives job config as a JSON string argument. Uses asynth for
synthesis. Falls back to a simulated run if asynth is not installed.
"""

import json
import os
import sys
import time
from typing import Any


def _deserialize_strategy_params(raw: dict[str, Any]) -> Any:
    """Convert nested dicts to proper asynth dataclass instances."""
    from asynth.configs.params.synthesis_params import (
        AttributeCombination,
        DatasetSource,
        DocumentSegmentationParams,
        DocumentSource,
        ExampleSource,
        GeneralSynthesisParams,
        GeneratedAttribute,
        GeneratedAttributePostprocessingParams,
        MultiTurnAttribute,
        SampledAttribute,
        SampledAttributeValue,
        TextConversation,
        TextMessage,
        TransformationStrategy,
        TransformedAttribute,
    )

    def _to_dataclass(cls: type, val: Any) -> Any:
        if isinstance(val, cls):
            return val
        if isinstance(val, dict):
            return cls(**val)
        return val

    def _to_list(cls: type, items: Any) -> list[Any] | None:
        if items is None:
            return None
        return [_to_dataclass(cls, item) for item in items]

    coerced = dict(raw)

    if "sampled_attributes" in coerced and coerced["sampled_attributes"] is not None:
        attrs = []
        for sa in coerced["sampled_attributes"]:
            if isinstance(sa, dict):
                sa = dict(sa)
                if "possible_values" in sa and sa["possible_values"] is not None:
                    sa["possible_values"] = _to_list(SampledAttributeValue, sa["possible_values"])
                sa = SampledAttribute(**sa)
            attrs.append(sa)
        coerced["sampled_attributes"] = attrs

    if "generated_attributes" in coerced and coerced["generated_attributes"] is not None:
        attrs = []
        for ga in coerced["generated_attributes"]:
            if isinstance(ga, dict):
                ga = dict(ga)
                if "instruction_messages" in ga and ga["instruction_messages"] is not None:
                    ga["instruction_messages"] = _to_list(TextMessage, ga["instruction_messages"])
                if "postprocessing_params" in ga and isinstance(ga["postprocessing_params"], dict):
                    ga["postprocessing_params"] = GeneratedAttributePostprocessingParams(
                        **ga["postprocessing_params"]
                    )
                ga = GeneratedAttribute(**ga)
            attrs.append(ga)
        coerced["generated_attributes"] = attrs

    coerced["multiturn_attributes"] = _to_list(
        MultiTurnAttribute, coerced.get("multiturn_attributes")
    )

    if "transformed_attributes" in coerced and coerced["transformed_attributes"] is not None:
        attrs = []
        for ta in coerced["transformed_attributes"]:
            if isinstance(ta, dict):
                ta = dict(ta)
                if "transformation_strategy" in ta and isinstance(
                    ta["transformation_strategy"], dict
                ):
                    strat = dict(ta["transformation_strategy"])
                    if "chat_transform" in strat and isinstance(strat["chat_transform"], dict):
                        ct = dict(strat["chat_transform"])
                        if "messages" in ct and ct["messages"] is not None:
                            ct["messages"] = _to_list(TextMessage, ct["messages"])
                        strat["chat_transform"] = TextConversation(**ct)
                    ta["transformation_strategy"] = TransformationStrategy(**strat)
                ta = TransformedAttribute(**ta)
            attrs.append(ta)
        coerced["transformed_attributes"] = attrs

    if "input_data" in coerced and coerced["input_data"] is not None:
        coerced["input_data"] = _to_list(DatasetSource, coerced["input_data"])

    if "input_documents" in coerced and coerced["input_documents"] is not None:
        docs = []
        for ds in coerced["input_documents"]:
            if isinstance(ds, dict):
                ds = dict(ds)
                if "segmentation_params" in ds and isinstance(ds["segmentation_params"], dict):
                    ds["segmentation_params"] = DocumentSegmentationParams(
                        **ds["segmentation_params"]
                    )
                ds = DocumentSource(**ds)
            docs.append(ds)
        coerced["input_documents"] = docs

    coerced["input_examples"] = _to_list(ExampleSource, coerced.get("input_examples"))
    coerced["combination_sampling"] = _to_list(
        AttributeCombination, coerced.get("combination_sampling")
    )

    # Remove None entries for fields not present in the raw dict
    coerced = {k: v for k, v in coerced.items() if v is not None or k in raw}

    return GeneralSynthesisParams(**coerced)


def run_sdg(config: dict[str, Any]) -> None:
    """Execute a synthetic data generation job."""
    output_dir = str(config.get("output_dir", "./sdg_output"))
    os.makedirs(output_dir, exist_ok=True)

    try:
        from asynth import LiteLLMInferenceConfig, SynthesisConfig, synthesize
    except ImportError:
        _simulate_sdg(config, output_dir)
        return

    inference_config = LiteLLMInferenceConfig(
        model=config["model"],
        api_base=config.get("api_base"),
        api_key=config.get("api_key"),
        temperature=config.get("temperature", 0.7),
        max_concurrency=config.get("max_concurrent", 16),
        max_tokens=config.get("max_tokens"),
        top_p=config.get("top_p"),
        seed=config.get("seed"),
        num_retries=config.get("num_retries", 3),
    )

    raw_strategy = config.get("strategy_params")
    if raw_strategy and isinstance(raw_strategy, dict):
        merged = dict(raw_strategy)
        if config.get("input_data") and "input_data" not in merged:
            merged["input_data"] = config["input_data"]
        if config.get("input_documents") and "input_documents" not in merged:
            merged["input_documents"] = config["input_documents"]
        strategy_params = _deserialize_strategy_params(merged)
    else:
        kwargs: dict[str, Any] = {}
        if config.get("input_data"):
            kwargs["input_data"] = config["input_data"]
        if config.get("input_documents"):
            kwargs["input_documents"] = config["input_documents"]
        strategy_params = (
            _deserialize_strategy_params(kwargs) if kwargs else _deserialize_strategy_params({})
        )

    output_path = os.path.join(output_dir, "generated_data.jsonl")

    synth_config = SynthesisConfig(
        num_samples=config.get("num_samples", 100),
        output_path=output_path,
        inference_config=inference_config,
        strategy_params=strategy_params,
    )

    results = synthesize(synth_config)

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(
            {
                "total_completed": len(results),
                "total_requested": config.get("num_samples", 100),
                "status": "completed",
            },
            f,
            indent=2,
        )


def _simulate_sdg(config: dict[str, Any], output_dir: str) -> None:
    """Simulate SDG when asynth is not installed."""
    total_rows = config.get("num_samples", 100)
    output_path = os.path.join(output_dir, "generated_data.jsonl")

    with open(output_path, "w") as f:
        for i in range(total_rows):
            row = {
                "instruction": f"Sample instruction {i}",
                "input": f"Sample input {i}",
                "output": f"Sample output {i}",
            }
            f.write(json.dumps(row) + "\n")
            time.sleep(0.01)

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(
            {
                "total_completed": total_rows,
                "total_requested": config.get("num_samples", 100),
                "status": "completed",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m amortized.runners.sdg_runner '<config_json>'")
        sys.exit(1)

    config_data = json.loads(sys.argv[1])
    run_sdg(config_data)
