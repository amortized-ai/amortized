"""Tests for the SDG runner — simulation fallback and config mapping."""

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

from amortized.runners.sdg_runner import _deserialize_strategy_params, run_sdg


class TestSimulationFallback:
    """When asynth is not installed, the runner falls back to simulation."""

    def _run_sdg_simulated(self, config: dict[str, Any]) -> None:
        """Run SDG with asynth import blocked so the fallback path is used."""
        with patch.dict(
            sys.modules,
            {
                "asynth": None,
                "asynth.configs": None,
                "asynth.configs.params": None,
                "asynth.configs.params.synthesis_params": None,
            },
        ):
            from importlib import reload

            import amortized.runners.sdg_runner as mod

            reload(mod)
            mod.run_sdg(config)

    def test_simulation_produces_output(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "output")
        config = {"output_dir": output_dir, "model": "openai/gpt-4o", "num_samples": 10}
        self._run_sdg_simulated(config)

        output_path = os.path.join(output_dir, "generated_data.jsonl")
        assert os.path.exists(output_path)

        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 10

        first_row = json.loads(lines[0])
        assert "instruction" in first_row
        assert "output" in first_row

    def test_simulation_writes_stats(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "output")
        config = {"output_dir": output_dir, "model": "openai/gpt-4o", "num_samples": 10}
        self._run_sdg_simulated(config)

        stats_path = os.path.join(output_dir, "stats.json")
        assert os.path.exists(stats_path)

        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["total_completed"] == 10
        assert stats["status"] == "completed"

    def test_default_output_dir(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        config = {"model": "openai/gpt-4o"}
        self._run_sdg_simulated(config)
        assert os.path.exists("./sdg_output/generated_data.jsonl")


class TestConfigMapping:
    """Test that config dict is correctly mapped to asynth objects."""

    def test_basic_config_mapping(self, tmp_path: Any) -> None:
        mock_synthesize = MagicMock(return_value=[{"q": "test", "a": "answer"}])
        mock_inference_cls = MagicMock()
        mock_synth_config_cls = MagicMock()
        mock_params_cls = MagicMock()

        mock_asynth = MagicMock()
        mock_asynth.SynthesisConfig = mock_synth_config_cls
        mock_asynth.LiteLLMInferenceConfig = mock_inference_cls
        mock_asynth.synthesize = mock_synthesize

        mock_params_mod = MagicMock()
        mock_params_mod.GeneralSynthesisParams = mock_params_cls

        output_dir = str(tmp_path / "output")
        config = {
            "output_dir": output_dir,
            "model": "openai/gpt-4o",
            "api_base": "http://localhost:8000/v1",
            "api_key": "sk-test",
            "temperature": 0.5,
            "max_concurrency": 8,
            "num_samples": 50,
        }

        with patch.dict(
            sys.modules,
            {
                "asynth": mock_asynth,
                "asynth.configs": MagicMock(),
                "asynth.configs.params": MagicMock(),
                "asynth.configs.params.synthesis_params": mock_params_mod,
            },
        ):
            run_sdg(config)

        mock_inference_cls.assert_called_once_with(
            model="openai/gpt-4o",
            api_base="http://localhost:8000/v1",
            api_key="sk-test",
            temperature=0.5,
            max_concurrency=8,
            max_tokens=None,
            top_p=None,
            seed=None,
            num_retries=3,
        )
        mock_synthesize.assert_called_once()

    def test_strategy_params_passthrough(self, tmp_path: Any) -> None:
        mock_synthesize = MagicMock(return_value=[])
        mock_inference_cls = MagicMock()
        mock_synth_config_cls = MagicMock()
        mock_params_cls = MagicMock()

        mock_asynth = MagicMock()
        mock_asynth.SynthesisConfig = mock_synth_config_cls
        mock_asynth.LiteLLMInferenceConfig = mock_inference_cls
        mock_asynth.synthesize = mock_synthesize

        mock_params_mod = MagicMock()
        mock_params_mod.GeneralSynthesisParams = mock_params_cls

        output_dir = str(tmp_path / "output")
        strategy = {"sampled_attributes": [{"name": "domain", "values": ["science"]}]}
        config = {
            "output_dir": output_dir,
            "model": "openai/gpt-4o",
            "strategy_params": strategy,
        }

        with patch.dict(
            sys.modules,
            {
                "asynth": mock_asynth,
                "asynth.configs": MagicMock(),
                "asynth.configs.params": MagicMock(),
                "asynth.configs.params.synthesis_params": mock_params_mod,
            },
        ):
            run_sdg(config)

        mock_params_cls.from_dict.assert_called_once_with(strategy)


class TestStrategyParamsDeserialization:
    """Nested dicts in strategy_params are converted to proper dataclasses."""

    @staticmethod
    def _mock_asynth_modules() -> dict[str, Any]:
        """Build mock asynth modules that use real dataclasses."""
        import dataclasses

        @dataclasses.dataclass
        class SampledAttributeValue:
            id: str
            name: str
            description: str
            sample_rate: float | None = None

        @dataclasses.dataclass
        class SampledAttribute:
            id: str
            name: str
            description: str
            possible_values: list[SampledAttributeValue] = dataclasses.field(default_factory=list)

        @dataclasses.dataclass
        class TextMessage:
            role: str
            content: str

        @dataclasses.dataclass
        class GeneratedAttributePostprocessingParams:
            id: str

        @dataclasses.dataclass
        class GeneratedAttribute:
            id: str
            instruction_messages: list[TextMessage] = dataclasses.field(default_factory=list)
            postprocessing_params: GeneratedAttributePostprocessingParams | None = None

        @dataclasses.dataclass
        class MultiTurnAttribute:
            id: str
            min_turns: int = 1
            max_turns: int = 5

        @dataclasses.dataclass
        class TextConversation:
            messages: list[TextMessage] = dataclasses.field(default_factory=list)

        @dataclasses.dataclass
        class TransformationStrategy:
            type: str
            string_transform: str | None = None
            chat_transform: TextConversation | None = None

        @dataclasses.dataclass
        class TransformedAttribute:
            id: str
            transformation_strategy: TransformationStrategy | None = None

        @dataclasses.dataclass
        class DatasetSource:
            path: str
            id: str | None = None

        @dataclasses.dataclass
        class DocumentSegmentationParams:
            id: str

        @dataclasses.dataclass
        class DocumentSource:
            path: str
            id: str
            segmentation_params: DocumentSegmentationParams | None = None

        @dataclasses.dataclass
        class ExampleSource:
            examples: list[dict[str, Any]] = dataclasses.field(default_factory=list)

        @dataclasses.dataclass
        class AttributeCombination:
            combination: dict[str, str] = dataclasses.field(default_factory=dict)
            sample_rate: float = 0.5

        @dataclasses.dataclass
        class GeneralSynthesisParams:
            sampled_attributes: list[SampledAttribute] | None = None
            generated_attributes: list[GeneratedAttribute] | None = None
            multiturn_attributes: list[MultiTurnAttribute] | None = None
            transformed_attributes: list[TransformedAttribute] | None = None
            input_data: list[DatasetSource] | None = None
            input_documents: list[DocumentSource] | None = None
            input_examples: list[ExampleSource] | None = None
            combination_sampling: list[AttributeCombination] | None = None
            passthrough_attributes: list[str] | None = None

        mock_params_mod = MagicMock()
        mock_params_mod.GeneralSynthesisParams = GeneralSynthesisParams
        mock_params_mod.SampledAttribute = SampledAttribute
        mock_params_mod.SampledAttributeValue = SampledAttributeValue
        mock_params_mod.GeneratedAttribute = GeneratedAttribute
        mock_params_mod.GeneratedAttributePostprocessingParams = (
            GeneratedAttributePostprocessingParams
        )
        mock_params_mod.TextMessage = TextMessage
        mock_params_mod.TextConversation = TextConversation
        mock_params_mod.TransformationStrategy = TransformationStrategy
        mock_params_mod.TransformedAttribute = TransformedAttribute
        mock_params_mod.MultiTurnAttribute = MultiTurnAttribute
        mock_params_mod.DatasetSource = DatasetSource
        mock_params_mod.DocumentSource = DocumentSource
        mock_params_mod.DocumentSegmentationParams = DocumentSegmentationParams
        mock_params_mod.ExampleSource = ExampleSource
        mock_params_mod.AttributeCombination = AttributeCombination

        return {
            "asynth": MagicMock(),
            "asynth.configs": MagicMock(),
            "asynth.configs.params": MagicMock(),
            "asynth.configs.params.synthesis_params": mock_params_mod,
            "_classes": {
                "SampledAttribute": SampledAttribute,
                "SampledAttributeValue": SampledAttributeValue,
                "GeneratedAttribute": GeneratedAttribute,
                "DatasetSource": DatasetSource,
                "GeneralSynthesisParams": GeneralSynthesisParams,
            },
        }

    def test_sampled_attributes_with_nested_values(self) -> None:
        mods = self._mock_asynth_modules()
        classes = mods.pop("_classes")
        raw = {
            "sampled_attributes": [
                {
                    "id": "urgency",
                    "name": "Urgency Level",
                    "description": "How urgent the request is",
                    "possible_values": [
                        {"id": "high", "name": "High", "description": "Very urgent"},
                        {"id": "low", "name": "Low", "description": "Not urgent"},
                    ],
                }
            ]
        }
        with patch.dict(sys.modules, mods):
            result = _deserialize_strategy_params(raw)

        assert isinstance(result, classes["GeneralSynthesisParams"])
        assert len(result.sampled_attributes) == 1
        attr = result.sampled_attributes[0]
        assert isinstance(attr, classes["SampledAttribute"])
        assert attr.id == "urgency"
        assert len(attr.possible_values) == 2
        assert isinstance(attr.possible_values[0], classes["SampledAttributeValue"])
        assert attr.possible_values[0].id == "high"

    def test_input_data_as_dicts(self) -> None:
        mods = self._mock_asynth_modules()
        classes = mods.pop("_classes")
        raw = {
            "input_data": [
                {"path": "data.jsonl"},
                {"path": "hf:org/dataset", "id": "ds1"},
            ]
        }
        with patch.dict(sys.modules, mods):
            result = _deserialize_strategy_params(raw)

        assert len(result.input_data) == 2
        assert isinstance(result.input_data[0], classes["DatasetSource"])
        assert result.input_data[0].path == "data.jsonl"

    def test_generated_attributes_with_nested_messages(self) -> None:
        mods = self._mock_asynth_modules()
        classes = mods.pop("_classes")
        raw = {
            "generated_attributes": [
                {
                    "id": "response",
                    "instruction_messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Answer: {question}"},
                    ],
                }
            ]
        }
        with patch.dict(sys.modules, mods):
            result = _deserialize_strategy_params(raw)

        assert len(result.generated_attributes) == 1
        ga = result.generated_attributes[0]
        assert isinstance(ga, classes["GeneratedAttribute"])
        assert len(ga.instruction_messages) == 2

    def test_empty_strategy_params(self) -> None:
        mods = self._mock_asynth_modules()
        classes = mods.pop("_classes")
        with patch.dict(sys.modules, mods):
            result = _deserialize_strategy_params({})

        assert isinstance(result, classes["GeneralSynthesisParams"])
        assert result.sampled_attributes is None
