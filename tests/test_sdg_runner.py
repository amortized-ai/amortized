"""Tests for the SDG runner — simulation fallback, config mapping, and output quality checks."""

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

from amortized.runners.sdg_runner import _check_output_quality, run_sdg


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

    def test_no_api_key_passes_none(self, tmp_path: Any) -> None:
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
        config = {
            "output_dir": output_dir,
            "model": "openai/gpt-4o",
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

        call_kwargs = mock_inference_cls.call_args[1]
        assert call_kwargs["api_key"] is None


class TestOutputQualityCheck:
    """_check_output_quality warns on identical samples (template echo)."""

    def test_warns_on_identical_samples(self, caplog: Any) -> None:
        results = [{"answer": "same"} for _ in range(5)]
        config = {
            "strategy_params": {
                "generated_attributes": [{"id": "answer"}],
            }
        }
        with caplog.at_level("WARNING"):
            _check_output_quality(results, config)
        assert "identical" in caplog.text

    def test_no_warning_on_diverse_samples(self, caplog: Any) -> None:
        results = [{"answer": f"answer_{i}"} for i in range(5)]
        config = {
            "strategy_params": {
                "generated_attributes": [{"id": "answer"}],
            }
        }
        with caplog.at_level("WARNING"):
            _check_output_quality(results, config)
        assert "identical" not in caplog.text

    def test_no_warning_without_strategy_params(self, caplog: Any) -> None:
        results = [{"answer": "same"} for _ in range(5)]
        config: dict[str, Any] = {}
        with caplog.at_level("WARNING"):
            _check_output_quality(results, config)
        assert caplog.text == ""

    def test_no_warning_on_single_result(self, caplog: Any) -> None:
        results = [{"answer": "same"}]
        config = {
            "strategy_params": {
                "generated_attributes": [{"id": "answer"}],
            }
        }
        with caplog.at_level("WARNING"):
            _check_output_quality(results, config)
        assert "identical" not in caplog.text
