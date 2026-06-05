"""Tests for the SDG runner — simulation fallback and config mapping."""

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

from amortized.runners.sdg_runner import run_sdg


class TestSimulationFallback:
    """When asynth is not installed, the runner falls back to simulation."""

    def test_simulation_produces_output(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "output")
        config = {"output_dir": output_dir, "model": "openai/gpt-4o", "num_samples": 10}
        run_sdg(config)

        output_path = os.path.join(output_dir, "generated_data.jsonl")
        assert os.path.exists(output_path)

        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 50

        first_row = json.loads(lines[0])
        assert "instruction" in first_row
        assert "output" in first_row

    def test_simulation_writes_stats(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "output")
        config = {"output_dir": output_dir, "model": "openai/gpt-4o", "num_samples": 10}
        run_sdg(config)

        stats_path = os.path.join(output_dir, "stats.json")
        assert os.path.exists(stats_path)

        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["total_completed"] == 50
        assert stats["status"] == "completed"

    def test_default_output_dir(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        config = {"model": "openai/gpt-4o"}
        run_sdg(config)
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
            "max_concurrent": 8,
            "num_samples": 50,
        }

        with patch.dict(sys.modules, {
            "asynth": mock_asynth,
            "asynth.configs": MagicMock(),
            "asynth.configs.params": MagicMock(),
            "asynth.configs.params.synthesis_params": mock_params_mod,
        }):
            run_sdg(config)

        mock_inference_cls.assert_called_once_with(
            model="openai/gpt-4o",
            api_base="http://localhost:8000/v1",
            api_key="sk-test",
            temperature=0.5,
            max_concurrency=8,
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

        with patch.dict(sys.modules, {
            "asynth": mock_asynth,
            "asynth.configs": MagicMock(),
            "asynth.configs.params": MagicMock(),
            "asynth.configs.params.synthesis_params": mock_params_mod,
        }):
            run_sdg(config)

        mock_params_cls.assert_called_once_with(**strategy)
