"""Training job builder tests — CPU branch + GPU byte-identical invariant (issue #442, Phase 2)."""

from typing import Any

import pytest
import yaml

import amortized.config as config_mod
from amortized.backends import Resources
from amortized.jobs import training as training_builder
from amortized.jobs.base import JobBuildResult


def _gpu_config() -> dict[str, Any]:
    """A representative GPU config — the shape jobs have used to date."""
    return {
        "algorithm": "sft",
        "model_name_or_path": "Qwen/Qwen3.5-4B",
        "data_path": "./data.jsonl",
        "num_train_epochs": 1,
        "bf16": True,
        "nproc_per_node": 1,
    }


def _cpu_config(**overrides: Any) -> dict[str, Any]:
    base = {
        "algorithm": "sft",
        "model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "data_path": "./data.jsonl",
        "device": "cpu",
    }
    return {**base, **overrides}


async def _build(config: dict[str, Any]) -> JobBuildResult:
    return await training_builder.build({"id": "j-buildertest01", "config": config}, config, {})


def _default_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_mod.settings, "image_registry", "ghcr.io/amortized-ai")
    monkeypatch.setattr(config_mod.settings, "training_cpu_image_tag", "2026-09-15")


class TestGpuBranchUnchanged:
    """The merge-safety invariant: GPU output is byte-identical to pre-Phase-2."""

    async def test_gpu_spec_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _default_registry(monkeypatch)
        config = _gpu_config()
        result = await _build(dict(config))

        expected_yaml = {
            "model_path": "Qwen/Qwen3.5-4B",
            "data_path": "./data.jsonl",
            "num_epochs": 1,
            "bf16": True,
            "nproc_per_node": 1,
            "ckpt_output_dir": "/amortized/work/output",
            "data_output_dir": "/amortized/work/output/processed_data",
            "effective_batch_size": 8,
            "max_seq_len": 2048,
            "max_batch_len": 60000,
        }
        assert result == JobBuildResult(
            command=["thub", "sft", "--config", "/amortized/config.yaml"],
            config_files={"config.yaml": yaml.dump(expected_yaml, sort_keys=False)},
            env={},
            resources=Resources(gpus=1),
            image="ghcr.io/amortized-ai/training:latest",
            timeout=None,
            resolved_config=config,
            post_commands=[
                "mlflow artifacts log-artifacts -l /amortized/work/output"
                " -r $MLFLOW_RUN_ID -a model"
            ],
        )

    async def test_config_without_device_is_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Old/persisted configs without `device` produce identical GPU specs."""
        _default_registry(monkeypatch)
        config = _gpu_config()
        without_result = await _build(dict(config))
        with_result = await _build(dict(_gpu_config(), device="gpu"))

        # Everything except resolved_config (which merely echoes the extra key)
        # must be identical between the old and new config shapes.
        without_result.resolved_config = {}
        with_result.resolved_config = {}
        assert without_result == with_result
        assert without_result.resources == Resources(gpus=1)
        assert without_result.image == "ghcr.io/amortized-ai/training:latest"
        assert without_result.timeout is None
        assert without_result.env == {}
        assert without_result.command == ["thub", "sft", "--config", "/amortized/config.yaml"]

    async def test_gpu_nproc_maps_to_gpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _default_registry(monkeypatch)
        result = await _build(dict(_gpu_config(), nproc_per_node=2))
        assert result.resources == Resources(gpus=2)

    async def test_gpu_timeout_stays_none_even_if_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _default_registry(monkeypatch)
        result = await _build(dict(_gpu_config(), timeout_seconds=600))
        assert result.timeout is None


class TestCpuBranch:
    async def test_resources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _default_registry(monkeypatch)
        result = await _build(dict(_cpu_config()))
        assert result.resources == Resources(gpus=0, cpus=4, memory_gb=8)

    @pytest.mark.parametrize(
        ("model", "memory_gb"),
        [
            ("Qwen/Qwen3.5-0.8B", 8),
            ("Qwen/Qwen3.5-2B", 16),
            ("Qwen/Qwen3.5-4B", 32),
            ("unknown/model", 8),
        ],
    )
    async def test_memory_per_model(
        self, monkeypatch: pytest.MonkeyPatch, model: str, memory_gb: int
    ) -> None:
        _default_registry(monkeypatch)
        result = await _build(_cpu_config(model_name_or_path=model))
        assert result.resources.memory_gb == memory_gb
        assert result.resources.gpus == 0
        assert result.resources.cpus == 4

    async def test_env_matches_cpu_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _default_registry(monkeypatch)
        result = await _build(dict(_cpu_config()))
        assert result.env == {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
        }

    async def test_cpu_safe_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _default_registry(monkeypatch)
        result = await _build(_cpu_config(bf16=True, nproc_per_node=4))
        assert result.resolved_config["bf16"] is False
        assert result.resolved_config["nproc_per_node"] == 1
        thub = yaml.safe_load(result.config_files["config.yaml"])
        assert thub["bf16"] is False
        assert thub["nproc_per_node"] == 1

    async def test_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _default_registry(monkeypatch)
        result = await _build(dict(_cpu_config()))
        assert result.image == "ghcr.io/amortized-ai/training-cpu:2026-09-15"

    async def test_image_uses_configured_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config_mod.settings, "image_registry", "registry.local:5000")
        monkeypatch.setattr(config_mod.settings, "training_cpu_image_tag", "deadbeef")
        result = await _build(dict(_cpu_config()))
        assert result.image == "registry.local:5000/training-cpu:deadbeef"
        gpu_result = await _build(dict(_gpu_config()))
        assert gpu_result.image == "registry.local:5000/training:latest"

    async def test_timeout_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _default_registry(monkeypatch)
        result = await _build(dict(_cpu_config()))
        assert result.timeout == 3600

    async def test_timeout_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _default_registry(monkeypatch)
        result = await _build(_cpu_config(timeout_seconds=120))
        assert result.timeout == 120
        # timeout_seconds must not leak into the thub YAML
        thub = yaml.safe_load(result.config_files["config.yaml"])
        assert "timeout_seconds" not in thub
        assert "device" not in thub
