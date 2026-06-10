"""Tests for Pydantic model validation."""

import pytest
from pydantic import ValidationError

from amortized.models import (
    JobStatus,
    JobType,
    MemoryEstimateRequest,
    SynthJobConfig,
    TrainingJobConfig,
    TrainingMetric,
)


class TestTrainingJobConfig:
    def test_minimal_config(self) -> None:
        config = TrainingJobConfig(
            algorithm="sft",
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            data_path="./data.jsonl",
        )
        assert config.model_name_or_path == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.algorithm == "sft"
        assert config.output_dir is None
        assert config.learning_rate is None
        assert config.lora_r is None

    def test_full_config(self) -> None:
        config = TrainingJobConfig(
            algorithm="sft",
            model_name_or_path="meta-llama/Llama-3-8B",
            data_path="/data/train.jsonl",
            output_dir="/outputs/run1",
            learning_rate=1e-4,
            num_train_epochs=5,
            lora_r=32,
            lora_alpha=64,
            load_in_4bit=True,
            per_device_train_batch_size=4,
            batch_size=8,
            max_length=4096,
            gradient_checkpointing=True,
            model_max_length=8192,
        )
        assert config.lora_r == 32
        assert config.load_in_4bit is True
        assert config.batch_size == 8
        assert config.gradient_checkpointing is True
        assert config.model_max_length == 8192

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            TrainingJobConfig(  # type: ignore[call-arg]
                model_name_or_path="test",
                data_path="test",
                # missing algorithm
            )

    def test_invalid_lora_r(self) -> None:
        with pytest.raises(ValidationError):
            TrainingJobConfig(
                algorithm="sft",
                model_name_or_path="test",
                data_path="test",
                lora_r=0,
            )

    def test_exclude_none_serialization(self) -> None:
        config = TrainingJobConfig(
            algorithm="sft",
            model_name_or_path="test",
            data_path="test",
        )
        dumped = config.model_dump(exclude_none=True)
        assert "learning_rate" not in dumped
        assert "output_dir" not in dumped
        assert "model_name_or_path" in dumped
        assert "algorithm" in dumped


class TestSynthJobConfig:
    def test_minimal_config(self) -> None:
        config = SynthJobConfig(
            model="openai/gpt-4o",
        )
        assert config.model == "openai/gpt-4o"
        assert config.api_base is None
        assert config.num_samples == 100
        assert config.strategy_params is None

    def test_full_config(self) -> None:
        config = SynthJobConfig(
            model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="sk-test",
            num_samples=200,
            max_concurrency=8,
            temperature=0.5,
            strategy_params={"sampled_attributes": [{"name": "domain", "values": ["science"]}]},
        )
        assert config.strategy_params is not None
        assert config.max_concurrency == 8

    def test_missing_model(self) -> None:
        with pytest.raises(ValidationError):
            SynthJobConfig()  # type: ignore[call-arg]


class TestMemoryEstimateRequest:
    def test_defaults(self) -> None:
        req = MemoryEstimateRequest(model_name_or_path="test/model")
        assert req.lora_r == 16
        assert req.batch_size == 2
        assert req.max_length == 2048
        assert req.load_in_4bit is False

    def test_custom_values(self) -> None:
        req = MemoryEstimateRequest(
            model_name_or_path="test/model",
            lora_r=64,
            batch_size=8,
            max_length=4096,
            load_in_4bit=True,
        )
        assert req.lora_r == 64
        assert req.load_in_4bit is True


class TestEnums:
    def test_job_status_values(self) -> None:
        assert JobStatus.queued.value == "queued"
        assert JobStatus.provisioning.value == "provisioning"
        assert JobStatus.running.value == "running"
        assert JobStatus.succeeded.value == "succeeded"
        assert JobStatus.failed.value == "failed"
        assert JobStatus.cancelled.value == "cancelled"
        # Backward-compat aliases
        assert JobStatus.pending is JobStatus.queued
        assert JobStatus.completed is JobStatus.succeeded

    def test_job_type_values(self) -> None:
        assert JobType.training.value == "training"
        assert JobType.sdg.value == "sdg"


class TestTrainingMetric:
    def test_full_metric(self) -> None:
        metric = TrainingMetric(step=10, loss=2.345, epoch=1.0, learning_rate=1e-5, max_steps=1000)
        assert metric.step == 10
        assert metric.loss == 2.345

    def test_minimal_metric(self) -> None:
        metric = TrainingMetric(step=1, loss=3.0)
        assert metric.epoch is None
        assert metric.learning_rate is None
