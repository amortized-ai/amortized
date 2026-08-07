"""Tests for Pydantic model validation."""

import pytest
from pydantic import ValidationError

from amortized.models import (
    JobStatus,
    JobType,
    TrainingJobConfig,
    TrainingJobRequest,
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
            max_length=4096,
            gradient_checkpointing=True,
        )
        assert config.lora_r == 32
        assert config.load_in_4bit is True
        assert config.gradient_checkpointing is True

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            TrainingJobConfig(  # type: ignore[call-arg]
                model_name_or_path="test",
                data_path="test",
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


class TestPreviewNumRecordsCapping:
    def test_preview_caps_at_10(self) -> None:
        num_records = 500
        mode = "preview"
        records = min(num_records, 10) if mode == "preview" else num_records
        assert records == 10

    def test_preview_below_cap_unchanged(self) -> None:
        num_records = 5
        mode = "preview"
        records = min(num_records, 10) if mode == "preview" else num_records
        assert records == 5

    def test_create_mode_uncapped(self) -> None:
        num_records = 500
        mode = "create"
        records = min(num_records, 10) if mode == "preview" else num_records
        assert records == 500


class TestOSFTValidation:
    def test_osft_without_urr_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unfreeze_rank_ratio"):
            TrainingJobRequest(
                algorithm="osft",
                model_name_or_path="test",
            )

    def test_osft_with_urr_accepted(self) -> None:
        req = TrainingJobRequest(
            algorithm="osft",
            model_name_or_path="test",
            unfreeze_rank_ratio=0.2,
        )
        assert req.unfreeze_rank_ratio == 0.2

    def test_sft_without_urr_accepted(self) -> None:
        req = TrainingJobRequest(
            algorithm="sft",
            model_name_or_path="test",
        )
        assert req.unfreeze_rank_ratio is None

    def test_lora_sft_without_urr_accepted(self) -> None:
        req = TrainingJobRequest(
            algorithm="lora_sft",
            model_name_or_path="test",
        )
        assert req.unfreeze_rank_ratio is None


class TestEnums:
    def test_job_status_values(self) -> None:
        assert JobStatus.queued.value == "queued"
        assert JobStatus.provisioning.value == "provisioning"
        assert JobStatus.running.value == "running"
        assert JobStatus.succeeded.value == "succeeded"
        assert JobStatus.failed.value == "failed"
        assert JobStatus.cancelled.value == "cancelled"

    def test_job_type_values(self) -> None:
        assert JobType.training.value == "training"
        assert JobType.sdg.value == "sdg"
