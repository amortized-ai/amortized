"""Tests for training resume (training_job_id) and check_model_source."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from amortized.jobs import training as training_builder
from amortized.models import TrainingJobConfig


class TestCheckModelSource:
    def test_requires_at_least_one_model_source(self) -> None:
        with pytest.raises(ValidationError, match="model_name_or_path"):
            TrainingJobConfig(algorithm="sft")

    def test_model_name_or_path_alone_accepted(self) -> None:
        config = TrainingJobConfig(
            algorithm="sft",
            model_name_or_path="meta-llama/Llama-3.1-8B",
        )
        assert config.model_name_or_path == "meta-llama/Llama-3.1-8B"
        assert config.training_job_id == ""

    def test_training_job_id_alone_accepted(self) -> None:
        config = TrainingJobConfig(
            algorithm="sft",
            training_job_id="abc-123",
        )
        assert config.training_job_id == "abc-123"
        assert config.model_name_or_path == ""


class TestBuildResumeFromFinetuned:
    @pytest.fixture
    def _mock_parent(self, monkeypatch):
        """Shared mocks for validate_training_parent and training_display_name."""
        parent = {
            "id": "parent-1",
            "type": "training",
            "status": "succeeded",
            "config": {
                "algorithm": "lora_sft",
                "model_name_or_path": "meta-llama/Llama-3.1-8B",
            },
            "mlflow_run_id": "run-lora-1",
        }

        async def fake_validate(config):
            return parent, "run-lora-1"

        async def fake_display_name(p, run_id):
            return "mdl-parent-run"

        monkeypatch.setattr(
            training_builder, "validate_training_parent", fake_validate
        )
        monkeypatch.setattr(
            training_builder, "training_display_name", fake_display_name
        )
        return parent

    @pytest.mark.asyncio
    async def test_lora_parent_merges_and_trains(
        self, monkeypatch, _mock_parent
    ) -> None:
        """LoRA parent: adapter merged into base, resolved config shows base model."""

        async def fake_resolve(config, *, skip_display_name=False):
            return (
                "",
                "/amortized/work/served_model/merged",
                [
                    "mlflow artifacts download -r run-lora-1 -a model"
                    " -d /amortized/work/served_model",
                    "python3 /amortized/merge_lora.py"
                    " /amortized/work/served_model/model"
                    " /amortized/work/served_model/merged",
                    "python3 /amortized/patch_model_config.py"
                    " /amortized/work/served_model/merged",
                ],
            )

        monkeypatch.setattr(
            training_builder, "resolve_training_model", fake_resolve
        )

        config = {
            "algorithm": "lora_sft",
            "training_job_id": "parent-1",
            "data_path": "/data/train.jsonl",
        }
        result = await training_builder.build(
            {"id": "j1", "type": "training"}, config, {}
        )

        assert result.resolved_config["model_name_or_path"] == (
            "meta-llama/Llama-3.1-8B"
        )
        assert result.resolved_config["resume_model_name"] == "mdl-parent-run"
        pre = "\n".join(result.pre_commands)
        assert "merge_lora.py" in pre
        assert "merge_lora.py" in result.config_files

    @pytest.mark.asyncio
    async def test_full_finetune_parent_uses_last_step(
        self, monkeypatch, _mock_parent
    ) -> None:
        """Non-LoRA parent: last step used, resolved config shows base model."""

        async def fake_resolve(config, *, skip_display_name=False):
            return (
                "",
                "$SERVE_MODEL_DIR",
                [
                    "mlflow artifacts download -r run-sft-1 -a model"
                    " -d /amortized/work/served_model",
                    "SERVE_MODEL_DIR=$(find"
                    " /amortized/work/served_model/model/hf_format"
                    " -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)",
                    "python3 /amortized/patch_model_config.py"
                    " $SERVE_MODEL_DIR",
                ],
            )

        monkeypatch.setattr(
            training_builder, "resolve_training_model", fake_resolve
        )

        config = {
            "algorithm": "sft",
            "training_job_id": "parent-2",
            "data_path": "/data/train.jsonl",
        }
        result = await training_builder.build(
            {"id": "j2", "type": "training"}, config, {}
        )

        assert result.resolved_config["model_name_or_path"] == (
            "meta-llama/Llama-3.1-8B"
        )
        assert result.resolved_config["resume_model_name"] == "mdl-parent-run"
        assert "ln -sfn" in result.pre_commands[-1]
        assert "merge_lora.py" in result.config_files

    @pytest.mark.asyncio
    async def test_build_without_training_job_id_uses_model_name(self) -> None:
        config = {
            "algorithm": "sft",
            "model_name_or_path": "meta-llama/Llama-3.1-8B",
            "data_path": "/data/train.jsonl",
        }
        result = await training_builder.build(
            {"id": "j3", "type": "training"}, config, {}
        )

        assert not result.pre_commands
        assert "merge_lora.py" not in result.config_files


class TestBuildGpuResources:
    """embedding_sft trains CPU-only in-process (no torchrun); other algorithms scale GPUs
    with nproc_per_node. See issue #486."""

    @pytest.mark.asyncio
    async def test_embedding_sft_requests_zero_gpus(self) -> None:
        config = {
            "algorithm": "embedding_sft",
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "data_path": "/data/train.jsonl",
        }
        result = await training_builder.build({"id": "emb1", "type": "training"}, config, {})
        assert result.resources.gpus == 0

    @pytest.mark.asyncio
    async def test_embedding_sft_ignores_nproc_per_node(self) -> None:
        # nproc_per_node is ignored by embedding_sft, so it must not request GPUs even if set.
        config = {
            "algorithm": "embedding_sft",
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "data_path": "/data/train.jsonl",
            "nproc_per_node": 4,
        }
        result = await training_builder.build({"id": "emb2", "type": "training"}, config, {})
        assert result.resources.gpus == 0

    @pytest.mark.asyncio
    async def test_classifier_alias_requests_zero_gpus(self) -> None:
        config = {
            "algorithm": "classifier",  # alias -> embedding_sft
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "data_path": "/data/train.jsonl",
        }
        result = await training_builder.build({"id": "emb3", "type": "training"}, config, {})
        assert result.resources.gpus == 0

    @pytest.mark.asyncio
    async def test_sft_still_requests_gpu(self) -> None:
        config = {
            "algorithm": "sft",
            "model_name_or_path": "meta-llama/Llama-3.1-8B",
            "data_path": "/data/train.jsonl",
        }
        result = await training_builder.build({"id": "sft1", "type": "training"}, config, {})
        assert result.resources.gpus == 1


class TestDataRunIdSkipped:
    def test_data_run_id_not_in_training_hub_config(self) -> None:
        config = {
            "algorithm": "sft",
            "model_name_or_path": "test/model",
            "data_path": "/data/train.jsonl",
            "data_run_id": "run-xyz",
        }
        yaml_str = training_builder._training_hub_config_yaml("sft", config)
        assert "data_run_id" not in yaml_str
        assert "run-xyz" not in yaml_str

    def test_training_job_id_not_in_training_hub_config(self) -> None:
        config = {
            "algorithm": "sft",
            "model_name_or_path": "test/model",
            "data_path": "/data/train.jsonl",
            "training_job_id": "tj-abc",
        }
        yaml_str = training_builder._training_hub_config_yaml("sft", config)
        assert "training_job_id" not in yaml_str
        assert "tj-abc" not in yaml_str


class TestOnSuccessResumeName:
    @pytest.mark.asyncio
    async def test_on_success_uses_parent_base_model(self) -> None:
        parent_job = {
            "id": "parent-job-id-1234",
            "config": {
                "algorithm": "sft",
                "model_name_or_path": "meta-llama/Llama-3.1-8B",
            },
            "mlflow_run_id": "parent-run-1",
        }

        mock_repo = AsyncMock()
        mock_repo.get_job = AsyncMock(return_value=parent_job)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=MagicMock()
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        registered = False

        class FakeMLflow:
            def __init__(self, uri):
                pass

            async def register_model(self, name, run_id, description):
                nonlocal registered
                registered = True
                assert "Llama-3.1-8B" in name
                assert "meta-llama/Llama-3.1-8B" in description
                assert "resumed from" in description
                return True

            async def get_run(self, run_id):
                return {
                    "info": {"run_name": "run-name"},
                    "data": {"tags": []},
                }

            async def set_registered_model_tag(self, name, key, value):
                pass

        job = {
            "id": "child-job-id-5678",
            "config": {
                "algorithm": "sft",
                "model_name_or_path": "/amortized/work/served_model/merged",
                "training_job_id": "parent-job-id-1234",
            },
        }

        async def fake_display_name(parent, run_id):
            return "mdl-parent-run"

        with (
            patch(
                "amortized.jobs.training.MLflowClient", FakeMLflow
            ),
            patch(
                "amortized.jobs.training.config_mod"
            ) as mock_config,
            patch(
                "amortized.db.connection.get_pool",
                return_value=mock_pool,
            ),
            patch(
                "amortized.db.repository.Repository",
                return_value=mock_repo,
            ),
            patch(
                "amortized.jobs.common.training_display_name",
                side_effect=fake_display_name,
            ),
            patch(
                "amortized.jobs.training.set_mlflow_run_tag",
                new_callable=AsyncMock,
            ),
        ):
            mock_config.settings.mlflow_tracking_uri = "http://mlflow:5000"
            await training_builder.on_success(job, "child-run-1")

        assert registered
