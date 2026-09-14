"""Tests for the CPU training guardrail matrix (issue #442, Phase 1)."""

import pytest

from amortized.core.cpu_policy import check_cpu_policy
from amortized.core.model_catalog import training_model_cpu_compatibility


def _config(**overrides: object) -> dict:
    base = {
        "algorithm": "sft",
        "model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "data_path": "./data.jsonl",
        "device": "cpu",
    }
    return {**base, **overrides}


class TestModelCatalogCpuFlags:
    def test_ok_models(self) -> None:
        assert training_model_cpu_compatibility("Qwen/Qwen3.5-0.8B") == "ok"
        assert training_model_cpu_compatibility("Qwen/Qwen3.5-2B") == "ok"

    def test_warn_model(self) -> None:
        assert training_model_cpu_compatibility("Qwen/Qwen3.5-4B") == "warn"

    def test_reject_model(self) -> None:
        assert training_model_cpu_compatibility("Qwen/Qwen3.5-9B") == "reject"

    def test_matches_by_size(self) -> None:
        assert training_model_cpu_compatibility("org/foo-2B") == "ok"
        assert training_model_cpu_compatibility("2B") == "ok"

    def test_unknown_model_is_neutral(self) -> None:
        assert training_model_cpu_compatibility("test/model") is None


class TestCpuPolicyMatrix:
    def test_gpu_device_untouched(self) -> None:
        cfg = _config(device="gpu", model_name_or_path="Qwen/Qwen3.5-9B", algorithm="grpo")
        assert check_cpu_policy(cfg) == ([], [])

    def test_missing_device_untouched(self) -> None:
        cfg = _config()
        del cfg["device"]
        assert check_cpu_policy(cfg) == ([], [])

    def test_cpu_ok_model_clean_sft_lora(self) -> None:
        errors, warnings = check_cpu_policy(
            _config(algorithm="lora_sft", use_peft=True, bf16=False)
        )
        assert errors == []
        assert len(warnings) == 1  # only the slow-by-design notice
        assert "slow by design" in warnings[0]

    def test_cpu_reject_model(self) -> None:
        errors, warnings = check_cpu_policy(
            _config(model_name_or_path="Qwen/Qwen3.5-9B", algorithm="lora_sft")
        )
        assert len(errors) == 1
        assert "too large" in errors[0]
        assert len(warnings) == 1

    def test_cpu_warn_model(self) -> None:
        errors, warnings = check_cpu_policy(
            _config(model_name_or_path="Qwen/Qwen3.5-4B", algorithm="lora_sft")
        )
        assert errors == []
        assert any("very slow" in w for w in warnings)

    def test_vllm_dependent_algorithms_rejected(self) -> None:
        for algo in ("grpo", "lora_grpo", "gepa"):
            errors, _ = check_cpu_policy(_config(algorithm=algo))
            assert len(errors) == 1, algo
            assert "vLLM" in errors[0]

    def test_non_vllm_algorithms_not_rejected_for_method(self) -> None:
        for algo in ("sft", "lora_sft", "osft", "dpo", "kto", "gkd"):
            errors, _ = check_cpu_policy(
                _config(algorithm=algo, use_peft=True, unfreeze_rank_ratio=0.2)
            )
            assert errors == [], algo

    def test_qlora_rejected(self) -> None:
        errors, _ = check_cpu_policy(
            _config(algorithm="lora_sft", use_peft=True, load_in_4bit=True)
        )
        assert len(errors) == 1
        assert "QLoRA" in errors[0]

    def test_qlora_flag_rejected(self) -> None:
        errors, _ = check_cpu_policy(_config(algorithm="lora_sft", qlora=True))
        assert len(errors) == 1
        assert "QLoRA" in errors[0]

    def test_bf16_warns(self) -> None:
        errors, warnings = check_cpu_policy(_config(algorithm="lora_sft", use_peft=True, bf16=True))
        assert errors == []
        assert any("bf16" in w for w in warnings)

    def test_full_param_sft_on_ok_model_warns(self) -> None:
        for size in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"):
            errors, warnings = check_cpu_policy(_config(model_name_or_path=size))
            assert errors == [], size
            assert any("AdamW" in w for w in warnings), size

    def test_peft_sft_does_not_warn_full_param(self) -> None:
        errors, warnings = check_cpu_policy(_config(algorithm="lora_sft", use_peft=True))
        assert errors == []
        assert not any("AdamW" in w for w in warnings)

    def test_nproc_per_node_gt_1_warns(self) -> None:
        errors, warnings = check_cpu_policy(
            _config(algorithm="lora_sft", use_peft=True, nproc_per_node=4)
        )
        assert errors == []
        assert any("nproc_per_node" in w for w in warnings)

    def test_nproc_per_node_1_does_not_warn(self) -> None:
        errors, warnings = check_cpu_policy(
            _config(algorithm="lora_sft", use_peft=True, nproc_per_node=1)
        )
        assert errors == []
        assert not any("nproc_per_node" in w for w in warnings)

    def test_multiple_warnings_accumulate(self) -> None:
        errors, warnings = check_cpu_policy(
            _config(
                algorithm="lora_sft",
                use_peft=True,
                bf16=True,
                nproc_per_node=2,
                model_name_or_path="Qwen/Qwen3.5-4B",
            )
        )
        assert errors == []
        assert len(warnings) == 4

    def test_errors_and_warnings_coexist(self) -> None:
        errors, warnings = check_cpu_policy(
            _config(algorithm="grpo", model_name_or_path="Qwen/Qwen3.5-9B")
        )
        assert len(errors) == 2
        assert len(warnings) == 1  # notice only — 9B never reaches the warn branch


class TestValidateTrainingJobWiring:
    """Call the endpoint function directly — data_path-set configs never touch the DB."""

    @pytest.mark.asyncio
    async def test_cpu_rejection_is_validation_error(self) -> None:
        from fastapi import HTTPException

        from amortized.api.jobs import validate_training_job
        from amortized.models import TrainingJobRequest

        request = TrainingJobRequest(**_config(model_name_or_path="Qwen/Qwen3.5-9B"))
        with pytest.raises(HTTPException) as exc_info:
            await validate_training_job(request=request, db=None)  # type: ignore[arg-type]
        assert exc_info.value.status_code == 422
        assert any("too large" in str(e) for e in exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_cpu_warnings_surfaced(self) -> None:
        from amortized.api.jobs import validate_training_job
        from amortized.models import TrainingJobRequest

        request = TrainingJobRequest(**_config(algorithm="lora_sft", use_peft=True, bf16=True))
        result = await validate_training_job(request=request, db=None)  # type: ignore[arg-type]
        assert result.valid is True
        assert result.warnings
        assert any("bf16" in w for w in result.warnings)
        assert result.config["device"] == "cpu"

    @pytest.mark.asyncio
    async def test_gpu_config_has_no_warnings(self) -> None:
        from amortized.api.jobs import validate_training_job
        from amortized.models import TrainingJobRequest

        request = TrainingJobRequest(**_config(device="gpu", algorithm="lora_sft"))
        result = await validate_training_job(request=request, db=None)  # type: ignore[arg-type]
        assert result.warnings == []
        assert "device" in result.config

    @pytest.mark.asyncio
    async def test_config_without_device_omits_field_and_warnings(self) -> None:
        from amortized.api.jobs import validate_training_job
        from amortized.models import TrainingJobRequest

        body = _config()
        del body["device"]
        request = TrainingJobRequest(**body)
        result = await validate_training_job(request=request, db=None)  # type: ignore[arg-type]
        # Backward compat: no device in config (exclude_unset), no warnings —
        # identical validation outcome to pre-change behavior.
        assert "device" not in result.config
        assert result.warnings == []


class TestDeviceSkippedFromThubConfig:
    def test_device_never_reaches_thub_yaml(self) -> None:
        import yaml

        from amortized.jobs.training import _training_hub_config_yaml

        config = _config(nproc_per_node=1)
        parsed = yaml.safe_load(_training_hub_config_yaml("sft", config))
        assert "device" not in parsed
        assert parsed["model_path"] == "Qwen/Qwen3.5-0.8B"
