"""Unit tests for the validate-time confirmation-card surface: the trainer-default
hyperparameter list (algorithm-aware, alias-canonicalized) and its carriage on
ValidatedJobConfig. No DB — these exercise pure helpers and the response model.
"""

from amortized.api import jobs as jobs_mod
from amortized.models import JobType, ValidatedJobConfig


class TestTrainerDefaultFields:
    def test_generic_knobs_defaulted_when_unset(self) -> None:
        fields = jobs_mod._trainer_default_fields({"algorithm": "sft"})
        assert "learning_rate" in fields
        assert "num_train_epochs" in fields

    def test_set_knob_is_not_reported(self) -> None:
        fields = jobs_mod._trainer_default_fields({"algorithm": "sft", "learning_rate": 0.0002})
        assert "learning_rate" not in fields

    def test_algorithm_alias_resolves_to_lora_knobs(self) -> None:
        # 'qlora' is an ALGO_ALIASES alias for 'lora_sft'. Without canonicalization the
        # LoRA tunables would never be marked trainer-default (the raw alias is not in
        # _LORA_ALGORITHMS and use_peft is unset here).
        fields = jobs_mod._trainer_default_fields({"algorithm": "qlora"})
        assert "lora_r" in fields

    def test_non_lora_algorithm_omits_lora_knobs(self) -> None:
        fields = jobs_mod._trainer_default_fields({"algorithm": "sft"})
        assert "lora_r" not in fields


class TestValidatedJobConfigTrainerDefaults:
    def test_trainer_default_fields_round_trips(self) -> None:
        # The field must be declared on the model; otherwise pydantic drops the kwarg
        # silently (no extra="allow") and the card never sees the list.
        cfg = ValidatedJobConfig(
            job_type=JobType.training,
            trainer_default_fields=["learning_rate", "num_train_epochs"],
        )
        assert cfg.trainer_default_fields == ["learning_rate", "num_train_epochs"]

    def test_trainer_default_fields_defaults_empty(self) -> None:
        cfg = ValidatedJobConfig(job_type=JobType.sdg)
        assert cfg.trainer_default_fields == []
