"""Tests for the effective-config echo (2a): validate_training_job reports which tunable
hyperparameters the config left unset, so the card shows "trainer default" instead of a
value the model may have narrated but never wrote into the config. The candidate set is
algorithm-aware — generic knobs always, LoRA knobs only for PEFT/LoRA runs, embedding knobs
only for embedding runs."""

from amortized.api.jobs import (
    _EMBEDDING_TUNABLE_FIELDS,
    _GENERIC_TUNABLE_FIELDS,
    _LORA_TUNABLE_FIELDS,
    _trainer_default_fields,
)


def test_generic_fields_all_unset_for_plain_sft() -> None:
    # A non-LoRA, non-embedding run defaults the generic optimization knobs and nothing else.
    assert _trainer_default_fields({"algorithm": "sft"}) == list(_GENERIC_TUNABLE_FIELDS)


def test_empty_config_reports_generic_only() -> None:
    # No algorithm → no LoRA/embedding knobs, just the generic set.
    assert _trainer_default_fields({}) == list(_GENERIC_TUNABLE_FIELDS)


def test_set_fields_are_excluded() -> None:
    config = {
        "algorithm": "osft",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
    }
    defaults = _trainer_default_fields(config)
    assert "num_train_epochs" not in defaults
    assert "per_device_train_batch_size" not in defaults
    # the ones left unset still report as trainer-default
    assert "learning_rate" in defaults
    assert "gradient_accumulation_steps" in defaults
    assert "max_length" in defaults


def test_none_valued_field_counts_as_unset() -> None:
    # an explicit None (e.g. from exclude_none not stripping it) is still "unset"
    assert "learning_rate" in _trainer_default_fields({"learning_rate": None})


def test_only_candidate_fields_reported() -> None:
    # non-hyperparameter config keys never appear in the trainer-default list
    defaults = _trainer_default_fields({"algorithm": "osft", "model_name_or_path": "Qwen/x"})
    allowed = set(_GENERIC_TUNABLE_FIELDS) | set(_LORA_TUNABLE_FIELDS) | set(
        _EMBEDDING_TUNABLE_FIELDS
    )
    assert set(defaults) <= allowed


def test_lora_algorithm_includes_lora_knobs() -> None:
    defaults = _trainer_default_fields({"algorithm": "lora_sft"})
    assert set(_LORA_TUNABLE_FIELDS) <= set(defaults)
    # still no embedding knobs on a LoRA run
    assert not (set(_EMBEDDING_TUNABLE_FIELDS) & set(defaults))


def test_use_peft_includes_lora_knobs_even_for_sft() -> None:
    # PEFT on a plain SFT run means the LoRA knobs are live and default-able.
    defaults = _trainer_default_fields({"algorithm": "sft", "use_peft": True})
    assert set(_LORA_TUNABLE_FIELDS) <= set(defaults)


def test_non_lora_algorithm_excludes_lora_knobs() -> None:
    # OSFT / full fine-tunes never use LoRA adapters, so don't claim those knobs are defaulted.
    defaults = _trainer_default_fields({"algorithm": "osft"})
    assert not (set(_LORA_TUNABLE_FIELDS) & set(defaults))


def test_set_lora_knob_is_excluded() -> None:
    defaults = _trainer_default_fields({"algorithm": "lora_sft", "lora_r": 16})
    assert "lora_r" not in defaults
    assert "lora_alpha" in defaults


def test_embedding_algorithm_includes_embedding_knobs_not_lora() -> None:
    defaults = _trainer_default_fields({"algorithm": "embedding_sft"})
    assert set(_EMBEDDING_TUNABLE_FIELDS) <= set(defaults)
    assert not (set(_LORA_TUNABLE_FIELDS) & set(defaults))
    # generic knobs still apply to embedding runs
    assert "learning_rate" in defaults


def test_algorithm_hyphen_normalized() -> None:
    # "lora-sft" normalizes to lora_sft so the LoRA knobs still surface.
    defaults = _trainer_default_fields({"algorithm": "lora-sft"})
    assert set(_LORA_TUNABLE_FIELDS) <= set(defaults)
