"""Shared test fixtures and markers."""

import pytest


def _has_training_hub() -> bool:
    try:
        import training_hub  # noqa: F401

        return True
    except ImportError:
        return False


def _training_hub_functional() -> bool:
    """Check that training_hub can actually run estimates (not just import)."""
    try:
        from training_hub import LoRAEstimator

        LoRAEstimator(model_path="test", lora_r=16, batch_size=2, max_seq_len=512)
        return True
    except Exception:
        return False


requires_training_hub = pytest.mark.skipif(
    not _has_training_hub(),
    reason="training_hub not installed",
)

requires_training_hub_functional = pytest.mark.skipif(
    not _training_hub_functional(),
    reason="training_hub not installed or not functional",
)
