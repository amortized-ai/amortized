"""Job builder registry — maps job types to their builder modules."""

from __future__ import annotations

from types import ModuleType

from amortized.models import JobType


def get_builder(job_type: str) -> ModuleType | None:
    """Return the builder module for a job type, or None if unsupported."""
    if job_type == JobType.training.value:
        from amortized.jobs import training
        return training
    if job_type == JobType.sdg.value:
        from amortized.jobs import sdg
        return sdg
    if job_type == JobType.upload.value:
        from amortized.jobs import upload
        return upload
    return None
