"""Base types for job builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from amortized.backends import Resources


class JobBuildError(Exception):
    """Raised when job building fails and the job should be marked failed."""


@dataclass
class JobBuildResult:
    """Output of a job builder — everything needed to construct a JobSpec."""

    command: list[str]
    config_files: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)
    image: str = ""
    resolved_config: dict[str, Any] = field(default_factory=dict)
    pre_commands: list[str] = field(default_factory=list)
    post_commands: list[str] = field(default_factory=list)


class JobBuilder(Protocol):
    """Protocol for job-type-specific builders."""

    async def build(
        self,
        job: dict[str, Any],
        config: dict[str, Any],
        config_files: dict[str, str],
    ) -> JobBuildResult:
        """Build the job spec from config. May raise to fail the job."""
        ...

    async def on_success(
        self,
        job: dict[str, Any],
        mlflow_run_id: str,
    ) -> None:
        """Post-completion hook — set MLflow tags, register models, etc."""
        ...
