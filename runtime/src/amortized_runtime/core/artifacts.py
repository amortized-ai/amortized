"""Artifact registration domain logic — zero HTTP imports."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from amortized_runtime.db.repository import Repository

logger = logging.getLogger("amortized_runtime.core.artifacts")

ARTIFACT_PATTERNS: dict[str, list[str]] = {
    "adapter_weights": ["adapter_model.safetensors", "adapter_model.bin"],
    "adapter_config": ["adapter_config.json"],
    "training_metrics": ["training_metrics.jsonl"],
    "tokenizer": [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ],
    "generated_data": ["*.jsonl", "*.parquet"],
}


async def register_artifacts_for_job(
    repo: Repository, job_id: str, output_dir: str
) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    now = datetime.now(UTC).isoformat()
    registered: list[dict[str, Any]] = []

    async def _register(artifact_type: str, file_path: Path) -> None:
        artifact = await repo.create_artifact(
            artifact_id=str(uuid.uuid4()),
            job_id=job_id,
            artifact_type=artifact_type,
            path=str(file_path),
            size=file_path.stat().st_size,
            created_at=now,
        )
        registered.append(artifact)

    for artifact_type, patterns in ARTIFACT_PATTERNS.items():
        for pattern in patterns:
            if "*" in pattern:
                for file_path in output_path.glob(pattern):
                    if file_path.is_file():
                        await _register(artifact_type, file_path)
            else:
                file_path = output_path / pattern
                if file_path.is_file():
                    await _register(artifact_type, file_path)

    # Scan checkpoint-N/ subdirectories
    for subdir in sorted(output_path.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("checkpoint-"):
            continue
        for artifact_type, patterns in ARTIFACT_PATTERNS.items():
            if artifact_type == "generated_data":
                continue
            for pattern in patterns:
                if "*" in pattern:
                    continue
                sub_file = subdir / pattern
                if sub_file.is_file():
                    await _register(artifact_type, sub_file)

    # SDG checkpoints subdirectory
    checkpoint_dir = output_path / "checkpoints"
    if checkpoint_dir.exists():
        for file_path in checkpoint_dir.glob("*.jsonl"):
            if file_path.is_file():
                await _register("checkpoint", file_path)

    logger.info("Registered %d artifacts for job %s", len(registered), job_id)
    return registered


async def register_log_artifacts(
    repo: Repository, job_id: str, output_dir: str
) -> list[dict[str, Any]]:
    now = datetime.now(UTC).isoformat()
    registered: list[dict[str, Any]] = []
    for log_name in ("stdout.log", "stderr.log"):
        log_path = Path(output_dir) / log_name
        if log_path.is_file() and log_path.stat().st_size > 0:
            artifact = await repo.create_artifact(
                artifact_id=str(uuid.uuid4()),
                job_id=job_id,
                artifact_type="log",
                path=str(log_path),
                size=log_path.stat().st_size,
                created_at=now,
            )
            registered.append(artifact)
    return registered


async def list_artifacts(
    repo: Repository, job_id: str
) -> list[dict[str, Any]]:
    return await repo.list_artifacts(job_id)
