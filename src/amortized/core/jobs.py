"""Job lifecycle domain logic — zero HTTP imports."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from amortized.backends import BackendHandle
from amortized.config import settings as _settings
from amortized.core.compute import get_backend
from amortized.core.placeholders import find_prompt_refs, find_unresolved_placeholders
from amortized.models import INLINE_API_KEY_ERROR, JobStatus, JobType

if TYPE_CHECKING:
    from amortized.db.repository import Repository

logger = logging.getLogger("amortized.core.jobs")

# embedding_sft (and its classifier/embedding aliases) trains in-process via
# SentenceTransformerTrainer — CPU-only, no torchrun, nproc_per_node ignored — so it requests
# 0 GPUs. Kept in sync with the embedding_sft branch in jobs/training.py (issue #486) and the
# ALGO_ALIASES there.
_CPU_TRAINING_ALGORITHMS = frozenset({"embedding_sft", "classifier", "embedding"})


def estimate_gpu_demand(job: dict[str, Any]) -> int:
    """Best-effort count of GPUs a job's pod will request, from its (type, config).

    The worker gates admission on the namespace GPU budget, but a job's real GPU
    count is only known after the (async, side-effecting) builder runs. This mirrors
    the per-builder ``Resources(gpus=...)`` so the worker can decide what fits WITHOUT
    building. Keep in sync with ``amortized.jobs.*``:
      - training, embedding_sft/classifier -> train_on_gpu ? 1 : 0 (single-device, in-process;
                                              CPU by default, 1-GPU opt-in; nproc_per_node ignored)
      - training, other algorithms         -> nproc_per_node (default 1)
      - sdg / upload     -> 0
      - eval, classification mode        -> 0 (CPU-only; nproc_per_node ignored)
      - eval, embedded serving (training_job_id / model_name_or_path, no endpoint)
                                          -> nproc_per_node (default 1)
      - eval, external endpoint          -> 0
    """
    config = job.get("config") or {}
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except (ValueError, TypeError):
            config = {}

    def _nproc(default: int) -> int:
        try:
            return max(0, int(config.get("nproc_per_node", default)))
        except (TypeError, ValueError):
            return default

    job_type = job.get("type")
    if job_type == JobType.training.value:
        if str(config.get("algorithm", "")).strip() in _CPU_TRAINING_ALGORITHMS:
            # embedding_sft trains in-process on a single device: CPU by default,
            # 1-GPU opt-in via train_on_gpu. nproc_per_node is ignored (no torchrun).
            return 1 if config.get("train_on_gpu") else 0
        return _nproc(1)
    if job_type == JobType.eval.value:
        if config.get("eval_mode") == "classification":
            return 0  # classification eval is CPU-only by design; nproc_per_node is ignored
        has_model_source = bool(
            str(config.get("training_job_id", "")).strip()
            or str(config.get("model_name_or_path", "")).strip()
        )
        if has_model_source and not config.get("endpoint"):
            return _nproc(1)
        return 0
    return 0


def gpu_budget_error(job_type: JobType, config: dict[str, Any]) -> str | None:
    """Error when the job needs more GPUs than the whole per-user budget, else None.

    Checked at validation too, so the agent sees it before the user confirms."""
    demand = estimate_gpu_demand({"type": job_type.value, "config": config})
    if demand <= _settings.gpu_budget:
        return None
    # Name the knob that actually drove the demand so the error is actionable:
    # embedding_sft/classifier reach it via train_on_gpu (a single GPU), every other
    # training algorithm (and embedded-serving eval) via nproc_per_node.
    budget = _settings.gpu_budget
    raise_budget = "raise the budget (GPU_PER_USER / AMORTIZED_GPU_BUDGET)"
    if (
        job_type == JobType.training
        and str(config.get("algorithm", "")).strip() in _CPU_TRAINING_ALGORITHMS
    ):
        knob, remedy = "train_on_gpu", f"set train_on_gpu=false to train on CPU or {raise_budget}"
    elif budget == 0:
        # nproc_per_node cannot go below 1, so with no GPU budget there is nothing to reduce.
        knob = "nproc_per_node"
        alternative = (
            "evaluate through an endpoint (a provider model or URL) instead of serving"
            " the model in the job"
            if job_type == JobType.eval
            else "only CPU training (embedding_sft) runs without GPUs"
        )
        remedy = f"{alternative}, or {raise_budget}"
    else:
        knob, remedy = "nproc_per_node", f"reduce nproc_per_node or {raise_budget}"
    return (
        f"job requests {demand} GPU(s) ({knob}) but the per-user GPU budget is {budget}"
        f" — {remedy}"
    )


def _url_carries_credentials(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return False
    return bool(parts.username or parts.password or parts.query)


def _api_key_paths(value: Any, path: str = "") -> list[str]:
    if isinstance(value, dict):
        found: list[str] = []
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            name = str(key).lower().replace("-", "_")
            if name in ("api_key", "apikey"):
                if item not in (None, ""):
                    found.append(child)
            elif name == "base_url" and _url_carries_credentials(item):
                found.append(child)
            else:
                found.extend(_api_key_paths(item, child))
        return found
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in _api_key_paths(item, f"{path}[{i}]")]
    return []


def inline_credential_error(config: dict[str, Any]) -> str | None:
    """Error when a job config carries an API key, else None.

    A config is stored, shown on the confirmation card and in the API, and copied by
    retries, so a key in it would leak through all of those."""
    paths = _api_key_paths(config)
    if not paths:
        return None
    return f"{', '.join(paths)}: {INLINE_API_KEY_ERROR}"


async def create_job(
    repo: Repository,
    *,
    job_type: JobType,
    config: dict[str, Any],
    recipe: str = "",
    parent_job_id: str = "",
    user_id: str = "",
    retry_of: str = "",
) -> dict[str, Any]:
    if not parent_job_id:
        parent_job_id = config.get("parent_job_id", "")

    credential_error = inline_credential_error(config)
    if credential_error:
        raise InvalidJobStateError(credential_error)

    # Placeholder tokens are agent-resolution hints, never valid values. The form
    # strips them, but the agent/API path could still submit one — reject it here
    # with a clear error instead of letting the job fail later at runtime.
    unresolved = find_unresolved_placeholders({**config, "parent_job_id": parent_job_id})
    if unresolved:
        raise InvalidJobStateError(
            "config contains unresolved placeholder token(s): "
            + ", ".join(unresolved)
            + " — resolve them (select real documents/models/datasets) before running"
        )

    # Prompt-store refs (#541) must resolve before the job is stored: a dangling
    # <prompt:...> ref would fail at the container boundary (build()) where the text
    # is materialized. Fail closed here with a clear error instead. The config is
    # stored ref-form (small, clone-safe); refs are expanded only at build time.
    prompt_refs = find_prompt_refs(config)
    if prompt_refs:
        missing = [
            ref for ref in prompt_refs if await repo.get_prompt_content(ref) is None
        ]
        if missing:
            raise InvalidJobStateError(
                "config references prompt(s) with no stored text: "
                + ", ".join(missing)
                + " — show_prompt stores a prompt and returns its reference"
            )

    # A job that needs more GPUs than the entire budget can never be admitted (the
    # worker would skip it forever), so reject it at creation with a clear error
    # instead of letting it sit silently queued.
    budget_error = gpu_budget_error(job_type, config)
    if budget_error:
        raise InvalidJobStateError(budget_error)

    job_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    stored_config = dict(config)
    if "parent_job_id" in stored_config:
        stored_config.pop("parent_job_id")

    row = await repo.create_job(
        job_id=job_id,
        job_type=job_type,
        config=stored_config,
        created_at=now,
        recipe=recipe,
        parent_job_id=parent_job_id,
        user_id=user_id,
        k8s_namespace=_settings.compute_namespace,
        retry_of=retry_of,
    )

    logger.info("Created %s job %s", job_type.value, job_id)
    return row


async def get_job(repo: Repository, job_id: str) -> dict[str, Any] | None:
    return await repo.get_job(job_id)


async def list_jobs(
    repo: Repository,
    *,
    status: JobStatus | None = None,
    job_type: JobType | None = None,
    k8s_namespace: str = "",
) -> list[dict[str, Any]]:
    return await repo.list_jobs(
        status=status, job_type=job_type, k8s_namespace=k8s_namespace,
    )


async def cancel_job(repo: Repository, job_id: str) -> dict[str, Any]:
    row = await repo.get_job(job_id)
    if row is None:
        raise JobNotFoundError(job_id)

    current_status = row["status"]
    if current_status in (JobStatus.succeeded.value, JobStatus.failed.value):
        raise InvalidJobStateError(f"Cannot cancel job with status '{current_status}'")
    if current_status == JobStatus.cancelled.value:
        return row

    now = datetime.now(UTC)

    if current_status == JobStatus.running.value:
        handle_json = row.get("backend_handle")
        cancelled = await cancel_job_via_backend(job_id, handle_json)
        if not cancelled:
            logger.warning("Could not cancel job %s via backend", job_id)

    updated = await repo.update_job(
        job_id,
        status=JobStatus.cancelled.value,
        completed_at=now,
    )
    logger.info("Cancelled job %s", job_id)
    assert updated is not None
    return updated


def deserialize_handle(raw: str | None) -> BackendHandle | None:
    if not raw:
        return None
    d = json.loads(raw)
    raw_secrets = d.get("secret_names")
    secret_names = [tuple(s) for s in raw_secrets] if raw_secrets else None
    return BackendHandle(
        backend_name=d["backend_name"],
        job_id=d["job_id"],
        remote_pid=d.get("remote_pid"),
        remote_dir=d.get("remote_dir"),
        container_id=d.get("container_id"),
        scheduler_id=d.get("scheduler_id"),
        secret_names=secret_names,
    )


async def cancel_job_via_backend(job_id: str, handle_json: str | None) -> bool:
    handle = deserialize_handle(handle_json)
    if handle is None:
        return False
    try:
        backend = get_backend(handle.backend_name)
        await backend.cancel(handle)
        return True
    except (KeyError, OSError):
        return False


class JobNotFoundError(Exception):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job {job_id} not found")


class InvalidJobStateError(Exception):
    pass


_TERMINAL_STATUSES = frozenset(
    {
        JobStatus.succeeded.value,
        JobStatus.failed.value,
        JobStatus.cancelled.value,
    }
)


async def delete_job(repo: Repository, job_id: str) -> None:
    row = await repo.get_job(job_id)
    if row is None:
        raise JobNotFoundError(job_id)
    if row["status"] not in _TERMINAL_STATUSES:
        raise InvalidJobStateError(
            f"Cannot delete job in state '{row['status']}' — cancel it first"
        )
    await repo.delete_job(job_id)
    logger.info("Deleted job %s", job_id)
