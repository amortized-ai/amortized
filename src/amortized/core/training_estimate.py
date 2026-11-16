"""Run training-hub's memory estimator as a short job in the training image.

The server doesn't ship torch or training_hub, so the estimate runs where they are
installed: a CPU-only job on the configured compute backend that executes
jobs/assets/estimate_vram.py and prints its result as the last line of output.
All requests of one call share a job (the pod start dominates), and results are cached
per training image, since the estimate only depends on the request and training-hub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from amortized import config as config_mod
from amortized.backends import BackendHandle, JobSpec, Resources
from amortized.core.compute import get_backend

logger = logging.getLogger("amortized.core.training_estimate")

_SCRIPT = Path(__file__).resolve().parent.parent / "jobs" / "assets" / "estimate_vram.py"
_TIMEOUT_S = 300
_POLL_S = 2.0


_cache: dict[tuple[str, str], dict[str, Any]] = {}


class TrainingEstimateError(RuntimeError):
    """The estimate job failed, timed out, or printed no result."""


async def run_training_estimates(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estimate ``requests`` in at most one job. Each result is the estimate (values in
    bytes, plus ``training_hub_version``) or ``{"error": ...}`` for that request alone."""
    image = config_mod.job_image("training", config_mod.settings.resolved_training_image_tag)
    keys = [(image, json.dumps(r, sort_keys=True)) for r in requests]
    todo = {k: r for k, r in zip(keys, requests, strict=True) if k not in _cache}
    fresh: dict[tuple[str, str], dict[str, Any]] = {}
    if todo:
        output = await _run_job(image, list(todo.values()))
        for key, item in zip(todo, output["results"], strict=True):
            if "error" not in item:
                item = {**item, "training_hub_version": output["training_hub_version"]}
                _cache[key] = item
            fresh[key] = item
    return [fresh.get(k) or _cache[k] for k in keys]


async def _run_job(image: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
    backend = get_backend(config_mod.settings.resolved_default_backend)
    job_id = f"estimate-{uuid.uuid4().hex[:12]}"
    env = {
        name: value for name in config_mod.settings.forward_env if (value := os.environ.get(name))
    }
    spec = JobSpec(
        job_id=job_id,
        command=["python3", "/amortized/estimate_vram.py", json.dumps(requests)],
        env=env,
        work_dir=str(config_mod.settings.data_dir / "estimates" / job_id),
        image=image,
        resources=Resources(gpus=0, cpus=2, memory_gb=8),
        config_files={"estimate_vram.py": _SCRIPT.read_text(encoding="utf-8")},
        job_type="estimate",
    )

    handle: BackendHandle | None = None
    try:
        handle = await backend.submit(spec)
        deadline = time.monotonic() + _TIMEOUT_S
        while True:
            status = await backend.status(handle)
            if not status.running:
                break
            if time.monotonic() > deadline:
                raise TrainingEstimateError(f"estimate job timed out after {_TIMEOUT_S}s")
            await asyncio.sleep(_POLL_S)

        lines = [line async for line in backend.logs(handle)]
        result = _parse_result(lines)
        if result is not None and "results" not in result:
            raise TrainingEstimateError(result["error"])
        if (
            status.exit_code not in (0, None)
            or result is None
            or len(result["results"]) != len(requests)
        ):
            tail = "\n".join(line for line in lines if line.strip())[-1500:]
            reason = status.error or f"exit code {status.exit_code}"
            raise TrainingEstimateError(f"estimate job failed ({reason}):\n{tail}")
        return result
    except TrainingEstimateError:
        raise
    except Exception as exc:
        # Backend errors (k8s API, quota rejection, ...) become a readable 502, not a 500.
        raise TrainingEstimateError(f"estimate job could not run: {exc}") from exc
    finally:
        if handle is not None:
            try:
                await backend.cancel(handle)
            except Exception:
                logger.warning("could not clean up estimate job %s", job_id, exc_info=True)


_RESULT_KEYS = frozenset({"training_hub_version", "results"})


def _parse_result(lines: list[str]) -> dict[str, Any] | None:
    # Keep scanning past non-result brace lines: the local backend yields stderr after stdout,
    # so a dict-like stderr line can come after the real result in reversed order.
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (
            parsed.keys() >= _RESULT_KEYS or parsed.keys() == {"error"}
        ):
            return parsed
    return None
