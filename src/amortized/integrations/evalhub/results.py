"""Compatible product views over engine-owned metrics and artifact references."""

import hashlib
import json
import logging
import math
from typing import Any

from pydantic import ValidationError

from amortized.db.evalhub import EvalHubRepository
from amortized.db.repository import Repository

logger = logging.getLogger(__name__)


def _runtime_image(provider: dict[str, Any]) -> str:
    """The adapter image in Hub's provider record."""
    runtime = (provider.get("runtime") or {}).get("k8s") or {}
    # Hub 1.0.3 serializes the Go field as "Image"; provider YAML uses "image".
    return str(runtime.get("Image") or runtime.get("image") or "")


def _engine(execution: dict[str, Any]) -> dict[str, Any]:
    """What in Hub's provider record decides a score: the provider, its adapter build and
    the selected entry. Not timestamps, resource limits or descriptions."""
    provider = (execution.get("inputs") or {}).get("catalog") or {}
    image = _runtime_image(provider)
    selected = {b.get("id") for b in (execution.get("request") or {}).get("benchmarks") or []}
    return {
        "provider_id": (provider.get("resource") or {}).get("id"),
        "adapter": image.split("@", 1)[1] if "@sha256:" in image else image,
        "benchmarks": [
            {"id": benchmark.get("id")}
            for benchmark in provider.get("benchmarks") or []
            if benchmark.get("id") in selected
        ],
    }


def _numeric(raw: Any) -> float | int | None:
    value = raw.get("value", raw.get("metric_value")) if isinstance(raw, dict) else raw
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return value


def present(execution: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = execution.get("snapshot") or {}
    benchmarks = (snapshot.get("results") or {}).get("benchmarks") or []
    if not benchmarks:
        return None
    scores: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    counts: dict[str, int] = {}
    task_summary: dict[str, Any] = {}
    for benchmark in benchmarks:
        info = benchmark.get("additional_info") or {}
        summary = info.get("amortized_results") or {}
        if summary:
            task_summary.update(summary)
        prefix = f"{benchmark['id']}/" if len(benchmarks) > 1 else ""
        for name, raw in (benchmark.get("metrics") or {}).items():
            value = _numeric(raw)
            if value is None:
                continue
            key = prefix + name
            scores[key] = value
            metadata[key] = (summary.get("metric_metadata") or {}).get(
                name,
                {
                    "unit": "value",
                    "direction": None,
                    "range": None,
                },
            )
            if name in (summary.get("scores_n") or {}):
                counts[key] = summary["scores_n"][name]
    return {
        "num_records": task_summary.get("num_records"),
        "num_skipped": task_summary.get("num_skipped"),
        "model": task_summary.get("model") or {},
        **task_summary,
        "scores": scores,
        "scores_n": counts,
        "metric_metadata": metadata,
        "benchmarks": benchmarks,
        "remote_id": execution.get("remote_id"),
        "remote_state": (snapshot.get("status") or {}).get("state"),
        "observed_at": str(execution.get("observed_at") or ""),
    }


async def comparison_groups(repo: Repository, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not jobs:
        return []
    groups: dict[str, Any] = {}
    store = EvalHubRepository(repo)
    for job in jobs:
        # Like the native groups: a failed or cancelled run's partial scores are no result.
        if job.get("status") != "succeeded":
            continue
        execution = await store.get(job["id"])
        if execution is None:
            continue
        results = present(execution)
        if not results or not results["scores"]:
            continue
        from amortized.models import EvalJobConfig

        stored = job.get("config") or {}
        try:
            config = EvalJobConfig(**stored).model_dump(exclude_none=True)
        except ValidationError:
            # Written by a build with other rules, e.g. a benchmark eval read after a rollback.
            # Hiding the run would tell Morty the model was never evaluated.
            logger.warning(
                "Eval %s: config doesn't parse in this version; compared as stored", job["id"]
            )
            config = stored
        dataset = str((execution.get("inputs") or {}).get("dataset_identity") or "")
        if not dataset:
            continue
        semantics = {
            key: config.get(key)
            for key in (
                "evaluation",
                "rubric",
                "temperature",
                "max_samples",
                "judge_max_samples",
                "eval_mode",
                "anchors_per_class",
                "top_k",
                "tau",
                "seed",
                "text_column",
                "label_column",
                "class_labels",
            )
        }
        judge = config.get("judge") or {}
        semantics["judge"] = {key: judge.get(key) for key in ("base_url", "model")}
        semantics["catalog"] = _engine(execution)
        semantics["metric_metadata"] = {
            name: {
                k: v
                for k, v in meta.items()
                if k not in {"num_samples", "num_invalid", "num_total"}
            }
            for name, meta in results["metric_metadata"].items()
        }
        key = hashlib.sha256(json.dumps([dataset, semantics], sort_keys=True).encode()).hexdigest()
        group = groups.setdefault(
            key,
            {
                "id": f"hub-{key[:20]}",
                "dataset": {
                    "run_id": str((execution.get("inputs") or {}).get("dataset_run_id") or ""),
                    "identity": dataset,
                    "name": config.get("topic") or "Task dataset",
                },
                "metric_names": list(results["scores"]),
                "metric_metadata": results["metric_metadata"],
                "evals": [],
                "latest_created_at": job.get("completed_at") or job.get("created_at"),
            },
        )
        group["evals"].append(
            {
                "job_id": job["id"],
                "model": (config.get("endpoint") or {}).get("model")
                or config.get("served_model_name")
                or config.get("training_job_id")
                or config.get("model_name_or_path"),
                "status": job["status"],
                "created_at": job.get("created_at"),
                "mlflow_run_id": job.get("mlflow_run_id", ""),
                "scores": results["scores"],
                "num_samples": results.get("model", {}).get("num_samples"),
                "topic": config.get("topic", ""),
                "config": {
                    "temperature": config.get("temperature", 0),
                    "max_samples": config.get("max_samples", 0),
                    "judge_max_samples": config.get("judge_max_samples", 0),
                    "judge_model": judge.get("model", ""),
                },
            }
        )
    return list(groups.values())
