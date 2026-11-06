"""Compatible product views over engine-owned metrics and artifact references."""

import hashlib
import json
import math
from typing import Any

from amortized.db.evalhub import EvalHubRepository
from amortized.db.repository import Repository


def primary_metric(execution: dict[str, Any], benchmark_id: str) -> str:
    """The catalog metric that is the benchmark's score, as submitted to Hub.

    Jobs submitted before Amortized sent primary_score fall back to the catalog Hub held
    when the job was prepared, which is where Hub itself took it from.
    """
    sources = [
        (execution.get("request") or {}).get("benchmarks") or [],
        ((execution.get("inputs") or {}).get("catalog") or {}).get("benchmarks") or [],
    ]
    for benchmarks in sources:
        for benchmark in benchmarks:
            metric = (benchmark.get("primary_score") or {}).get("metric")
            if benchmark.get("id") == benchmark_id and metric:
                return str(metric)
    return ""


def present(execution: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = execution.get("snapshot") or {}
    benchmarks = (snapshot.get("results") or {}).get("benchmarks") or []
    if not benchmarks:
        return None
    primary: list[str] = []
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
        metric = primary_metric(execution, benchmark["id"])
        if metric:
            primary.append(prefix + metric)
        for name, raw in (benchmark.get("metrics") or {}).items():
            value = raw.get("value", raw.get("metric_value")) if isinstance(raw, dict) else raw
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
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
    # Exact name only, as Hub scores it: a catalog metric the adapter did not report gives
    # no score rather than a stand-in. Scores list the primary metric first.
    reported = [key for key in primary if key in scores]
    scores = {key: scores[key] for key in reported} | scores
    return {
        "num_records": task_summary.get("num_records"),
        "num_skipped": task_summary.get("num_skipped"),
        "model": task_summary.get("model") or {},
        **task_summary,
        "scores": scores,
        "primary_metric": primary[0] if len(primary) == 1 else None,
        "primary_score": scores.get(primary[0]) if len(primary) == 1 else None,
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
        execution = await store.get(job["id"])
        if execution is None:
            continue
        results = present(execution)
        if not results or not results["scores"]:
            continue
        from amortized.models import EvalJobConfig

        config = EvalJobConfig(**(job.get("config") or {})).model_dump(exclude_none=True)
        definition = config.get("evaluation") or {}
        # The recipe fingerprint covers everything that decides a benchmark score
        # (entry, adapter digest, settings) and leaves out how the model was served,
        # so a base endpoint and a tuned model share a group.
        pinned = definition.get("kind") == "benchmark" and definition.get("recipe_fingerprint")
        if pinned:
            dataset = "benchmark:" + str(definition["recipe_fingerprint"])
        elif definition.get("kind") == "benchmark":
            dataset = "benchmark:" + json.dumps(definition, sort_keys=True)
        else:
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
        semantics["catalog"] = (execution.get("inputs") or {}).get("catalog")
        if pinned:
            semantics = {key: None for key in semantics}
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
                    "name": definition.get("benchmark_id") or config.get("topic") or "Task dataset",
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
