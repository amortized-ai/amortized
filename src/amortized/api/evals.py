"""Evaluation comparisons — group eval jobs by dataset + metric set.

One evaluation group = one eval dataset evaluated with one metric set;
the group's entries are the models evaluated against it, joined with
their per-metric scores from MLflow run tags. This powers the Studio
Evaluation tab's cross-model comparison table.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType

logger = logging.getLogger("amortized.api.evals")

router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluations"])


def _tags(run: dict[str, Any]) -> dict[str, str]:
    return {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}


def _job_config(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job.get("config", {})
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except ValueError:
            cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def _metric_set_signature(cfg: dict[str, Any]) -> str:
    """Stable signature of the metrics + rubric criteria of an eval.

    Rubric criterion descriptions are part of the signature: same names
    with rewritten descriptions mean the judge scores different things,
    so those evals must not share a comparison group.
    """
    metrics = sorted(m for m in (cfg.get("metrics") or []) if m)
    rubric = sorted(
        (c.get("name", ""), c.get("description", "") or "")
        for c in (cfg.get("rubric") or [])
        if isinstance(c, dict) and c.get("name")
    )
    return json.dumps({"m": metrics, "r": rubric}, sort_keys=True)


def _semantic_config(cfg: dict[str, Any], run_tags: dict[str, str]) -> dict[str, Any]:
    """The config fields that change what an eval's numbers MEAN.

    Stored configs are NOT comparable directly: keys are absent when the
    submitter relied on defaults, the worker later injects serving
    plumbing (gpu_uuids, port, ...), and defaults drift across schema
    versions. So parse through the CURRENT schema (fills defaults) and
    keep only the semantic fields. For the sample counts prefer what
    actually ran (MLflow tags) over the config — legacy jobs predate the
    "0 = all" default and their absent key meant a different number.
    """
    from amortized.models import EvalJobConfig

    try:
        parsed = EvalJobConfig(**cfg)
        temperature = float(parsed.temperature)
        max_samples = int(parsed.max_samples)
        judge_max_samples = int(parsed.judge_max_samples)
    except Exception:
        temperature = float(cfg.get("temperature", 0.0) or 0.0)
        max_samples = int(cfg.get("max_samples", 0) or 0)
        judge_max_samples = int(cfg.get("judge_max_samples", 0) or 0)
    judge = cfg.get("judge") or {}
    judge_model = str(judge.get("model", "") or "") if isinstance(judge, dict) else ""
    # What actually ran beats what the config says.
    if run_tags.get("num_samples"):
        with _suppress():
            max_samples = int(run_tags["num_samples"])
    if run_tags.get("num_scored"):
        with _suppress():
            judge_max_samples = int(run_tags["num_scored"])
    elif run_tags.get("num_samples"):
        # No judge ran (no num_scored) — 0 stays 0.
        pass
    return {
        "temperature": temperature,
        "max_samples": max_samples,
        "judge_max_samples": judge_max_samples,
        "judge_model": judge_model,
    }


def _signature_names(signature: str) -> list[str]:
    try:
        parsed = json.loads(signature)
        names = list(parsed.get("m", []))
        # rubric entries are [name, description] pairs — keep the names
        for r in parsed.get("r", []):
            if isinstance(r, list) and r and r[0]:
                names.append(r[0])
            elif isinstance(r, str) and r:
                names.append(r)
        return [n for n in names if n]
    except ValueError:
        return []


async def _eval_runs_by_id() -> dict[str, dict[str, str]]:
    """MLflow runs of finished eval jobs, keyed by run id → tags."""
    tracking_uri = config_mod.settings.mlflow_tracking_uri
    if not tracking_uri:
        return {}
    client = MLflowClient(tracking_uri)
    exp_ids = await client.list_experiment_ids()
    if not exp_ids:
        return {}
    runs = await client.search_runs(
        exp_ids,
        filter_string="tags.job_type = 'eval'",
        max_results=500,
    )
    return {r.get("info", {}).get("run_id", ""): _tags(r) for r in runs}


def _entry_from_job(
    job: dict[str, Any], run_tags: dict[str, dict[str, str]]
) -> dict[str, Any]:
    cfg = _job_config(job)
    model = str(
        (cfg.get("endpoint") or {}).get("model")
        or (cfg.get("endpoint_tuned") or {}).get("model")
        or (cfg.get("endpoint_base") or {}).get("model")
        or ""
    )
    scores: dict[str, Any] = {}
    num_samples: int | None = None
    run_id = job.get("mlflow_run_id", "")
    tags = run_tags.get(run_id, {})
    if tags:
        if not model:
            model = tags.get("eval_model", "")
        if tags.get("eval_exact_match"):
            with _suppress():
                scores["exact_match"] = float(tags["eval_exact_match"])
        for key, value in tags.items():
            if key.startswith("eval_score_"):
                with _suppress():
                    scores[key[len("eval_score_"):]] = float(value)
        if tags.get("num_samples"):
            with _suppress():
                num_samples = int(tags["num_samples"])
    return {
        "job_id": job["id"],
        "model": model or "(unknown)",
        "status": job.get("status", ""),
        "created_at": job.get("created_at"),
        "mlflow_run_id": run_id,
        "scores": scores,
        "num_samples": num_samples,
        "topic": str(cfg.get("topic", "")),
        "config": _semantic_config(cfg, tags),
    }


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True  # swallow ValueError etc.


@router.get("")
async def list_evaluations(
    db: asyncpg.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Eval jobs grouped by (dataset, metric set) with per-model scores."""
    repo = Repository(db)
    eval_jobs = await repo.list_jobs(
        job_type=JobType.eval, k8s_namespace=config_mod.settings.compute_namespace
    )

    run_tags = await _eval_runs_by_id()

    # Dataset identity: parent SDG/upload job's MLflow run, else the
    # eval_data_run_id recorded in the config.
    dataset_run_by_job: dict[str, str] = {}
    dataset_name_by_run: dict[str, str] = {}
    parent_cache: dict[str, dict[str, Any] | None] = {}
    for job in eval_jobs:
        if job.get("status") != JobStatus.succeeded.value:
            continue
        cfg = _job_config(job)
        run_id = str(cfg.get("eval_data_run_id") or "")
        if not run_id:
            parent_id = job.get("parent_job_id", "")
            if parent_id and parent_id not in parent_cache:
                parent_cache[parent_id] = await repo.get_job(parent_id)
            parent = parent_cache.get(parent_id)
            if parent:
                run_id = parent.get("mlflow_run_id", "")
                dataset_name_by_run.setdefault(
                    run_id, str(_job_config(parent).get("topic", "") or run_id)
                )
        dataset_run_by_job[job["id"]] = run_id or "unknown"

    # Resolve dataset display names from the runs' own tags (sdg topic or
    # dataset_name), for groups keyed by eval_data_run_id.
    missing = {
        rid
        for rid in dataset_run_by_job.values()
        if rid not in dataset_name_by_run and rid != "unknown"
    }
    if missing:
        tracking_uri = config_mod.settings.mlflow_tracking_uri
        if tracking_uri:
            client = MLflowClient(tracking_uri)
            for rid in missing:
                try:
                    run = await client.get_run(rid)
                    tags = _tags(run)
                    dataset_name_by_run[rid] = (
                        tags.get("dataset_name")
                        or tags.get("dataset_topic")
                        or tags.get("topic")
                        or rid
                    )
                except Exception:
                    dataset_name_by_run[rid] = rid

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for job in sorted(eval_jobs, key=lambda j: j.get("created_at") or "", reverse=True):
        # Only succeeded evals are comparable — failed/cancelled runs have
        # no scores and would render as dead "(unknown)" columns.
        if job.get("status") != JobStatus.succeeded.value:
            continue
        cfg = _job_config(job)
        ds_run = dataset_run_by_job[job["id"]]
        sig = _metric_set_signature(cfg)
        key = (ds_run, sig)
        group = groups.setdefault(
            key,
            {
                "id": f"{ds_run[:8] if ds_run != 'unknown' else 'unknown'}-{abs(hash(key)) % 100000}",
                "dataset": {
                    "run_id": ds_run,
                    "name": dataset_name_by_run.get(ds_run, ds_run),
                },
                "metric_names": _signature_names(sig),
                "evals": [],
                "latest_created_at": None,
            },
        )
        group["evals"].append(_entry_from_job(job, run_tags))
        group["latest_created_at"] = job.get("created_at")

    # Only datasets that actually have evaluation results — a dataset whose
    # eval jobs all failed or predate the tag schema (no scores) is not
    # "evaluated" and should not appear as a comparison row.
    scored_groups = [
        g for g in groups.values() if any(e["scores"] for e in g["evals"])
    ]
    result = sorted(
        scored_groups,
        key=lambda g: g.get("latest_created_at") or "",
        reverse=True,
    )
    return {"groups": result}
