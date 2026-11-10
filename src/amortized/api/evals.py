"""Evaluation comparisons — group eval jobs by dataset + metric set.

One evaluation group = one eval dataset evaluated with one metric set;
the group's entries are the models evaluated against it, joined with
their per-metric scores from MLflow run tags. This powers the Studio
Evaluation tab's cross-model comparison table.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import amortized.config as config_mod
from amortized.core.mlflow_client import MLflowClient
from amortized.db import get_db as _get_db
from amortized.db.repository import Repository
from amortized.models import JobStatus, JobType

if TYPE_CHECKING:
    from amortized.api.jobs import _SdgSignature

logger = logging.getLogger("amortized.api.evals")

router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluations"])


@router.get("/providers", operation_id="list_evaluation_providers")
async def list_evaluation_providers() -> dict[str, Any]:
    """Read enabled task scoring and benchmark providers before assembling an eval."""
    from amortized.integrations.evalhub.client import client

    settings = config_mod.settings
    if not settings.evalhub_url:
        return {"enabled": False, "task_scoring": ["rubric", "embedding_anchor_query"]}
    async with client() as sdk:
        task = await sdk.providers.get(settings.evalhub_task_provider)
        benchmark_providers = []
        for name in settings.evalhub_providers:
            provider = await sdk.providers.get(name)
            benchmark_providers.append(
                {"id": name, "name": provider.title, "benchmarks": len(provider.benchmarks)}
            )
    return {
        "enabled": True,
        "providers": [task.model_dump(mode="json", exclude_none=True)],
        "task_scoring": ["rubric", "embedding_anchor_query@1"],
        "benchmark_providers": benchmark_providers,
        "benchmarks": (
            "Do not choose a benchmark from this list. Settle the model first, then call"
            " find_benchmarks to look one up by name or browse, and get_benchmark for its"
            " settings."
        ),
    }


class BenchmarkModel(BaseModel):
    """The model the benchmark will evaluate; it decides which benchmarks can run."""

    training_job_id: str = Field("", description="A finished training job (Amortized serves it)")
    model_name_or_path: str = Field("", description="A base model Amortized serves")
    endpoint: dict[str, Any] | None = Field(
        None, description="An outside OpenAI-compatible endpoint (base_url, model)"
    )
    declared_capabilities: list[Literal["completions_logprobs"]] = Field(
        default_factory=list,
        description=(
            "Outside endpoint only, and only if the user confirmed it: the endpoint serves"
            " /v1/completions with echo and logprobs"
        ),
    )


class FindBenchmarksRequest(BaseModel):
    model: BenchmarkModel
    name: str = Field("", description="A benchmark name the user gave, to look up exactly")
    picks: dict[str, str] = Field(
        default_factory=dict, description="The picks from the previous find_benchmarks response"
    )
    answer: str = Field("", description="The user's reply to the previous question, verbatim")


class GetBenchmarkRequest(BaseModel):
    provider_id: str = Field(..., description="provider_id from a find_benchmarks result")
    id: str = Field(..., description="Benchmark id from a find_benchmarks result")
    model: BenchmarkModel


def _capability(model: BenchmarkModel) -> Any:
    from amortized.integrations.evalhub.catalog import Capability

    try:
        return Capability.from_model(model.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _read_catalog(
    provider_ids: list[str] | None, with_shortlists: bool
) -> tuple[list[Any], list[dict[str, Any]]]:
    from amortized.integrations.evalhub import catalog
    from amortized.integrations.evalhub.client import client

    if not config_mod.settings.evalhub_url:
        raise HTTPException(
            status_code=422, detail="Benchmark evaluation is not enabled on this deployment"
        )
    try:
        async with client() as sdk:
            entries = await catalog.load(sdk, provider_ids)
            shortlists: list[dict[str, Any]] = []
            if with_shortlists:
                try:
                    shortlists = await catalog.load_shortlists(sdk)
                except Exception:
                    # Shortlists only help navigation; browsing works without them.
                    logger.warning("Could not read Eval Hub collections", exc_info=True)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Could not read the Eval Hub catalog", exc_info=True)
        raise HTTPException(
            status_code=503, detail="Cannot read the Eval Hub benchmark catalog; retry"
        ) from exc
    return entries, shortlists


@router.post(
    "/benchmarks/find",
    operation_id="find_benchmarks",
    summary=(
        "Find a benchmark for a settled model: look one up by exact name, or browse with"
        " narrowing questions. Follow the response's `next` instruction. Never pick or"
        " submit a benchmark id that did not come from this tool's result."
    ),
)
async def find_benchmarks(request: FindBenchmarksRequest) -> dict[str, Any]:
    from amortized.integrations.evalhub import catalog

    capability = _capability(request.model)
    entries, shortlists = await _read_catalog(None, with_shortlists=not request.name)
    try:
        return catalog.find(
            entries,
            capability,
            name=request.name.strip(),
            picks=request.picks,
            answer=request.answer.strip(),
            shortlists=shortlists,
        )
    except catalog.ChoiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/benchmarks/get",
    operation_id="get_benchmark",
    summary=(
        "Get one benchmark's details, compatibility with the model, and the only settings"
        " the user may change. Use the provider_id and id from a find_benchmarks result."
    ),
)
async def get_benchmark(request: GetBenchmarkRequest) -> dict[str, Any]:
    from amortized.integrations.evalhub import catalog

    capability = _capability(request.model)
    if request.provider_id not in config_mod.settings.evalhub_providers:
        raise HTTPException(
            status_code=404,
            detail=f"No enabled benchmark provider {request.provider_id!r}; use find_benchmarks",
        )
    # Every enabled provider, so the answer can name the benchmark's copies elsewhere.
    entries, _ = await _read_catalog(None, with_shortlists=False)
    entry = next(
        (e for e in entries if e.provider.id == request.provider_id and e.id == request.id), None
    )
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Benchmark {request.id!r} is not in provider {request.provider_id!r};"
                " use an id from a find_benchmarks result"
            ),
        )
    result = catalog.detail(entry, capability, entries)
    if result["compatibility"] == "compatible":
        result["next"] = (
            "Ask the user only about these settings (the defaults apply if they have no"
            " preference), then call validate_eval_job with evaluation = {kind: benchmark,"
            " provider_id, benchmark_id: this id, parameters: only the settings the user"
            " changed} and the same model fields."
        )
    else:
        result["next"] = (
            "This benchmark cannot be approved for this model. Explain the reason; do not"
            " offer a different benchmark in its place."
        )
    return result


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
        with _Suppress():
            max_samples = int(run_tags["num_samples"])
    if run_tags.get("num_scored"):
        with _Suppress():
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
    job: dict[str, Any],
    run_tags: dict[str, dict[str, str]],
    training_labels: dict[str, str] | None = None,
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
        for key, value in tags.items():
            if key.startswith("eval_score_"):
                with _Suppress():
                    scores[key[len("eval_score_") :]] = float(value)
        if tags.get("num_samples"):
            with _Suppress():
                num_samples = int(tags["num_samples"])
    # Classification evals have no endpoint and don't tag eval_model, so derive
    # the model identity from the config — otherwise a base-model run and a
    # tuned-model run both fall back to "(unknown)" and get merged into one
    # column instead of sitting side by side.
    if not model:
        training_job_id = str(cfg.get("training_job_id") or "").strip()
        if training_job_id:
            labels = training_labels or {}
            model = labels.get(training_job_id, f"tuned-{training_job_id[:8]}")
        elif cfg.get("model_name_or_path"):
            model = str(cfg.get("model_name_or_path"))
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


class _Suppress:
    def __enter__(self) -> _Suppress:
        return self

    def __exit__(self, *exc: object) -> bool:
        return True  # swallow ValueError etc.


@router.get(
    "",
    operation_id="list_evaluations",
    summary=(
        "List completed evaluations grouped by dataset and metric set, with the"
        " models already scored in each. Use to check whether a model was already"
        " evaluated on a dataset before running a new eval."
    ),
)
async def list_evaluations(
    db: asyncpg.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Eval jobs grouped by (dataset, metric set) with per-model scores."""
    repo = Repository(db)
    eval_jobs = await repo.list_jobs(
        job_type=JobType.eval, k8s_namespace=config_mod.settings.compute_namespace
    )
    from amortized.integrations.evalhub.results import comparison_groups

    hub_groups = await comparison_groups(
        repo, [j for j in eval_jobs if j.get("execution_owner") == "evalhub"]
    )
    eval_jobs = [j for j in eval_jobs if j.get("execution_owner") != "evalhub"]

    run_tags = await _eval_runs_by_id()

    # Readable labels for tuned-model (classification) evals that reference a
    # training job — mirrors the training on_success naming so the eval column
    # shows e.g. "all-MiniLM-L6-v2-embedding_sft-1a2b3c4d" instead of a raw id.
    training_labels: dict[str, str] = {}
    train_ids = {
        tid
        for job in eval_jobs
        if (tid := str(_job_config(job).get("training_job_id") or "").strip())
    }
    for tid in train_ids:
        tjob = await repo.get_job(tid)
        if not tjob:
            continue
        tcfg = _job_config(tjob)
        base = str(tcfg.get("model_name_or_path") or tcfg.get("model_id") or "model")
        algo = str(tcfg.get("algorithm") or "sft")
        training_labels[tid] = f"{base.split('/')[-1]}-{algo}-{tid[:8]}"

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
                # Stable across server restarts (str hash is salted per
                # process) and collision-free: sha1 of the full key.
                "id": (
                    f"{ds_run[:8] if ds_run != 'unknown' else 'unknown'}"
                    f"-{hashlib.sha1(repr(key).encode()).hexdigest()[:12]}"
                ),
                "dataset": {
                    "run_id": ds_run,
                    "name": dataset_name_by_run.get(ds_run, ds_run),
                },
                "metric_names": _signature_names(sig),
                "evals": [],
                "latest_created_at": None,
            },
        )
        group["evals"].append(_entry_from_job(job, run_tags, training_labels))
        # Jobs iterate newest-first; only the first (newest) one sets the
        # timestamp — later, older jobs must not clobber it. Prefer
        # completed_at: the eval finished then, not when it was queued.
        if group["latest_created_at"] is None:
            group["latest_created_at"] = job.get("completed_at") or job.get("created_at")

    # Only datasets that actually have evaluation results — a dataset whose
    # eval jobs all failed or predate the tag schema (no scores) is not
    # "evaluated" and should not appear as a comparison row.
    scored_groups = [g for g in groups.values() if any(e["scores"] for e in g["evals"])]
    result = sorted(
        scored_groups,
        key=lambda g: g.get("latest_created_at") or "",
        reverse=True,
    )
    return {
        "groups": sorted(
            [*result, *hub_groups], key=lambda g: g.get("latest_created_at") or "", reverse=True
        )
    }


async def _training_recipe_signature(
    repo: Repository, training_job_id: str
) -> _SdgSignature | None:
    """The (teacher, prompts) SDG recipe signature of a trained model's task.

    A tuned model is only comparable on the task it was trained for, so the
    recipe of its training data (teacher model + output/system prompt) is
    what makes a prior eval set "the same task". None when the model did not
    chain from an SDG job (trained from an upload/split) — then no eval set can
    be recipe-matched and the caller falls back to offering all scored sets."""
    from amortized.api.jobs import _sdg_signature

    training = await repo.get_job(training_job_id)
    if not training:
        return None
    train_sdg_id = str(training.get("parent_job_id") or "")
    if not train_sdg_id:
        return None
    sdg = await repo.get_job(train_sdg_id)
    if not sdg:
        return None
    return _sdg_signature(sdg.get("config"))


@router.get(
    "/reusable",
    operation_id="list_reusable_eval_sets",
    summary=(
        "List existing eval sets a TRAINED model can be scored on by REUSE,"
        " instead of generating a fresh one. Reusing a set another model was"
        " already scored on makes the scores directly comparable in the"
        " Evaluation tab. Each set is flagged `recipe_match=true` when it was"
        " generated by the same SDG recipe (teacher + output prompt) as the"
        " model's training data — those are the fair, same-task sets to offer"
        " first. Pass the chosen `eval_data_run_id` to validate_eval_job to"
        " reuse it; only fall back to clone_sdg_config_for_eval (fresh set)"
        " when the user wants a brand-new held-out set."
    ),
)
async def list_reusable_eval_sets(
    training_job_id: str,
    db: asyncpg.Connection = Depends(_get_db),
) -> dict[str, Any]:
    """Prior eval datasets a trained model can reuse, flagged by task match.

    Groups succeeded evals by their eval dataset (MLflow run), annotating each
    with the sibling models already scored on it and whether its generating SDG
    recipe matches the model's training recipe. Lets the eval workflow offer
    deterministic, comparable reuse options rather than always regenerating."""
    repo = Repository(db)
    train_sig = await _training_recipe_signature(repo, training_job_id)

    eval_jobs = await repo.list_jobs(
        job_type=JobType.eval, k8s_namespace=config_mod.settings.compute_namespace
    )
    run_tags = await _eval_runs_by_id()

    from amortized.api.jobs import _sdg_signature

    parent_cache: dict[str, dict[str, Any] | None] = {}

    async def _parent(pid: str) -> dict[str, Any] | None:
        if not pid:
            return None
        if pid not in parent_cache:
            parent_cache[pid] = await repo.get_job(pid)
        return parent_cache[pid]

    groups: dict[str, dict[str, Any]] = {}
    # Newest first so the group's latest_created_at is set by its most recent eval.
    for job in sorted(eval_jobs, key=lambda j: j.get("created_at") or "", reverse=True):
        if job.get("status") != JobStatus.succeeded.value:
            continue
        entry = _entry_from_job(job, run_tags)
        if job.get("execution_owner") == "evalhub":
            from amortized.db.evalhub import EvalHubRepository
            from amortized.integrations.evalhub.results import present

            execution = await EvalHubRepository(repo).get(job["id"])
            view = present(execution, job) if execution else None
            entry["scores"] = (view or {}).get("scores") or {}
        # Only sets with real scores are reusable comparison baselines.
        if not entry["scores"]:
            continue
        cfg = _job_config(job)
        parent_id = str(job.get("parent_job_id") or "")
        parent = await _parent(parent_id)
        run_id = str(cfg.get("eval_data_run_id") or "")
        if not run_id and parent:
            run_id = str(parent.get("mlflow_run_id") or "")
        if not run_id:
            continue  # cannot be addressed for reuse without a dataset run id

        grp = groups.get(run_id)
        if grp is None:
            grp = groups[run_id] = {
                "eval_data_run_id": run_id,
                "name": run_id,
                "models_scored": [],
                "metric_names": [],
                "recipe_match": False,
                "recipe_known": False,
                "latest_created_at": None,
            }
        # Recipe + display name come from the dataset's generating SDG parent;
        # the first eval in the group that resolves one fixes them (a later
        # reuse eval of the same set carries only eval_data_run_id, no parent).
        if parent and not grp["recipe_known"]:
            sig = _sdg_signature(parent.get("config"))
            if sig is not None:
                grp["recipe_known"] = True
                grp["recipe_match"] = train_sig is not None and sig == train_sig
            topic = str(_job_config(parent).get("topic", "") or "")
            if topic and grp["name"] == run_id:
                grp["name"] = topic
        model = entry["model"]
        if model and model != "(unknown)" and model not in grp["models_scored"]:
            grp["models_scored"].append(model)
        for name in _signature_names(_metric_set_signature(cfg)):
            if name not in grp["metric_names"]:
                grp["metric_names"].append(name)
        for name in entry["scores"]:
            if name not in grp["metric_names"]:
                grp["metric_names"].append(name)
        if grp["latest_created_at"] is None:
            grp["latest_created_at"] = job.get("completed_at") or job.get("created_at")

    eval_sets = sorted(
        groups.values(),
        # Same-task (recipe_match) sets first, then most recently used.
        key=lambda g: (g["recipe_match"], g.get("latest_created_at") or ""),
        reverse=True,
    )
    return {
        "training_job_id": training_job_id,
        "training_recipe_known": train_sig is not None,
        "recipe_match_available": any(g["recipe_match"] for g in eval_sets),
        "eval_sets": eval_sets,
    }
