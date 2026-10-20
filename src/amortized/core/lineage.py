"""Field-driven job-dependency enforcement.

Amortized's pipeline is a small chain of job types: an SDG job produces a dataset, a
training job consumes it and produces a model, an eval job consumes both. The one
invariant that must hold is a *local precondition*: a job may launch only once the
upstream jobs its config *references* have ``succeeded``. It says nothing about what
else happened — extra steps, loops, re-runs, out-of-order exploration are all fine.

So enforcement is **field-driven**. ``GATED_FIELDS`` is the universe of config keys
that reference an upstream *job* by id; ``present_gated_fields(config)`` returns the
ones actually present on the job being launched. Both enforcement layers call it:
  * the fail-closed validators in ``api/jobs.py`` (Layer 1 — a downstream job cannot
    be *created* until every upstream job it references is ``succeeded``);
  * the proxy stage-gate in ``api/agent.py`` (Layer 2 — Morty cannot *advance* a
    dispatch whose upstream is still in flight).
Because it keys on the fields present, not on a modelled job type, a type "beyond the
graph" is still gated on exactly the prerequisites it names, and nothing forces a
global sdg → training → eval sequence. Fields that reference an MLflow *run*
(``data_run_id`` / ``eval_data_run_id``) are intentionally absent: a run is an
artifact, not a job, and carries no job-ordering gate.

This module also offers an upstream *traversal* (``ancestors`` / ``nearest_ancestor``)
that follows those same reference fields to walk a job's lineage chain.

Pure: no FastAPI / opencode / DB deps, so both the API routes and the proxy can import
it. The DB status check itself stays in the callers; this module only owns *which*
fields are gated references and how to walk them.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from typing import Any

# A referenced upstream job counts as ready only once it reaches this status.
READY_STATUS = "succeeded"

# Job types that materialize a dataset artifact in MLflow (the thing a downstream
# training/eval job downloads). The dataset edge (parent_job_id) resolves to one of
# these. One place owns the fact, shared by the create-time resolvers and the worker's
# parent-artifact download.
DATASET_PRODUCER_TYPES = frozenset({"sdg", "upload"})


# The field-driven enforcement universe: every config key that references an upstream
# *job* (by id), with a generic label for gate messages. Enforcement keys on THIS, so
# any job — a modelled type or one "beyond the graph" — is gated on whatever of these
# it carries. Fields that reference an MLflow *run* (data_run_id / eval_data_run_id)
# are intentionally absent: a run is an artifact, not a job.
GATED_FIELDS: dict[str, str] = {
    "parent_job_id": "the upstream job that produced this job's input",
    "training_job_id": "the model's training job",
}


def present_gated_fields(config: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """The gated upstream references actually present on a job's config, as
    ``(field, job_id, label)``. Both enforcement layers call this, so "a job launches
    only once the prerequisites it references have succeeded" is defined in one place
    and applies to any job type."""
    out: list[tuple[str, str, str]] = []
    for field, label in GATED_FIELDS.items():
        job_id = str(config.get(field, "") or "")
        if job_id:
            out.append((field, job_id, label))
    return out


# ---------------------------------------------------------------------------
# Ancestry traversal (walk the graph upstream)
# ---------------------------------------------------------------------------

# A fetch callable maps a job id to its row (or None). Injected so this module stays
# DB-free; the API routes and worker pass a Repository-backed lookup (e.g. repo.get_job).
JobFetch = Callable[[str], Awaitable[Mapping[str, Any] | None]]


def _coerce_config(cfg: Any) -> Mapping[str, Any]:
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except ValueError:
            return {}
    return cfg if isinstance(cfg, Mapping) else {}


def job_ref(job: Mapping[str, Any], field: str) -> str:
    """Read an upstream-reference field from a job row, whether it lives as a top-level
    column (``parent_job_id``) or inside the config JSON (``training_job_id``,
    ``data_run_id``). One reader so callers stop special-casing where an id is stored."""
    val = job.get(field)
    if not val:
        val = _coerce_config(job.get("config")).get(field)
    return str(val or "")


async def ancestors(
    job: Mapping[str, Any], fetch: JobFetch, *, _seen: set[str] | None = None
) -> AsyncIterator[Mapping[str, Any]]:
    """Yield each upstream job once, following every gated reference field each job
    carries (``GATED_FIELDS``), depth-first and cycle-safe. Replaces the one-hop,
    per-call-site parent walks: the chain (eval → training → sdg) is traversed in full.
    ``job`` is a job row; ``fetch(id)`` resolves an id to a row (or None)."""
    seen = _seen if _seen is not None else set()
    for field in GATED_FIELDS:
        up_id = job_ref(job, field)
        if not up_id or up_id in seen:
            continue
        seen.add(up_id)
        parent = await fetch(up_id)
        if parent is None:
            continue
        yield parent
        async for a in ancestors(parent, fetch, _seen=seen):
            yield a


async def nearest_ancestor(
    job: Mapping[str, Any], job_types: str | Collection[str], fetch: JobFetch
) -> Mapping[str, Any] | None:
    """The closest upstream job whose type is in ``job_types`` (multi-hop), or None.
    e.g. ``nearest_ancestor(eval, "sdg", fetch)`` finds the eval's dataset SDG even if
    it is reached through the training job."""
    wanted = {job_types} if isinstance(job_types, str) else set(job_types)
    async for a in ancestors(job, fetch):
        if str(a.get("type") or "") in wanted:
            return a
    return None
