"""Field-driven job-dependency enforcement.

Amortized's pipeline is a small chain of job types: an SDG job produces a dataset, a
training job consumes it and produces a model, an eval job consumes both. The one
invariant that must hold is a *local precondition*: a job may launch only once the
upstream jobs its config *references* have ``succeeded``. It says nothing about what
else happened — extra steps, loops, re-runs, out-of-order exploration are all fine.

So enforcement is **field-driven**. ``GATED_FIELDS`` is the universe of config keys
that reference an upstream *job* by id; ``present_gated_fields(config)`` returns the
ones actually present on the job being launched. Enforcement lives at a single place —
the ``validate_*`` tool boundary: the fail-closed validators in ``api/jobs.py`` call
it so a downstream job cannot be *created* (nor its config *validated*) until every
upstream job it references is ``succeeded``. A weak driver reads the ``422`` as a ReAct
observation and self-corrects; there is no separate reactive proxy gate.
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
    job: Mapping[str, Any], fetch: JobFetch
) -> AsyncIterator[Mapping[str, Any]]:
    """Yield each upstream job once, following every gated reference field each job
    carries (``GATED_FIELDS``), breadth-first (nearest first) and cycle-safe. Replaces the
    one-hop, per-call-site parent walks: the chain (eval -> training -> sdg) is traversed in
    full. ``job`` is a job row; ``fetch(id)`` resolves an id to a row (or None).

    Breadth-first so a consumer that wants the *closest* match of some type can stop at the
    first hit - see ``nearest_ancestor``."""
    seen: set[str] = set()
    frontier: list[Mapping[str, Any]] = [job]
    while frontier:
        next_frontier: list[Mapping[str, Any]] = []
        for node in frontier:
            for field in GATED_FIELDS:
                up_id = job_ref(node, field)
                if not up_id or up_id in seen:
                    continue
                seen.add(up_id)
                parent = await fetch(up_id)
                if parent is None:
                    continue
                yield parent
                next_frontier.append(parent)
        frontier = next_frontier


async def nearest_ancestor(
    job: Mapping[str, Any], job_types: str | Collection[str], fetch: JobFetch
) -> Mapping[str, Any] | None:
    """The closest upstream job whose type is in ``job_types`` (multi-hop), or None.
    e.g. ``nearest_ancestor(eval, "sdg", fetch)`` finds the eval's dataset SDG even if it is
    reached through the training job.

    Reuses the breadth-first ``ancestors`` walk and stops at the first hit, so "closest" is
    by hop distance: a one-hop match (e.g. an SDG referenced directly via ``data_run_id``)
    wins over a two-hop match reached through a parent."""
    wanted = {job_types} if isinstance(job_types, str) else set(job_types)
    async for a in ancestors(job, fetch):
        if str(a.get("type") or "") in wanted:
            return a
    return None
