"""The canonical job-dependency DAG.

Amortized's pipeline is a small directed graph of job types: an SDG job produces a
dataset, a training job consumes it and produces a model, an eval job consumes both
the dataset and the model. Those edges are already encoded, implicitly, in several
places:
  * the fail-closed validators in ``api/jobs.py`` (Layer 1 — a downstream job cannot
    be *created* until every upstream job it references is ``succeeded``);
  * the proxy stage-gate in ``api/agent.py`` (Layer 2 — Morty cannot *advance* a
    dispatch whose upstream is still in flight);
  * the prose of the workflow prompts under ``agents/`` (soft guidance to the agent).

This module makes the graph explicit and authoritative. It encodes a *local
precondition*, not a prescribed path: a node may launch iff the upstream jobs its
config *references* have ``succeeded``. It says nothing about what else happened —
extra steps, loops, re-runs, out-of-order exploration are all fine. So enforcement is
**field-driven** (``GATED_FIELDS`` / ``present_gated_fields``): both layers gate
whichever upstream-reference fields are present on the job being launched, regardless
of its type. A job type not in ``PIPELINE`` ("beyond the graph") is still gated on the
prerequisites it names — and nothing forces a global sdg→training→eval sequence.

``PIPELINE`` is the richer *structure* layered on top: per-node producer edges, the
topological order, and the orchestrator capability headings. It drives the
prompt-consistency contract (``check_workflow_consistency``) and the delegation
advisory's producer scoping — it is **not** a runtime path gate (``pipeline_order``
is used only by the consistency test, never to constrain a user action).

Pure: no FastAPI / opencode / DB deps, so both the API routes and the proxy can
import it. The DB status check itself stays in the callers; this module only owns
*which* fields are gated references, their producing job type, and the topology.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A referenced upstream job counts as ready only once it reaches this status.
READY_STATUS = "succeeded"

# Job types that materialize a dataset artifact in MLflow (the thing a downstream
# training/eval job downloads). The dataset edge (parent_job_id) resolves to one of
# these. One place owns the fact, shared by the create-time resolvers and the worker's
# parent-artifact download.
DATASET_PRODUCER_TYPES = frozenset({"sdg", "upload"})


@dataclass(frozen=True)
class Upstream:
    """One dependency edge: a config field on the downstream job carrying the id of
    an upstream job that must have produced its output first.

    ``field``    — the config key holding the upstream job id (e.g. ``parent_job_id``).
    ``producer`` — the job type expected to have produced it (documentation + the
                   edge direction for the pipeline topo-sort).
    ``label``    — human phrase for gate / advisory messages.
    ``gated``    — whether the stage-gate applies: the downstream job cannot be
                   created or advanced until this upstream is ``succeeded``. All
                   current edges are gated; the flag leaves room for a future
                   advisory-only reference.
    """

    field: str
    producer: str
    label: str
    gated: bool = True


@dataclass(frozen=True)
class JobNode:
    """A node in the pipeline DAG: a job type, the agent tool that dispatches it, the
    capability heading the orchestrator prompt presents it under, and its upstream
    edges."""

    job_type: str
    validate_tool: str
    orchestrator_phrase: str
    upstreams: tuple[Upstream, ...] = ()


# The pipeline DAG. Edges point from a downstream job to the upstream job(s) it
# depends on. ``data_run_id`` / ``eval_data_run_id`` are MLflow run ids (not jobs),
# so they are NOT edges here — they are validated downstream as data availability,
# not gated as job ordering. The eval dataset parent may be an SDG *or* an upload
# job; ``producer`` records the canonical SDG chain, but the gate only checks the
# referenced job's status, not its type.
PIPELINE: dict[str, JobNode] = {
    "sdg": JobNode(
        job_type="sdg",
        validate_tool="validate_sdg_job",
        orchestrator_phrase="Synthetic data generation",
        upstreams=(),
    ),
    "training": JobNode(
        job_type="training",
        validate_tool="validate_training_job",
        orchestrator_phrase="Model training",
        upstreams=(Upstream("parent_job_id", "sdg", "the training dataset's job"),),
    ),
    "eval": JobNode(
        job_type="eval",
        validate_tool="validate_eval_job",
        orchestrator_phrase="Model evaluation",
        upstreams=(
            Upstream("parent_job_id", "sdg", "the eval dataset's job"),
            Upstream("training_job_id", "training", "the model's training job"),
        ),
    ),
}


# The field-driven enforcement universe: every config key that references an upstream
# *job* (by id), with a generic label for gate messages. Enforcement keys on THIS, not
# on PIPELINE membership, so a job type "beyond the graph" is still gated on whatever
# of these it carries. Fields that reference an MLflow *run* (data_run_id /
# eval_data_run_id) are intentionally absent: a run is an artifact, not a job, and
# carries no job-ordering gate. Kept in sync with PIPELINE's edges by a test.
GATED_FIELDS: dict[str, str] = {
    "parent_job_id": "the upstream job that produced this job's input",
    "training_job_id": "the model's training job",
}


def present_gated_fields(config: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """The gated upstream references actually present on a job's config, as
    ``(field, job_id, label)``. Both enforcement layers call this, so "a node launches
    only once the prerequisites it references have succeeded" is defined in one place
    and applies to any job type, declared in PIPELINE or not."""
    out: list[tuple[str, str, str]] = []
    for field, label in GATED_FIELDS.items():
        job_id = str(config.get(field, "") or "")
        if job_id:
            out.append((field, job_id, label))
    return out


def upstreams(job_type: str) -> tuple[Upstream, ...]:
    """Every declared upstream edge for ``job_type`` (empty for roots/unknown)."""
    node = PIPELINE.get(job_type)
    return node.upstreams if node else ()


def gated_upstreams(job_type: str) -> tuple[Upstream, ...]:
    """The declared upstream edges for ``job_type`` subject to the stage-gate. Used by
    the prompt-consistency check and tests; runtime enforcement is field-driven via
    ``present_gated_fields`` (which also covers types not in PIPELINE)."""
    return tuple(u for u in upstreams(job_type) if u.gated)


def dispatch_parent_type() -> dict[str, str]:
    """``{validate_tool: downstream_job_type}`` for the proxy's stale-reuse (dispatch)
    check. Only nodes with a gated upstream appear: a root (SDG) has no prerequisite,
    so cloning a recipe for an eval set may run while training is still in flight."""
    return {n.validate_tool: n.job_type for n in PIPELINE.values() if gated_upstreams(n.job_type)}


_DISPATCH_TOOL_RE = re.compile(r"^validate_(?P<type>\w+)_job$")


def dispatch_job_type(tool_name: str) -> str | None:
    """The job type a ``validate_<type>_job`` dispatch tool launches, or None if the
    tool is not a dispatch or launches a *known root* (a PIPELINE node with no
    upstream — e.g. SDG, whose eval-set prep is allowed to run alongside training).

    An unknown type (not in PIPELINE) returns the type, not None: a dispatch "beyond
    the graph" is still stage-gated on whatever prerequisites it references."""
    m = _DISPATCH_TOOL_RE.match(tool_name)
    if not m:
        return None
    jt = m.group("type")
    node = PIPELINE.get(jt)
    if node is not None and not node.upstreams:
        return None  # a declared root — no prerequisite, never stage-gated
    return jt


def pipeline_order() -> list[str]:
    """Job types in dependency order (a stable topological sort): every type appears
    after all of its producers. For the current graph this is sdg → training → eval.
    Raises on a cycle (the DAG must stay acyclic)."""
    visited: dict[str, int] = {}  # 0 = visiting, 1 = done
    order: list[str] = []

    def visit(jt: str) -> None:
        state = visited.get(jt)
        if state == 1:
            return
        if state == 0:
            raise ValueError(f"cycle in job-dependency DAG at '{jt}'")
        visited[jt] = 0
        for up in upstreams(jt):
            if up.producer in PIPELINE:
                visit(up.producer)
        visited[jt] = 1
        order.append(jt)

    for jt in PIPELINE:
        visit(jt)
    return order


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
    """Yield each upstream job once, following every declared gated edge of each node's
    type, depth-first and cycle-safe. Replaces the one-hop, per-call-site parent walks:
    the chain (eval → training → sdg) is traversed in full. ``job`` is a job row;
    ``fetch(id)`` resolves an id to a row (or None)."""
    seen = _seen if _seen is not None else set()
    for up in upstreams(str(job.get("type") or "")):
        up_id = job_ref(job, up.field)
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


# ---------------------------------------------------------------------------
# Prompt-consistency contract
# ---------------------------------------------------------------------------

DEFAULT_AGENTS_DIR = Path(__file__).resolve().parents[3] / "agents"


def _agent_prompt_corpus(agents_dir: Path, job_type: str) -> str | None:
    """The agent's workflow.md plus all of its skills' SKILL.md, concatenated. A
    dependency field may be documented in either place, so the consistency check
    looks across the union ("including the skill"). Returns None if the agent has no
    workflow.md at all."""
    workflow = agents_dir / job_type / "workflow.md"
    if not workflow.is_file():
        return None
    parts = [workflow.read_text()]
    skills = agents_dir / job_type / "skills"
    if skills.is_dir():
        parts.extend(p.read_text() for p in sorted(skills.glob("*/SKILL.md")))
    return "\n".join(parts)


def check_workflow_consistency(agents_dir: Path | None = None) -> list[str]:
    """Verify the code DAG agrees with the agent prompts. Returns a list of
    inconsistencies (empty = consistent):

      1. Every gated upstream field is mentioned in its agent's workflow.md or a
         skill — the prompt must tell the agent about every dependency the code gates.
      2. The orchestrator prompt presents the capabilities in pipeline order.

    A test asserts this is empty, so code and prompts cannot silently drift.
    """
    agents_dir = agents_dir or DEFAULT_AGENTS_DIR
    problems: list[str] = []

    for node in PIPELINE.values():
        gated = gated_upstreams(node.job_type)
        if not gated:
            continue
        corpus = _agent_prompt_corpus(agents_dir, node.job_type)
        if corpus is None:
            problems.append(f"{node.job_type}: no workflow.md under agents/{node.job_type}/")
            continue
        for up in gated:
            if up.field not in corpus:
                problems.append(
                    f"{node.job_type}: prompt never mentions gated upstream field"
                    f" '{up.field}' ({up.label}) — code gates it but the agent is"
                    " not told how to wire it"
                )

    orch = agents_dir / "orchestrator" / "workflow.md"
    if not orch.is_file():
        problems.append("orchestrator/workflow.md missing — cannot check pipeline order")
        return problems

    text = orch.read_text()
    found: list[tuple[str, int]] = []
    for jt in pipeline_order():
        phrase = PIPELINE[jt].orchestrator_phrase
        idx = text.find(phrase)
        if idx < 0:
            problems.append(
                f"orchestrator/workflow.md never presents '{phrase}' for the '{jt}' stage"
            )
        else:
            found.append((jt, idx))

    appear_order = [jt for jt, _ in sorted(found, key=lambda p: p[1])]
    expected_order = [jt for jt in pipeline_order() if jt in {f[0] for f in found}]
    if appear_order != expected_order:
        problems.append(
            "orchestrator/workflow.md presents the stages out of pipeline order:"
            f" prompt shows {appear_order}, DAG requires {expected_order}"
        )

    return problems
