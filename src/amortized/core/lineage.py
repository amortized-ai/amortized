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

This module makes the graph explicit and authoritative, so the two enforcement
layers derive their edge set from one declaration instead of each hard-coding it
(they drifted apart otherwise). It also carries the prompt-consistency contract
(``check_workflow_consistency``): the fields this DAG gates must be documented in the
matching ``agents/<type>/workflow.md`` or one of that agent's skills, and the
pipeline order here must match the order the orchestrator prompt presents. A test
runs that check, so a change to the code graph that leaves the prompts (or skills)
behind — or vice versa — fails CI.

Pure: no FastAPI / opencode / DB deps, so both the API routes and the proxy can
import it. The DB status check itself stays in the callers; this module only owns
*which* fields are edges, their producing job type, and whether they are gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# A referenced upstream job counts as ready only once it reaches this status.
READY_STATUS = "succeeded"


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


def upstreams(job_type: str) -> tuple[Upstream, ...]:
    """Every declared upstream edge for ``job_type`` (empty for roots/unknown)."""
    node = PIPELINE.get(job_type)
    return node.upstreams if node else ()


def gated_upstreams(job_type: str) -> tuple[Upstream, ...]:
    """The upstream edges subject to the stage-gate — those a downstream job must
    wait on. This is the single source both enforcement layers consume."""
    return tuple(u for u in upstreams(job_type) if u.gated)


def dispatch_parent_type() -> dict[str, str]:
    """``{validate_tool: downstream_job_type}`` for the proxy's dispatch gates."""
    return {n.validate_tool: n.job_type for n in PIPELINE.values()}


def dispatch_upstream_fields() -> dict[str, list[tuple[str, str]]]:
    """``{validate_tool: [(field, label), ...]}`` of gated upstreams, for the proxy's
    not-ready stage-gate. Only tools with at least one gated upstream appear (SDG,
    with none, is intentionally absent — cloning a recipe for an eval set may run
    while training is still in flight)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for node in PIPELINE.values():
        fields = [(u.field, u.label) for u in node.upstreams if u.gated]
        if fields:
            out[node.validate_tool] = fields
    return out


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
