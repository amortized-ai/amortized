# Design: field-driven job-dependency enforcement (`core/lineage.py`)

**Status:** implemented. `core/lineage.py` (field-driven gate + upstream traversal) +
Layer 1 / Layer 2 rewiring + producer-scoped delegation advisory + `tests/test_lineage.py`.
**Branch:** `feat/lineage` (off `main`)

## Problem

Amortized's pipeline is a tiny chain of job types — SDG produces a dataset, training
consumes it and produces a model, eval consumes both. The dependency edges were declared
independently in several places, free to drift:

- **Layer 1** (`api/jobs.py`) — the fail-closed create-time validators hard-coded which
  fields to gate (`parent_job_id`, `training_job_id`).
- **Layer 2** (`api/agent.py`) — the proxy stage-gate hard-coded the same edges again.

## Core idea: one field-driven precondition

The invariant is deliberately narrow: **a job may launch iff the upstream jobs its config
*references* have `succeeded`.** It says nothing about what else happened — extra steps,
loops, re-runs, out-of-order exploration are all fine. So enforcement is **field-driven**,
not keyed on a modelled job type:

- `GATED_FIELDS` — the flat universe of config keys that reference an upstream *job* by id
  (`parent_job_id`, `training_job_id`), each with a generic label.
- `present_gated_fields(config)` — the gated references actually present on a job, as
  `(field, job_id, label)`. **Both** enforcement layers call it, so the rule lives in one
  place and applies to **any** job — including a type not otherwise modelled ("beyond the
  graph"): it is still gated on exactly the prerequisites it names.

`data_run_id` / `eval_data_run_id` are MLflow **run** ids, not jobs, so they carry no
job-ordering gate and are absent from `GATED_FIELDS` (validated downstream as data
availability instead).

### Root types

`ROOT_TYPES = {"sdg"}` is the one piece of type knowledge kept: a root produces from
scratch and references no upstream job, so a dispatch of one is never stage-gated
(`dispatch_job_type` returns `None` for it). This is what lets cloning an SDG recipe for an
eval set run **while a training job is still in flight** ("prep eval set now"). Everything
else is decided from the fields present.

## What the module provides

- `GATED_FIELDS` / `present_gated_fields(config)` — the field-driven gate (both layers).
- `dispatch_job_type(tool)` — the job type a `validate_<type>_job` dispatch launches, or
  `None` for a non-dispatch tool or a `ROOT_TYPES` dispatch.
- `ancestors(job, fetch)` / `nearest_ancestor(job, types, fetch)` — upstream traversal that
  follows the same `GATED_FIELDS` references depth-first (cycle-safe), replacing the
  one-hop per-call-site parent walks. `job_ref(job, field)` reads a reference whether it
  lives in a top-level column or the config JSON.
- `READY_STATUS`, `DATASET_PRODUCER_TYPES` — shared constants.

Pure: no FastAPI/opencode/DB deps. The DB status check stays in the callers; the module
only owns *which* fields are gated references and how to walk them.

## Rewiring (field-driven)

- **Layer 1** (`api/jobs.py`): `_gate_upstream_refs(repo, refs)` iterates
  `present_gated_fields(refs)` and calls `_require_dependency_succeeded` per referenced
  upstream job. The data-availability fork (parent_job_id vs data_path vs data_run_id)
  stays inline — it is about *which source* is chosen, not graph ordering — so training
  still skips the parent gate when a direct `data_path` is supplied.
- **Layer 2** (`api/agent.py`): `_not_ready_violations` calls `dispatch_job_type(tool)` to
  recognise a gated dispatch, then `present_gated_fields(tool_input)` to gate each
  referenced upstream.
- **Delegation advisory** (`_upstream_not_ready_advisory`): the soft, prose-level
  companion at the handoff boundary. It is scoped to the target's upstream producer types
  via a small local map in `agent.py` (`_TARGET_UPSTREAM_TYPES = {"training": {"sdg"},
  "eval": {"sdg", "training"}}`) so an in-flight job of a type the target does **not**
  depend on — unrelated work mentioned alongside the handoff — is not treated as a
  prerequisite. Fail-safe: any uncertainty returns `None`, leaving the handoff unchanged.
- **Stale-reuse check** (`_dispatch_provenance_violations`): uses a small local map in
  `agent.py` (`_DISPATCH_PARENT_TYPE = {"validate_training_job": "training",
  "validate_eval_job": "eval"}`) — the data-dependent downstream types; SDG is absent
  because it has no parent chain.

Create-time 422 behavior is unchanged for the real graph.

## What was deliberately removed

An earlier draft carried a richer structural layer — a `PIPELINE` DAG of `JobNode` /
`Upstream` objects (per-node producer edges, orchestrator capability headings), a
topological sort, and a `check_workflow_consistency()` that asserted the code graph agreed
with the agent prompts. It was dropped as **not necessary** for correctness and
**brittle** under workflow churn:

- The hard gate never needed it — `GATED_FIELDS` / `present_gated_fields` is the source of
  truth, and `ROOT_TYPES` covers the one type fact that remains.
- The consistency check coupled CI to prompt *wording* (exact-phrase matching of capability
  headings, substring matching of field names), so routine prompt edits caused false CI
  failures exactly when iterating. The small win (catching a gated field the prompts never
  mention) did not justify that friction; prompt/code agreement is caught in review.

The two remaining type-relationship facts (advisory producer scoping, stale-reuse tool map)
are tiny local constants at their call sites, not a shared graph.

## Tests

- `tests/test_lineage.py`: `present_gated_fields` (referenced jobs only, runs excluded,
  beyond-graph coverage, empty for a root dispatch); `dispatch_job_type` (tool-name
  derivation, root skip, beyond-graph, non-dispatch); ancestry (`job_ref`, full-chain walk,
  nearest-ancestor single/multi-hop, type-set, cycle-safety). ✅
- `tests/test_agent_stage_gate.py`: Layer-2 not-ready gate, concurrent SDG prep NOT gated,
  delegation advisory producer scoping. ✅
