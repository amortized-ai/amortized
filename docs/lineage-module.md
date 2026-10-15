# Design: field-driven job-dependency enforcement (`core/lineage.py`)

**Status:** implemented. `core/lineage.py` (field-driven gate + upstream traversal) +
Layer 1 / Layer 2 rewiring + `tests/test_lineage.py`.
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

No job type is special-cased. An SDG job that chains from an unfinished parent is gated
exactly like a training or eval job; an SDG with no upstream reference gates nothing,
because it carries no gated field — not because of an exception for its type.

## What the module provides

- `GATED_FIELDS` / `present_gated_fields(config)` — the field-driven gate (both layers).
- `ancestors(job, fetch)` / `nearest_ancestor(job, types, fetch)` — upstream traversal that
  follows the same `GATED_FIELDS` references depth-first (cycle-safe). Used by the eval-SDG
  mirror warning in `jobs.py` (not by the gate) to find a dataset producer across multi-hop
  resume chains. `job_ref(job, field)` reads a reference whether it lives in a top-level
  column or the config JSON.
- `READY_STATUS`, `DATASET_PRODUCER_TYPES` — shared constants.

Pure: no FastAPI/opencode/DB deps. The DB status check stays in the callers; the module
only owns *which* fields are gated references and how to walk them.

## Rewiring (field-driven)

- **Layer 1** (`api/jobs.py`): `_gate_upstream_refs(repo, refs)` iterates
  `present_gated_fields(refs)` and calls `_require_dependency_succeeded` per referenced
  upstream job. The data-availability fork (parent_job_id vs data_path vs data_run_id)
  stays inline — it is about *which source* is chosen, not graph ordering — so training
  still skips the parent gate when a direct `data_path` is supplied.
- **Layer 2** (`api/agent.py`): `_not_ready_violations` recognises a committing dispatch via
  the validate-tool→type map (`_TYPE_BY_VALIDATE_TOOL`; tool names are a proxy concept, so
  this correspondence is owned in the proxy, not the pure module — its single source is
  `_JOB_TYPE_FOR_CONFIRM` in `agent_confirm`, the confirm cards being these same tools), then
  `present_gated_fields(tool_input)` gates each referenced upstream.
- **Stale-reuse check** (`_dispatch_provenance_violations`): uses `_DISPATCH_PARENT_TYPE`,
  derived in `agent.py` as `_TYPE_BY_VALIDATE_TOOL` minus SDG — the data-dependent downstream
  types (SDG is a dataset producer, not a consumer of one).

The only soft, prose-level advisory left at the delegation boundary is the **data
availability fork** (`_no_data_advisory`): when no dataset has been chosen this
conversation, steer the subagent to let the user pick existing-vs-fresh rather than
default to SDG. The earlier *upstream-not-ready* advisory was removed — it duplicated the
hard create-time/dispatch gate in prose and needed prose job-id scraping to work.

Create-time 422 behavior is unchanged for the real graph.

## What was deliberately removed

An earlier draft carried a richer structural layer — a `PIPELINE` DAG of `JobNode` /
`Upstream` objects (per-node producer edges, orchestrator capability headings), a
topological sort, and a `check_workflow_consistency()` that asserted the code graph agreed
with the agent prompts. It was dropped as **not necessary** for correctness and
**brittle** under workflow churn:

- The hard gate never needed it — `GATED_FIELDS` / `present_gated_fields` is the source of
  truth. No type is special-cased: an SDG is gated on its own references like anything else.
- The consistency check coupled CI to prompt *wording* (exact-phrase matching of capability
  headings, substring matching of field names), so routine prompt edits caused false CI
  failures exactly when iterating. The small win (catching a gated field the prompts never
  mention) did not justify that friction; prompt/code agreement is caught in review.

The residual type knowledge (which tools are committing dispatches) is a tiny local
constant in `agent.py` at its call site, not a shared graph. `jobs.py` deals in job types
directly and never needs it.

## Tests

- `tests/test_lineage.py`: `present_gated_fields` (referenced jobs only, runs excluded,
  any-type coverage, empty when no reference); ancestry (`job_ref`, full-chain walk,
  nearest-ancestor single/multi-hop, type-set, cycle-safety). ✅
- `tests/test_agent_stage_gate.py`: Layer-2 not-ready gate, SDG dispatch is field-driven (no
  special case), the data-availability fork (`_no_data_advisory`). ✅
