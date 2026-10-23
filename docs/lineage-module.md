# Design: field-driven job-dependency enforcement (`core/lineage.py`)

**Status:** implemented. `core/lineage.py` (field-driven gate + upstream traversal) +
a single create-time gate in `api/jobs.py` + `tests/test_lineage.py`.
**Branch:** `feat/lineage` (off `main`)

## Problem

Amortized's pipeline is a tiny chain of job types — SDG produces a dataset, training
consumes it and produces a model, eval consumes both. The dependency edges were declared
independently in several places, free to drift: the fail-closed create-time validators in
`api/jobs.py` hard-coded which fields to gate (`parent_job_id`, `training_job_id`), and an
earlier draft *also* re-asserted the same edges in a reactive proxy stage-gate in
`api/agent.py`. Two copies of the same rule meant two things to keep in sync.

## Core idea: one field-driven precondition

The invariant is deliberately narrow: **a job may launch iff the upstream jobs its config
*references* have `succeeded`.** It says nothing about what else happened — extra steps,
loops, re-runs, out-of-order exploration are all fine. So enforcement is **field-driven**,
not keyed on a modelled job type:

- `GATED_FIELDS` — the flat universe of config keys that reference an upstream *job* by id
  (`parent_job_id`, `training_job_id`), each with a generic label.
- `present_gated_fields(config)` — the gated references actually present on a job, as
  `(field, job_id, label)`. The single create-time gate calls it, so the rule lives in one
  place and applies to **any** job — including a type not otherwise modelled ("beyond the
  graph"): it is still gated on exactly the prerequisites it names.

`data_run_id` / `eval_data_run_id` are MLflow **run** ids, not jobs, so they carry no
job-ordering gate and are absent from `GATED_FIELDS` (validated downstream as data
availability instead).

No job type is special-cased. An SDG job that chains from an unfinished parent is gated
exactly like a training or eval job; an SDG with no upstream reference gates nothing,
because it carries no gated field — not because of an exception for its type.

## Why `succeeded` is the right bar

The deliverable upload (`generated_data` for SDG, `model` for training) is a
`post_commands` step that `_wrap_command` chains with `&&` (fail-fast), and a job is marked
`succeeded` only after that whole chain exits 0 and the MLflow run is finalized. So
`succeeded` ⇔ the deliverable was fully logged, and gating consumption on it has no false
positives — a non-succeeded producer has nothing complete to consume anyway. (The
console-log upload is deliberately the opposite — `|| true`, best-effort — so logs survive
a crash while the deliverable stays fail-fast.)

## What the module provides

- `GATED_FIELDS` / `present_gated_fields(config)` — the field-driven gate.
- `ancestors(job, fetch)` — upstream traversal that follows the same `GATED_FIELDS`
  references **breadth-first** (nearest first, cycle-safe), yielding each upstream job once;
  the chain (eval → training → sdg) is traversed in full.
- `nearest_ancestor(job, types, fetch)` — the closest upstream job of a given type; **reuses
  the breadth-first `ancestors` walk** and stops at the first hit, so "closest" is by hop
  distance (a one-hop `data_run_id` SDG wins over a two-hop SDG reached through the parent).
  Used by the eval-SDG mirror warning in `jobs.py` (not by the gate) to find a dataset
  producer across multi-hop resume chains.
- `job_ref(job, field)` — reads a reference whether it lives in a top-level column or the
  config JSON.
- `READY_STATUS`, `DATASET_PRODUCER_TYPES` — shared constants.

Pure: no FastAPI/opencode/DB deps. The DB status check stays in the callers; the module
only owns *which* fields are gated references and how to walk them.

## Enforcement — a single create-time gate

All hard enforcement lives at **one** place, the create boundary in `api/jobs.py`. There is
**no** separate reactive proxy gate in `api/agent.py`.

- `_gate_upstream_refs(repo, refs)` iterates `present_gated_fields(refs)` and calls
  `_require_dependency_succeeded` per referenced upstream job. It runs in the fail-closed
  create-time validators (`_validate_training_data`, `_validate_eval_data`) and in
  `create_sdg_job`, so SDG chaining from an unsucceeded parent is rejected like any other
  consumer.
- The data-availability fork (parent_job_id vs data_path vs data_run_id) stays inline — it
  is about *which source* is chosen, not graph ordering — so training still skips the parent
  gate when a direct `data_path` is supplied.
- Create-time 422 behavior is unchanged for the real graph.

The only lineage-related steer left at the delegation boundary is the soft, prose-level
**data availability fork** (`_no_data_advisory` in `agent.py`): when no dataset has been
chosen this conversation, nudge the subagent to let the user pick existing-vs-fresh rather
than default to SDG. It is advisory only (no gate), and needs no job-id scraping.

## What was deliberately removed

- **The Layer-2 proxy stage-gate** (`_not_ready_violations` in `agent.py`) and the
  **stale-reuse provenance check** (`_dispatch_provenance_violations`, its
  `_TYPE_BY_VALIDATE_TOOL` / `_DISPATCH_PARENT_TYPE` maps, and the earlier
  *upstream-not-ready* advisory). They duplicated the hard create-time gate in prose and
  needed id-scraping of model-authored text to work. Enforcement is now consolidated at the
  create boundary; `agent.py` keeps only the soft `_no_data_advisory`.
- **A richer structural layer** from an earlier draft — a `PIPELINE` DAG of `JobNode` /
  `Upstream` objects (per-node producer edges, orchestrator capability headings), a
  topological sort, and a `check_workflow_consistency()` that asserted the code graph agreed
  with the agent prompts. Dropped as **not necessary** for correctness and **brittle** under
  workflow churn:
  - The hard gate never needed it — `GATED_FIELDS` / `present_gated_fields` is the source of
    truth. No type is special-cased.
  - The consistency check coupled CI to prompt *wording* (exact-phrase matching of
    capability headings, substring matching of field names), so routine prompt edits caused
    false CI failures exactly when iterating. The small win (catching a gated field the
    prompts never mention) did not justify that friction; prompt/code agreement is caught in
    review.

The residual type knowledge (which validators belong to which job type) is local to each
`jobs.py` call site, not a shared graph.

## Tests

- `tests/test_lineage.py`: `present_gated_fields` (referenced jobs only, runs excluded,
  any-type coverage, empty when no reference); ancestry (`job_ref`, full-chain walk,
  `nearest_ancestor` single/multi-hop + BFS-vs-DFS ordering, type-set, cycle-safety). ✅
- `tests/test_jobs_dependency_gate.py`: the create-time gate — `_require_dependency_succeeded`
  (missing / in-flight / succeeded, mlflow-required toggle), and `_validate_eval_data` /
  `_validate_training_data` rejecting in-flight parents/training refs, the `data_path`
  skip, and no double-gating of `training_job_id`. ✅
- `tests/test_agent_stage_gate.py`: the delegation advisory (`_no_data_advisory`) — fires
  when no data is chosen and does not stack, stays silent once data is chosen. ✅
