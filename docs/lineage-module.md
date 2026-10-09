# Design: a job-lineage module (`core/lineage.py`)

**Status:** implemented. `core/lineage.py` (the canonical DAG + field-driven gate +
prompt-consistency check) + field-driven Layer 1 / Layer 2 rewiring + producer-scoped
delegation advisory + `tests/test_lineage.py`.
**Branch:** `feat/lineage` (off the `feat/mirror-module` stack tip)

## Problem

Amortized's pipeline is a tiny directed graph of job types — SDG produces a dataset,
training consumes it and produces a model, eval consumes both. Those edges were
declared **three times**, independently:

- **Layer 1** (`api/jobs.py`) — the fail-closed create-time validators
  (`_validate_training_data`, `_validate_eval_data`) hard-coded which fields to gate
  (`parent_job_id`, `training_job_id`).
- **Layer 2** (`api/agent.py`) — the proxy stage-gate hard-coded the same edges twice
  more, as `_DISPATCH_PARENT_TYPE` and `_DISPATCH_UPSTREAM_FIELDS`.
- **The prompts** (`agents/*/workflow.md` + skills) — prose telling the agent to wire
  `parent_job_id` / `training_job_id`, with nothing tying that prose to the code.

Three copies of one graph drift. Worse, the prompt copy is invisible to CI: the code
could gate an edge the prompt never mentions (the agent is never told how to wire it),
or the prompt could describe an order the code doesn't enforce.

## The module

`core/lineage.py` is the single authoritative declaration:

```python
PIPELINE = {
  "sdg":      JobNode("sdg",      "validate_sdg_job",      "Synthetic data generation", ()),
  "training": JobNode("training", "validate_training_job", "Model training",
                      (Upstream("parent_job_id", "sdg", "the training dataset's job"),)),
  "eval":     JobNode("eval",     "validate_eval_job",     "Model evaluation",
                      (Upstream("parent_job_id",    "sdg",      "the eval dataset's job"),
                       Upstream("training_job_id", "training", "the model's training job"))),
}
```

Edges point downstream→upstream and carry the config `field`, the `producer` job type,
a human `label`, and a `gated` flag. Pure (no FastAPI/opencode/DB), so both the API
routes and the proxy import it.

### Local precondition, not a prescribed path

The invariant is deliberately narrow: **a node may launch iff the upstream jobs its
config *references* have `succeeded`.** It says nothing about what else happened —
extra steps, loops, re-runs, out-of-order exploration are all fine. So enforcement is
**field-driven**, not node-driven:

- `GATED_FIELDS` — the flat universe of config keys that reference an upstream *job*
  (`parent_job_id`, `training_job_id`), each with a generic label.
- `present_gated_fields(config)` — the gated references actually present on a job, as
  `(field, job_id, label)`. **Both** enforcement layers call this, so the rule lives
  in one place and applies to **any** job — including a type not in `PIPELINE`
  ("beyond the graph"): it is still gated on exactly the prerequisites it names.

`PIPELINE` is the richer *structure* on top (producer edges, topological order,
orchestrator headings). It drives the prompt-consistency check and the delegation
advisory's producer scoping — it is **not** a runtime path gate. Other helpers:
- `dispatch_job_type(tool)` — the job type a `validate_<type>_job` dispatch launches,
  or `None` for a non-dispatch or a *declared root* (SDG, no prerequisite). An unknown
  type returns the type, so a beyond-graph dispatch is still gated.
- `dispatch_parent_type()` — `{validate_tool: type}` for the stale-reuse check (roots
  excluded).
- `pipeline_order()` — a topological sort (`sdg → training → eval`); raises on a cycle.
  Used **only** by the consistency test, never to constrain a user action.

### `data_run_id` / `eval_data_run_id` are NOT gated

They are MLflow run ids, not jobs, so they carry no job-ordering gate — they are
validated downstream as data availability, and are absent from `GATED_FIELDS`. And a
root job (SDG) has no gated field: cloning an SDG recipe for an eval set may run while
training is still in flight ("prep eval set now"), so `dispatch_job_type` returns
`None` for it.

## Consistency with workflow.md (including skills)

`check_workflow_consistency(agents_dir)` verifies the code DAG agrees with the prompts
and returns the list of inconsistencies (empty = consistent):

1. **Every gated upstream field is documented** in its agent's `workflow.md` *or* one
   of that agent's `skills/*/SKILL.md` — the union, so a dependency may live in a
   skill ("including the skill"). If the code gates `training_job_id` but the eval
   prompt never mentions it, the check fails.
2. **The orchestrator presents the stages in pipeline order** — the capability
   headings ("Synthetic data generation" → "Model training" → "Model evaluation")
   must appear in the order `pipeline_order()` requires.

`tests/test_lineage.py` asserts `check_workflow_consistency() == []` against the real
`agents/` tree, plus negative cases (an omitted field, an out-of-order orchestrator)
on synthetic trees. So a change to the code graph that leaves the prompts behind — or
a prompt reorder that contradicts the code — fails CI.

## Rewiring (field-driven)

- **Layer 1** (`api/jobs.py`): a new `_gate_upstream_refs(repo, refs)` iterates
  `lineage.present_gated_fields(refs)` and calls `_require_dependency_succeeded` per
  referenced upstream job. `_validate_training_data` / `_validate_eval_data` build
  `refs` from the config (plus the popped `parent_job_id`) and call it. The
  data-availability fork (parent_job_id vs data_path vs data_run_id) stays inline —
  that is about *which alternative source* is chosen, not graph ordering — so training
  still skips the parent gate when a direct `data_path` is supplied.
- **Layer 2** (`api/agent.py`): `_not_ready_violations` no longer needs a per-tool
  field map — it calls `lineage.dispatch_job_type(tool)` to decide a tool is a gated
  dispatch, then `present_gated_fields(tool_input)` to gate each referenced upstream.
  `_DISPATCH_PARENT_TYPE` (for the separate stale-reuse check) still derives from
  `lineage.dispatch_parent_type()`.
- **Delegation advisory** (`_upstream_not_ready_advisory`): now scoped to the target's
  **producer types** (`lineage.upstreams(target)`). An in-flight job of a type the
  target does not depend on — unrelated work mentioned alongside the handoff — is not a
  prerequisite and no longer trips the advisory.

Create-time 422 behavior is unchanged for the declared graph; the `parent_job_id`
message now uses one generic label (it is shared across node types).

## Why field-driven (generalization)

The gate asks "what prerequisites does *this config* reference?", not "is this a known
node?". So it generalizes two ways at once: a new edge is added by extending
`GATED_FIELDS` + `PIPELINE` once (both layers pick it up), and a job type **beyond the
graph** is covered automatically — it launches only once the prerequisites it names
have succeeded, with no new wiring. `PIPELINE` stays the structural/documentation layer
(producer edges, order, prompt contract); it never constrains the overall path a user
takes.
