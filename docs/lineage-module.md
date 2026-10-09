# Design: a job-lineage module (`core/lineage.py`)

**Status:** implemented. `core/lineage.py` (the canonical DAG + prompt-consistency
check) + Layer 1 / Layer 2 rewiring + `tests/test_lineage.py`.
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

Helpers:
- `gated_upstreams(job_type)` — the edges a downstream job must wait on. Consumed by
  **both** enforcement layers, so they can no longer disagree on the edge set.
- `dispatch_parent_type()` / `dispatch_upstream_fields()` — rebuild the proxy's two
  dispatch tables from the DAG.
- `pipeline_order()` — a topological sort (`sdg → training → eval`); raises on a cycle.

### `data_run_id` / `eval_data_run_id` are NOT edges

They are MLflow run ids, not jobs, so they carry no job-ordering gate — they are
validated downstream as data availability. Only *job* references are edges. And
`validate_sdg_job` has no gated upstream on purpose: cloning an SDG recipe for an eval
set may run while training is still in flight ("prep eval set now"), so it is absent
from `dispatch_upstream_fields()`.

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

## Rewiring (behavior-preserving)

- `api/agent.py`: `_DISPATCH_PARENT_TYPE` / `_DISPATCH_UPSTREAM_FIELDS` now derive
  from `lineage.dispatch_parent_type()` / `dispatch_upstream_fields()`. The names and
  the rest of the gate are untouched.
- `api/jobs.py`: a new `_gate_declared_upstreams(repo, job_type, ids)` iterates
  `lineage.gated_upstreams(job_type)` and calls the existing
  `_require_dependency_succeeded` per referenced edge. `_validate_training_data` and
  `_validate_eval_data` call it instead of hard-coding the field list. The
  data-availability fork (parent_job_id vs data_path vs data_run_id) stays inline —
  that is about *which alternative source* is chosen, not graph ordering — so
  training still skips the parent gate when a direct `data_path` is supplied.

Error strings and 422 behavior are unchanged (labels are the field names, as before).

## Why a module (generalization)

The DAG is now a concept richer than any single caller: it owns the edges, the
topological order, and the prompt contract. New edges or a new job type are declared
once in `PIPELINE` and picked up by both layers and the consistency check — the shape
that lets lineage generalize beyond the two edges enforced today.
