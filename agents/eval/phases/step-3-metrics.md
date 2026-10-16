### Step 3 — Design the metrics (approval loop)
(`signal_phase` step: `step-3-metrics`)

**First, check for an existing metric set.** Call `get_dataset` with
the dataset's run ID — the response includes an `eval_metric_set`
field (`{rubric: [{name, description}]}`) when any
eval has run on this dataset before. If it is present, show the user
the persisted metric set as a table (same format as a new proposal)
and ask them to confirm reuse — do NOT silently reuse it, and do NOT
design a new one. Frame it as: "This dataset already has a metric set
(from earlier evals, so scores stay comparable across models). Reuse
it?" with options "Reuse these metrics" and "Suggest changes". If they
confirm reuse, use the persisted set verbatim. If they suggest
changes, incorporate their feedback (this counts as the explicit
change request below). Only design new metrics from scratch when the
field is absent (the first eval on this dataset) — UNLESS the user explicitly
asks to change or replace the metric set. In that case design the new
set with the user, use it for the eval, and note that it replaces the
dataset's old metric set (only evals created afterwards use it; past
scores in the Evaluation tab keep their own metrics). Also warn that
scores before and after the change are not comparable.

When designing (first eval on the dataset): do NOT assume which
metrics matter — ask the user first:

> "How do you want to evaluate the model? What should a good output
> look like for this task?"

Based on their answer, DESIGN a list of **custom rubric criteria**.
Each criterion is `{name, description}` where the name is a short key
(e.g. `factual_accuracy`) and the description is one sentence telling
the judge what to check. Design these FROM what the user said matters —
their words, translated into checkable criteria. The judge scores each
criterion 0-1 against the reference answer (absolute). An eval needs
at least one criterion.

Present the proposed list as a markdown table:

| Criterion | What it checks |
|---|---|
| factual_accuracy | response states facts consistent with the reference |
| completeness | response covers everything the reference asks for |

Then ask for approval with exactly two options:
- "Looks good, continue"
- "Suggest changes"

**If the user suggests changes**: incorporate their feedback into the
list (add, remove, reword, split, or merge criteria), then re-present
the revised table and ask again. Repeat until the user approves. Never
submit with a criteria list the user has not approved.

Judge (required — you set it, there is no default): the rubric criteria
are scored by an LLM judge, and you MUST set `config.judge` explicitly.
Ask the user which model to judge with — offer a soft suggestion, e.g.
"reuse the model we used for the SDG step as the judge, or a different
one?" — then set the judge from their choice. Get its endpoint (base_url
+ model) from `get_eval_endpoint_suggestions` / `list_models`, the same
way you build the model-under-eval endpoint. Never submit a rubric eval
without `config.judge`.

Use sensible defaults: `temperature` 0. Leave `max_samples` and
`judge_max_samples` unset — the eval runs and judges EVERY record in
the dataset; only set them (lower) if the user explicitly wants to cap
eval time or judge cost.
