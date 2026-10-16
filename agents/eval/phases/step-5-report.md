### Step 5 — Report results
(`signal_phase` step: `step-5-report`)

After the job is submitted, WAIT — do not hand control back to the
orchestrator while the job runs. Its `[SYSTEM EVENT]` completion
notification arrives in YOUR session; when it does (or on failure),
you handle it.

When the eval job completes, call `get_eval_results` with the job ID to
fetch the aggregate metrics. Report a compact table of the model's
scores:

| Criterion | Score |
|---|---|
| factual_accuracy | ... |
| completeness | ... |

Rubric criteria are absolute 0-1 scores (shown as percentages) — the
share of the reference-level quality the model reached on that
criterion, averaged over the scored samples.

If `get_eval_results` reports `num_scored` of 0 (or null/empty scores),
the eval produced no usable results — say so plainly, check the job logs,
and diagnose the cause.

Interpret the numbers
plainly and offer next steps: evaluate another model to compare, train
again with different data or parameters, or accept the model. If they
choose to evaluate another model (e.g. the base model for a tuned-vs-base
comparison), start a FRESH eval cycle for it — assemble that model's
config and go to Step 4 (`step-4-validate`) (`validate_eval_job` →
confirmation card). Do NOT ask the user to submit it or hand you a job ID,
and do NOT use `present_options` as a submit control.

Cross-model comparisons live in the Studio **Evaluation tab**: every
model evaluated on the same dataset with the same metric set appears
as a column there — point the user to it when they want to compare
numbers side by side.
