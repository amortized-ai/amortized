### Step 1.5 — Confirm how much of the dataset to evaluate
Call `signal_phase` this response with the step for the phase you are in — this phase: `step-1_5-how-much`.

Once the dataset is chosen, ALWAYS confirm data usage with the user
before assembling the job config — never silently evaluate the whole
dataset. Call `get_dataset` on the eval dataset's run ID and present
THAT record count — the eval set's own size. Do NOT reuse or reference
record counts from the training data (its preview or full sample count
mentioned in the handoff context are a different dataset); the eval set
does not "replace" or continue them. Then ask:

- "Use all N records" (the default)
- "Use a portion — split the dataset first" (the user gives a count
  or fraction)

Skip this question ONLY if the user already specified the portion in
this conversation (e.g. they asked for the 20% hold-out split in
Step 1 — the dataset they picked IS the portion; do not split again).

If the user wants a portion: call `split_dataset` with their count or
fraction (strategy random, seed 42 by default; create the complement
too when it could serve as a training set). Tell the user the split is
running — a monitor card appears automatically and they will be
notified when it finishes. When the split job completes, call
`get_job` with its ID and report both datasets from the config:
`split_run_id` (the portion) and `complement_run_id` (the rest), with
their record counts (`num_portion` / `num_complement`). Then use the
portion's run ID as `eval_data_run_id` and continue with Step 2
(`step-2-model-source`) — do not re-ask anything already decided.
