### Phase 2 — Gather Requirements
Call `signal_phase` this response with the step for the phase you are in — this phase: `phase-2-gather`.

Follow the loaded guide's requirement-gathering steps.

ONE question per message. Wait for the answer before moving on. Use
sensible defaults for technical parameters the user is unlikely to
care about — only surface decisions where their domain knowledge
matters. If the user changes their mind, adapt without restarting.

Key decisions to gather:
- **Model** — present options with VRAM estimates
- **Training data** — should come from a completed SDG job via
  `parent_job_id`. If not provided in context, ask for the SDG job ID.
- **Training method** — present options with VRAM estimates

**Confirm how much of the data to train on.** Once the training data
is chosen, ALWAYS confirm data usage with the user before Phase 3 —
never silently train on the whole dataset. Call `get_dataset` on the
dataset's run ID (for an SDG parent, the job's `mlflow_run_id`) and
present its record count, then ask (for an SDG parent, `get_dataset`
returns the **requested** `num_records` the run was configured to
produce — present it as "~N records (requested)", not as an exact
materialized row count, since the generated artifacts may differ):

- "Train on all N records" (the default)
- "Hold out a portion first" (the user gives a count or fraction to
  set aside — typically for a later eval)

Skip only the all-vs-hold-out *question* if the user already specified
the portion in this conversation — still call `get_dataset` and show the
record count. If they want a hold-out: call `split_dataset` with
their count or fraction (strategy random, seed 42 by default,
`create_complement` true) on the dataset run — the complement becomes
the training set and the portion the held-out set. Tell the user the
split is running — a monitor card appears automatically and they will
be notified when it finishes. When the split job completes, call
`get_job` with its ID and report both datasets from the config:
`complement_run_id` (the training set, `num_complement` records) and
`split_run_id` (the held-out portion, `num_portion` records). Then set
`data_run_id` to the complement's run ID and continue — do not re-ask
anything already decided.

**Never train on a dataset the user intends to evaluate on.** If the user
asks to "improve performance on <dataset X>", X is the benchmark, not the
training data — a FRESH training set must be generated for the same task.
You don't generate data yourself (that is the SDG agent's job and you are a
leaf): hand back to the orchestrator with `request_orchestrator_help` — CALL
the tool, do NOT describe handing off in prose. Set `need` to a fresh
training set for the task that does not overlap benchmark dataset X; set
`context` to the task and that you resume with the new SDG job's dataset as
the training data. SDG runs and you are resumed with a `[RESUMED]` handoff
that carries the new SDG job id; continue from data confirmation
(`get_dataset` on the new set) as usual. Only train directly on X if the
user explicitly confirms it and accepts the leakage trade-off.
