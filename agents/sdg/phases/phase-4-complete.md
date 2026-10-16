### Phase 4 — Signal Completion
(`signal_phase` step: `phase-4-complete`)

When the job succeeds, preview the generated data using the job's
`mlflow_run_id` so the user can verify quality.

Then call `signal_subagent_completion` to hand control back to the
orchestrator. If the user expressed a next intent (e.g. "Train a model
on this dataset"), include it in the summary as "User selected: ..."
so the orchestrator can act on it directly. Do NOT instruct the
orchestrator what to do — just relay the user's choice.
