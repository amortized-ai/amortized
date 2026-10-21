### Phase 4 — Signal Completion
Call `signal_phase` this response with the step for the phase you are in — this phase: `phase-4-complete`.

Only reach this phase once the FULL run (mode "create") has succeeded. A
`preview` job finishing is NOT completion — it is only the small sample. If the
job that just finished was a preview, do NOT complete: show the sample, then run
the full `mode: "create"` job and wait for IT to finish first.

When the full (create) job succeeds, preview the generated data using the job's
`mlflow_run_id` so the user can verify quality.

Then call `signal_subagent_completion` to hand control back to the
orchestrator. If the user expressed a next intent (e.g. "Train a model
on this dataset"), include it in the summary as "User selected: ..."
so the orchestrator can act on it directly. Do NOT instruct the
orchestrator what to do — just relay the user's choice.
