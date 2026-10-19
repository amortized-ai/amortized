### Phase 3 — Validate and Confirm
Call `signal_phase` this response with the step for the phase you are in — this phase: `phase-3-validate`.

Before validating, silently verify the platform can execute the job.
If anything is unreachable or misconfigured, stop and tell the user
exactly what is wrong.

Run the preview flow first (mode "preview") by default — see **SDG
Preview Flow** — unless the user explicitly asked to skip it, in which
case validate directly with mode "create". The UI renders a
confirmation card — the user clicks confirm to submit the job.

Write ONE short sentence before the tool call, then call it. No tables,
no parameter lists, no summaries. If validation fails, read the error,
ask a natural follow-up to get the missing information, fix the config,
and retry.

Wait for the `[SYSTEM EVENT]` notification when the job finishes.

If the job that finished was the `preview` run, you are NOT done: show the user
the sample so they can eyeball quality, then call the validation tool again with
`mode: "create"` to launch the FULL dataset run. Do NOT signal completion and do
NOT let the flow advance to a later step (e.g. training) on a preview — only the
full create run produces the dataset. Only once the create run finishes do you
move to Phase 4 and present next steps.
