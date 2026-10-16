### Step 6 — Signal Completion
(`signal_phase` step: `step-6-complete`)

Do NOT signal yet. After the job is submitted, stay active: the job's
`[SYSTEM EVENT]` completion notification arrives in YOUR session, and
Step 5 (`step-5-report`: report the scores table, then offer next steps) is
your work — handing control back before the job finishes would hand the
results reporting to the orchestrator, which has none of the eval context.

Only after you have reported the results (Step 5) — or the job failed
and you have explained the failure — call `signal_subagent_completion`
to hand control back to the orchestrator. If the user expressed a next
intent (e.g. "Evaluate another model" or "Train again"), include it in
the summary as "User selected: ..." so the orchestrator can act on it
directly. Do NOT instruct the orchestrator what to do — just relay the
user's choice.
