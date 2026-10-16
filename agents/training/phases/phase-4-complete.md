### Phase 4 — Signal Completion
(`signal_phase` step: `phase-4-complete`)

Call `signal_subagent_completion` to hand control back to the
orchestrator. If the user expressed a next intent (e.g. "Generate more
data"), include it in the summary as "User selected: ..." so the
orchestrator can act on it directly. Do NOT instruct the orchestrator
what to do — just relay the user's choice.
