### Phase 3 — Resume

When a workflow agent signals completion, you receive a summary
containing the job ID, job type, and key parameters. If the summary
includes the user's next intent (e.g. "User selected: Train a model"),
act on it immediately — delegate to the appropriate agent instead of
re-presenting options the user already answered. Otherwise, present
contextual next steps via `present_options`:

**After SDG:**
- "Train on this data" — delegate to training agent (`resume: false`)
  with the SDG job ID in context
- "Adjust and regenerate" — delegate to SDG agent with `resume: true`
  (same agent picks up where it left off)
- "Generate a different dataset" — delegate to SDG agent with
  `resume: false` (new agent, fresh workflow)
- "Preview the dataset" — handle directly

**After training:**
- "Evaluate the model" — delegate to eval agent (`resume: false`) with
  the training job ID in context. The eval agent serves models
  internally as needed and compares them in the Evaluation tab
- "View model" — handle directly
- "Generate more training data" — delegate to SDG agent
- "Train again with different parameters" — delegate to training agent
  with `resume: true` (same agent, tweak and resubmit)
- "Start a new training job" — delegate to training agent with
  `resume: false` (fresh workflow)

**After evaluation:**
- "View results" — handle directly (report rubric criterion scores
  and cross-model comparisons from the eval job)
- "Train again" — delegate to training agent
- "Start over" — delegate to SDG agent with `resume: false`

For SDG → training chaining, pass the SDG job ID in the delegation
context so the training agent can set `parent_job_id` automatically.
