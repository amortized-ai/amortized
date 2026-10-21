### Step 2 — Examples

Ask for examples first to extract format, criteria, and task-specific details
from them.

- **ICL examples** — 1-2 real content samples for style/format
  reference in the generation prompt.
- **Demonstration examples** — 1-2 (input, output) pairs showing the
  desired task behavior. If the user only has raw inputs (without
  outputs), continue to Steps 3–4 to establish the format and
  task criteria, then generate the system prompt (Step 8), and use
  it to generate outputs for the raw inputs (Step 9) before
  building the SDG config.

**Real-input corpus.** If the user's examples reveal they have an
existing corpus of real inputs (e.g., a backlog of tickets, a set of
RFEs, a document collection), infer this silently — do not ask "do
you want to generate inputs or use your own." Skip the input
generation column (Section 3) and wire the corpus directly into the
output column via a data-source column. Only the output half of the
pipeline is needed.

When you ask the user about this step, call present_options with step: "examples".
