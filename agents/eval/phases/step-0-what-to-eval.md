### Step 0 — Confirm WHAT to evaluate
Call `signal_phase` this response with the step for the phase you are in — this phase: `step-0-what-to-eval`.

**Never assume which model the user wants to evaluate.** Handoff
context may list recently completed jobs — that is background
information, NOT the user's choice. Do not open with "I'll help you
evaluate your <most recent model>" unless the user actually picked it.

Two legitimate entries:

- **The user explicitly chose a model** — e.g. they picked "Evaluate
  the model" right after a training job completed (the handoff context
  says so), or they named the model in their message. Then that model
  is the eval subject; skip to Step 1 (`step-1-dataset`).
- **The user just said "evaluate a model"** — ask which. Call
  `list_jobs` with type=training and present options:

  - One option per recent trained model: "Evaluate <tuned model name>"
  - "Evaluate another model — I'll pick or provide the endpoint"

  If there are no trained models at all, go straight to evaluating an
  arbitrary endpoint.

The model does NOT have to come from training. Any HF model or
OpenAI-compatible endpoint can be evaluated (e.g. a gateway model).

**Once the subject is settled, your next phase is the eval dataset
(`step-1-dataset`).** If no eval dataset exists in this conversation, you
cannot generate one yourself — you are a leaf and do not delegate. Your
move is to CALL `request_orchestrator_help`; do NOT describe handing off in
prose. (Full dataset logic — reuse, split, or hand back — is in the
`step-1-dataset` phase.)
