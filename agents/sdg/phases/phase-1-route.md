### Phase 1 — Route to Sub-Skill
Call `signal_phase` this response with the step for the phase you are in — this phase: `phase-1-route`.

**Eval set for a trained model? Clone, don't redesign.** If the handoff carries a
training job ID, don't pick a sub-skill or design a new recipe — ask only the record
count (`present_options`, sensible default), call
`clone_sdg_config_for_eval(training_job_id, num_records)`, and pass its config straight
to validate. Don't hand-assemble it (teacher/prompt drift), and don't compare datasets
afterward (fresh sampling is the isolation; eval validation also warns on overlap). If there's
no parent SDG to mirror, offer a fresh set from scratch or a user-provided config — and if
they choose a fresh set, route by task and load a sub-skill exactly like the "Otherwise"
step below (a fresh set still needs a loaded guide for Phase 2; don't skip skill-loading
just because a training job ID was present).

Otherwise, route by task. Determine whether this is an embedding-classifier (default for
classify/route/label tasks), a generative classification, knowledge-ingestion,
or task-distillation task. Use the context provided by the orchestrator to
make this decision. If the context does not make it clear, ask the user —
in particular, for a classifier confirm whether they want a compact embedding
classifier (recommended) or a generative/LLM one.

Once determined, load the matching skill with the `skill` tool (e.g.
`skill({ name: "sdg-task-distillation" })`). You MUST load it before
Phase 2 (`phase-2-gather`) — it contains the exact requirement-gathering steps
and config rules.
