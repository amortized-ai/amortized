### Phase 1 — Route to Sub-Skill
(`signal_phase` step: `phase-1-route`)

Determine which training sub-skill to use based on the handoff context:
knowledge-ingestion/OSFT for doc-grounded QA, or embedding-classifier for
classifying text into a fixed set of categories. For classification tasks,
embedding-classifier is the default — **unless** the user explicitly wants a
**generative/LLM** classifier (an LLM that emits the label as text on `messages`
data), which is a normal LLM fine-tune and routes to OSFT instead. Load the
matching skill (`skill({ name: "training-knowledge-ingestion-osft" })` or
`skill({ name: "training-embedding-classifier" })`). You MUST load it before
Phase 2 (`phase-2-gather`) — it contains the detailed guidance (the OSFT skill
also bundles the supported-model list). For the embedding classifier, skip the
VRAM/model-size/method steps.
