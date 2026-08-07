# Knowledge Ingestion — SDG Guide

Use this guide when building FAQ bots, QA assistants, document-grounded chat,
or RAG-deployed knowledge models.

## How This Works

You will create an SDG config for the user's specific domain
and documents. Before starting, read the template at
`templates/sdg/knowledge-ingestion.yaml` for the
config structure.

**Document analysis workflow:** Use `get_document_chunks(doc_id)` to
get the document's chunks with token counts and headings. Use
`get_document_content` if you need the full text for prompt writing.

**Keep it brief.** Do your analysis silently. Present results and ask
for confirmation — do not narrate your reasoning.

## Requirement Gathering

Ask the user these questions (one at a time, using `present_options`):

### Step 1 — What documents?

Check if they've already uploaded documents via the Documents page. Use
`list_documents` to show available documents with their IDs. If not
uploaded yet, guide them to upload first.

### Step 2 — Question types

What kinds of questions should it handle?
Default: factual (25%), procedural (35%), troubleshooting (25%),
comparison (15%). Adapt to the domain — a troubleshooting guide
needs more troubleshooting questions, a reference manual needs more
factual ones.

### Step 3 — Difficulty levels

Default: basic (35%), intermediate (45%), advanced (20%).

### Step 4 — Which teacher model?

Call `list_models` to get the models configured on the AI Gateway.
Present ONLY those models as options. Do NOT suggest models that aren't
returned by `list_models` — they won't work. If no models are returned,
stop and direct the user to Settings -> AI Gateway.

### Step 5 — How many samples?

Documents are chunked at upload time. Use
`get_document_chunks(doc_id)` to get the chunk count and token
statistics.

Use ALL chunks in the calculation — the worker sends every chunk
to DataDesigner, so do not filter or exclude any chunks here.

The goal is for total training tokens to be a multiple of total
source tokens. Research suggests ~5x source coverage as a good
target. Compute the per-chunk multiplier from the document's
actual chunk statistics:

```
avg_qa_tokens ≈ 200  (empirical median across past runs; ranges 80-300
                      depending on prompt style and answer depth)
mean_chunk_tokens = mean of num_tokens across ALL chunks
multiplier = coverage × mean_chunk_tokens / avg_qa_tokens
num_samples = multiplier × num_chunks  (num_chunks = total, not filtered)
```

Present three coverage tiers and show the computed sample counts:

1) 3x source coverage — Good starting point
2) 5x source coverage — Recommended (research-backed default)
3) 8x source coverage — Best quality, thorough coverage

For example, a document with 50 chunks of mean 400 tokens:
- 3x: multiplier = 3 × 400 / 200 = 6x → 300 samples
- 5x: multiplier = 5 × 400 / 200 = 10x → 500 samples
- 8x: multiplier = 8 × 400 / 200 = 16x → 800 samples

Show the user the actual numbers, the multiplier, and the coverage
tier so they understand the reasoning.

### Step 6 — System prompt for the trained model

Do NOT ask the user to write a system prompt. Generate a domain-specific
default based on the document content and include it in the config.
Only mention it in the confirmation table so the user can adjust if
they want to.

## Building the Config

Follow the template structure (`templates/sdg/knowledge-ingestion.yaml`). Customize
prompts and system prompt for the user's domain.

### Submission

Submit SDG jobs via `create_sdg_job`. The tool accepts typed
parameters directly — no wrapper object needed:

### document_ids

List of document IDs from the Documents page. The worker fetches parsed
markdown from MLflow for processing.

```json
"document_ids": ["59d4ba25a8864e7fbbbb35cfc09603a1"]
```

### model_configs

Which LLM to use. Use `provider: "gateway"` to route through the MLflow
AI Gateway. Always set `skip_health_check: true` with the gateway.

```json
"model_configs": [{
  "alias": "text",
  "model": "gpt-oss",
  "provider": "gateway",
  "skip_health_check": true,
  "inference_parameters": {
    "temperature": 0.7,
    "max_parallel_requests": 32
  }
}]
```

### columns

Columns define the generation pipeline. Each column can reference prior
columns and seed data via `{{ variable_name }}`.

**Sampler columns** — include difficulty and question_type. Each sampler
value MUST include a description after the name, separated by " - ".
This gives the LLM richer context for generation:

```json
{
  "column_type": "sampler",
  "name": "question_type",
  "sampler_type": "category",
  "params": {
    "values": [
      "Factual - Understanding what something is, why it works, or how components relate",
      "Procedural - Step-by-step question about accomplishing a specific task"
    ],
    "weights": [0.6, 0.4]
  }
}
```

Apply the same pattern to the difficulty sampler. See the template for
the full default values.

Do NOT include a `topic` sampler. The chunk content (`{{ content }}`)
determines what each QA pair is about — an independent topic sampler
creates mismatches where the LLM ignores the context and hallucinates
answers based on the topic instead.

**LLM text columns** — question and answer generators:

```json
{
  "column_type": "llm-text",
  "name": "question",
  "model_alias": "text",
  "system_prompt": "<domain-specific system prompt with answerability constraint>",
  "prompt": "Documentation context:\n{{ content }}\n\nDifficulty: {{ difficulty }}\nQuestion type: {{ question_type }}"
}
```

**Prompt variable format — ALWAYS place each variable on its own line
with a label:**

```
Difficulty: {{ difficulty }}
Question type: {{ question_type }}
```

Do NOT embed variables inline in a sentence like
"Generate a {{ difficulty }} {{ question_type }} question".
Separate lines make each attribute more salient to the LLM.

### processors — OUTPUT FORMAT

Use `schema_transform` to convert columns into SFT training format:

```json
"processors": [{
  "processor_type": "schema_transform",
  "name": "sft_format",
  "template": {
    "messages": [
      {"role": "system", "content": "<domain-specific system prompt for the trained model>"},
      {"role": "user", "content": "{{ question }}"},
      {"role": "assistant", "content": "{{ answer }}"}
    ]
  }
}]
```

Include columns in the processor template that would be useful for
post-analysis (e.g., source context, generation parameters). Training
uses `messages`; extra columns are ignored by the trainer but
preserved in the artifact for inspection. Default to preserving
rather than dropping.

The system prompt here defines how the TRAINED MODEL should behave at
inference time. It must be domain-specific and match what the user's
deployment will use.

## Prompt Engineering Rules

**Question system prompt — MUST include answerability constraint:**
"The question MUST be fully answerable using ONLY the provided documentation
context. Do not ask about concepts that are merely mentioned but not explained
in the context."

**Answer system prompt — ONLY from context:**
Include: "Answer ONLY using information from the provided documentation
context. Do not add commands, procedures, or details that are not
explicitly present in the context."
Do NOT include: "Include specific commands, YAML snippets" — this
encourages the model to fabricate details not in the context.
Do NOT include: "If the documentation does not cover the topic, say so." —
this teaches the model to refuse, which is undesirable for FAQ assistants.

**Groundedness is critical.** The generated QA pairs train a model.
Answers that fabricate details — even plausible ones — teach the model
to hallucinate. A short answer that only uses what's in the context is
better than a long answer that adds invented details. Do not add
trailing instructions like "Provide a thorough answer" or "Generate a
question that matches" to the user prompt — the system prompt is
sufficient. Keep user prompts minimal: context + variables only.

**Domain-specific:** Adapt both prompts to the user's domain. Reference the
specific product/technology name in the system prompts, not generic
"documentation" references.

## Quality Checklist

Before submitting the job, verify:

- [ ] No `topic` sampler — chunk content drives QA subject matter
- [ ] Prompt variables (`difficulty`, `question_type`) are on separate lines with labels
- [ ] Question system prompt includes the answerability constraint
- [ ] Answer system prompt includes "Answer ONLY using information from the provided documentation context"
- [ ] System prompt in the SFT processor is domain-specific
- [ ] `num_records` computed from chunk statistics (coverage x mean_chunk_tokens / avg_qa_tokens x num_chunks)

## After SDG — Training

Recommend OSFT training. Read `skills/training/knowledge-ingestion/osft/guide.md` for the
training config. The SDG job's output (stored in MLflow) becomes the
training job's `data_path` via parent job chaining.
