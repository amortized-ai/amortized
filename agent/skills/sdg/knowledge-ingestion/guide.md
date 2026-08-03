# Knowledge Ingestion — SDG Guide

Use this guide when building FAQ bots, QA assistants, document-grounded chat,
or RAG-deployed knowledge models.

## How This Works

You will create an SDG config for the user's specific domain
and documents. Before starting, read the template at
`skills/sdg/knowledge-ingestion/sdg-recipe-template.json` for the
config structure.

**Document analysis workflow:** Use `get_document_chunks(doc_id)` to
get the document's chunks with token counts and headings. Derive
topics from the chunk headings and content. Use `get_document_content`
if you need the full text for prompt writing.

**Keep it brief.** Do your analysis silently. Present results and ask
for confirmation — do not narrate your reasoning.

## Requirement Gathering

Ask the user these questions (one at a time, using `present_options`):

### Step 1 — What documents?

Check if they've already uploaded documents via the Documents page. Use
`list_documents` to show available documents with their IDs. If not
uploaded yet, guide them to upload first.

### Step 2 — Read the document and derive topics

After the user selects a document, call `get_document_chunks(doc_id)`
to get the chunks with their headings and token counts. Derive 5-9
topic values from the chunk headings. Then present them briefly:
"Here are the topics I suggest — proceed or adjust?" followed by
`present_options`.

**CRITICAL: Topics must be section-level descriptions.**
- GOOD: "Tier Management - Creating, editing, deleting tiers via dashboard, configuring token rate limits and request rate limits"
- BAD: "access control"

Each topic value should name the section AND summarize its key content
in one line. Use token counts from chunks to set weights — sections
with more content get higher weights. Weights must sum to 1.0 and no
topic should be below 0.05.

### Step 3 — Question types

What kinds of questions should it handle?
Default: factual (25%), procedural (35%), troubleshooting (25%),
comparison (15%). Adapt to the domain — a troubleshooting guide
needs more troubleshooting questions, a reference manual needs more
factual ones.

### Step 4 — Difficulty levels

Default: basic (35%), intermediate (45%), advanced (20%).

### Step 5 — Which teacher model?

Call `list_models` to get the models configured on the AI Gateway.
Present ONLY those models as options. Do NOT suggest models that aren't
returned by `list_models` — they won't work. If no models are returned,
stop and direct the user to Settings -> AI Gateway.

### Step 6 — How many samples?

Documents are chunked at upload time by docling-serve. Use
`get_document_chunks(doc_id)` to get the chunk count and token
statistics.

The goal is for total training tokens to be a multiple of total
source tokens. Research suggests ~5x source coverage as a good
target. Compute the per-chunk multiplier from the document's
actual chunk statistics:

```
avg_qa_tokens ≈ 200  (rough estimate — actual varies by domain)
median_chunk_tokens = median of num_tokens across all chunks
multiplier = coverage × median_chunk_tokens / avg_qa_tokens
num_samples = multiplier × num_chunks
```

Present three coverage tiers and show the computed sample counts:

1) 3x source coverage — Good starting point
2) 5x source coverage — Recommended (research-backed default)
3) 8x source coverage — Best quality, thorough coverage

For example, a document with 50 chunks of median 400 tokens:
- 3x: multiplier = 3 × 400 / 200 = 6x → 300 samples
- 5x: multiplier = 5 × 400 / 200 = 10x → 500 samples
- 8x: multiplier = 8 × 400 / 200 = 16x → 800 samples

A document with 20 chunks of median 2000 tokens:
- 3x: multiplier = 3 × 2000 / 200 = 30x → 600 samples
- 5x: multiplier = 5 × 2000 / 200 = 50x → 1000 samples
- 8x: multiplier = 8 × 2000 / 200 = 80x → 1600 samples

Show the user the actual numbers, the multiplier, and the coverage
tier so they understand the reasoning.

### Step 7 — System prompt for the trained model

Do NOT ask the user to write a system prompt. Generate a domain-specific
default based on the document content and include it in the config.
Only mention it in the confirmation table so the user can adjust if
they want to.

## Building the Config

Follow the template structure (`sdg-recipe-template.json`). Customize
topic values, prompts, and system prompt for the user's domain.

### Structure

The SDG job config is submitted as:

```json
{
  "type": "sdg",
  "config": {
    "document_ids": [...],
    "num_records": N,
    "model_configs": [ ... ],
    "columns": [ ... ],
    "processors": [ ... ]
  }
}
```

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

**Sampler columns** — always include difficulty, question_type, and topic:

```json
{
  "column_type": "sampler",
  "name": "topic",
  "sampler_type": "category",
  "params": {
    "values": ["Section-level description 1", "Section-level description 2"],
    "weights": [0.6, 0.4]
  }
}
```

**LLM text columns** — question and answer generators:

```json
{
  "column_type": "llm-text",
  "name": "question",
  "model_alias": "text",
  "system_prompt": "<domain-specific system prompt with answerability constraint>",
  "prompt": "Documentation context:\n{{ content }}\n\nTopic: {{ topic }}\nDifficulty: {{ difficulty }}\nQuestion type: {{ question_type }}"
}
```

**Prompt variable format — ALWAYS place each variable on its own line
with a label:**

```
Topic: {{ topic }}
Difficulty: {{ difficulty }}
Question type: {{ question_type }}
```

Do NOT embed variables inline in a sentence like
"Generate a {{ difficulty }} {{ question_type }} question about {{ topic }}".
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

The system prompt here defines how the TRAINED MODEL should behave at
inference time. It must be domain-specific and match what the user's
deployment will use.

## Prompt Engineering Rules

**Question system prompt — MUST include answerability constraint:**
"The question MUST be fully answerable using ONLY the provided documentation
context. Do not ask about concepts that are merely mentioned but not explained
in the context."

**Answer system prompt — trust the context:**
Include: "The question is answerable from the provided context, so read it
carefully and answer from it."
Do NOT include: "If the documentation does not cover the topic, say so." —
this teaches the model to refuse, which is undesirable for FAQ assistants.

**Domain-specific:** Adapt both prompts to the user's domain. Reference the
specific product/technology name in the system prompts, not generic
"documentation" references.

## Quality Checklist

Before submitting the job, verify:

- [ ] Topics are section-level descriptions derived from chunk headings, not abstract categories
- [ ] Topic weights are proportional to section content density, not equal
- [ ] Prompt variables (`topic`, `difficulty`, `question_type`) are on separate lines with labels
- [ ] Question system prompt includes the answerability constraint
- [ ] Answer system prompt includes "answerable from the provided context"
- [ ] System prompt in the SFT processor is domain-specific
- [ ] `num_records` computed from chunk statistics (coverage × median_chunk_tokens / avg_qa_tokens × num_chunks)

## After SDG — Training

Recommend OSFT training. Read `skills/training/knowledge-ingestion/osft/guide.md` for the
training config. The SDG job's output (stored in MLflow) becomes the
training job's `data_path` via parent job chaining.
