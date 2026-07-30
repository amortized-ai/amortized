# Knowledge Ingestion — SDG Guide

Use this guide when building FAQ bots, QA assistants, document-grounded chat,
or RAG-deployed knowledge models.

## How This Works

You will create a Data Designer config for the user's specific domain and
documents. Before starting, read BOTH of these files from your skills directory:

1. `skills/sdg/knowledge-ingestion/sdg-recipe-template.json` — the config
   structure and field reference
2. `skills/sdg/knowledge-ingestion/sdg-recipe-example.json` — a worked
   example with `_annotations` explaining WHY each value was chosen

The example shows what production-quality values look like. Read its
`_annotations` carefully — they explain the reasoning behind topic
specificity, prompt structure, and weight distribution. Apply the same
reasoning to the user's domain.

## Requirement Gathering

Ask the user these questions (one at a time, using `present_options`):

### Step 1 — What documents?

Check if they've already uploaded documents via the Documents page. Use
`list_documents` to show available documents with their IDs. If not
uploaded yet, guide them to upload first.

### Step 2 — Read the document and derive topics

After the user selects a document, call `get_document_content` with the
document ID to read its full parsed content. Then:

1. Scan the markdown for headings (##, ###) and major sections
2. Derive 5-9 topic values as **section-level descriptions**, not
   abstract categories

**CRITICAL: Topics must be section-level descriptions.**
- GOOD: "Tier Management - Creating, editing, deleting tiers via dashboard, configuring token rate limits and request rate limits"
- BAD: "access control"

Each topic value should name the section AND summarize its key content
in one line. This forces the LLM to ask specific questions about the
details in that section rather than repeating generic questions.

3. Weight topics proportionally to section length — longer/denser
   sections get higher weights. Do NOT use equal weights.
4. Present the derived topics to the user for confirmation. Let them
   adjust, add, or remove topics.

See the `_annotations.topics` and `_annotations.topic_weights` in the
example recipe for the reasoning behind this.

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

### Step 6 — Chunking granularity

How many sentences per chunk?
Default: 15 sentences. More = broader context, fewer = more focused.
Explain that this controls how documents are split for QA generation.

### Step 7 — How many samples?

Calculate based on document coverage.

**CRITICAL: The minimum sample count = number of document chunks.**
Each chunk gets at least one QA pair. If a document produces 100 chunks,
the minimum is 100 samples — anything less means parts of the document
won't be covered at all.

Estimate chunk count: document_char_count / (sentences_per_chunk x ~100 chars/sentence).
Use the document content length from `get_document_content` to estimate.

Present options relative to the chunk count:
1) N samples — Full coverage (1 QA per chunk, minimum recommended)
2) Nx2 samples — Double coverage (2 QAs per chunk, better diversity)
3) Nx3 samples — Triple coverage (3 QAs per chunk, best quality)

NEVER suggest a sample count below the estimated chunk count.
Explain to the user WHY: "With ~N chunks from your document, we need
at least N samples to cover every section."

### Step 8 — System prompt for the trained model

Ask the user what system prompt their deployed model should use.
This goes in the processor's SFT output and defines the trained model's
behavior at inference time. Suggest a domain-specific default based on
the document content.

## Building the Config

Follow the template structure. Customize topic values, prompts, and
system prompt for the user's domain. Read the example recipe to see
what production-quality values look like.

### Structure

The SDG job config is a Data Designer config submitted as:

```json
{
  "type": "sdg",
  "config": {
    "document_ids": [...],
    "num_records": N,
    "seed_config": { "source": { ... } },
    "model_configs": [ ... ],
    "columns": [ ... ],
    "processors": [ ... ]
  }
}
```

### document_ids

List of document IDs from the Documents page. The worker fetches parsed
markdown from MLflow and feeds it to DD's DocumentChunkerSeedSource.

```json
"document_ids": ["59d4ba25a8864e7fbbbb35cfc09603a1"]
```

### seed_config

Controls how documents are chunked. Each chunk becomes `{{ text }}` in
column prompts.

```json
"seed_config": {
  "source": {
    "sentences_per_chunk": 15,
    "min_text_length": 100
  }
}
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
  "inference_parameters": { "temperature": 0.7 }
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
  "prompt": "Documentation context:\n{{ text }}\n\nTopic: {{ topic }}\nDifficulty: {{ difficulty }}\nQuestion type: {{ question_type }}"
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

- [ ] Topics are section-level descriptions derived from `get_document_content`, not abstract categories
- [ ] Topic weights are proportional to section content density, not equal
- [ ] Prompt variables (`topic`, `difficulty`, `question_type`) are on separate lines with labels
- [ ] Question system prompt includes the answerability constraint
- [ ] Answer system prompt includes "answerable from the provided context"
- [ ] System prompt in the SFT processor is domain-specific
- [ ] `num_records` >= estimated chunk count

## After SDG — Training

Recommend OSFT training. Read `skills/training/knowledge-ingestion/osft/guide.md` for the
training config. The SDG job's output (stored in MLflow) becomes the
training job's `data_path` via parent job chaining.
