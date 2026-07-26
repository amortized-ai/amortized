# Knowledge Ingestion — SDG Guide

Use this guide when building FAQ bots, QA assistants, document-grounded chat,
or RAG-deployed knowledge models. The template and guidelines here are
suggestions — the user has the final call on each knob.

## Requirement Gathering

Ask the user these questions (one at a time, with numbered options):

1. **What documents?** — Check if they've already uploaded documents via the
   Documents page. If so, use `list_documents` to show available documents
   with their IDs. If not, guide them to upload first.
2. **What topics?** — What are the main topic areas in the documentation?
   Suggest 5-9 topics weighted by documentation depth per section.
3. **Question types** — What kinds of questions should it handle?
   Default: factual (25%), procedural (35%), troubleshooting (25%),
   comparison (15%)
4. **Difficulty levels** — Default: basic (35%), intermediate (45%),
   advanced (20%)
5. **How many samples?** — Suggest based on document size.
   ~50 for prototyping, ~500 for a small production model, ~3000+ for
   comprehensive coverage.
6. **Chunking granularity** — How many sentences per chunk?
   Default: 15 sentences. More = broader context, fewer = more focused.

## SDG Job Configuration

SDG jobs use NVIDIA Data Designer as the generation engine. The job config
is a Data Designer config with columns, model_configs, and processors.

### Key Config Fields

| Field | What it does |
|-------|-------------|
| `document_ids` | List of document IDs from the Documents page. Worker fetches parsed content from MLflow and feeds it to DD's DocumentChunkerSeedSource |
| `num_records` | Total QA pairs to generate |
| `seed_config.source.sentences_per_chunk` | Sentences per document chunk (default 15) |
| `seed_config.source.min_text_length` | Minimum chars to keep a chunk (default 100) |
| `model_configs` | Which LLM to use. Use `provider: "gateway"` to route through MLflow AI Gateway |
| `columns` | Pipeline steps: samplers for variation, llm-text for question/answer generation |
| `processors` | Output transform: `schema_transform` converts to SFT `messages` format |

### Column Types

- **`sampler`** with `sampler_type: "category"` — weighted random selection
  for difficulty, question_type, topic
- **`llm-text`** — LLM generates text from a prompt with `{{ variable }}`
  template references to prior columns and seed data
- **`expression`** — Jinja2 template combining columns (avoid for JSON
  output — use `schema_transform` processor instead)

### Chunking

Document chunking is handled by DD's `DocumentChunkerSeedSource`. The
worker writes the parsed document content to a file, and DD chunks it
into sentence groups automatically. Each chunk becomes the `{{ text }}`
variable in column prompts.

| Param | Default | What it does |
|-------|---------|-------------|
| `sentences_per_chunk` | 5 | Sentences per chunk. 15 ≈ 2048 tokens |
| `min_text_length` | 0 | Drop chunks shorter than N chars |
| `multi_doc` | false | Enable cross-document questions |

### Output Format

Use a `schema_transform` processor to convert raw columns into SFT
training format:

```json
{
  "processor_type": "schema_transform",
  "name": "sft_format",
  "template": {
    "messages": [
      {"role": "system", "content": "<system prompt>"},
      {"role": "user", "content": "{{ question }}"},
      {"role": "assistant", "content": "{{ answer }}"}
    ]
  }
}
```

This produces proper `messages` arrays ready for TRL training.

### Prompt Engineering

**Question generation — enforce answerability:**
"The question MUST be fully answerable using ONLY the provided documentation
context. Do not ask about concepts that are merely mentioned but not
explained in the context."

**Answer generation — trust the context:**
"The question is answerable from the provided context, so read it carefully
and answer from it."

### Model Selection

Use the MLflow AI Gateway endpoint name as the model. Common patterns:
- `model: "gpt-oss"` with `provider: "gateway"` — self-hosted model
- `model: "gpt-4o-mini"` with `provider: "openai"` — OpenAI direct
- Always set `skip_health_check: true` when using the gateway

## Recommended Training Method

**OSFT is the default for knowledge ingestion.** It outperforms standard
SFT by 30+ percentage points in open-book settings. When the user is
ready to train, recommend reading `skills/training/osft/guide.md`.

## Config Template

The config template at `skills/sdg/knowledge-ingestion/sdg-recipe-template.json`
shows the parameterization for a knowledge-ingestion SDG job. Customize
the topics, question types, system prompts, and `document_ids` based
on the user's requirements.
