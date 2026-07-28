# Knowledge Ingestion — SDG Guide

Use this guide when building FAQ bots, QA assistants, document-grounded chat,
or RAG-deployed knowledge models.

## How This Works

You will **create a brand new Data Designer config** from scratch based on
the user's requirements. The reference template at
`skills/sdg/knowledge-ingestion/sdg-recipe-template.json` shows the config
structure — study it to understand the format, but do NOT just fill in
placeholders. Build a fresh config tailored to the user's domain, documents,
and needs.

Every part of the config is dynamic:
- The columns (samplers, LLM prompts) are created based on the user's task
- The prompts are written for the user's specific domain
- The sampler values and weights reflect the user's content distribution
- The system prompt in the processor reflects how the trained model should behave

## Requirement Gathering

Ask the user these questions (one at a time, with numbered options):

1. **What documents?** — Check if they've already uploaded documents via the
   Documents page. Use `list_documents` to show available documents with
   their IDs. If not uploaded yet, guide them to upload first.
2. **What topics?** — What are the main topic areas in the documentation?
   Suggest 5-9 topics weighted by documentation depth per section.
3. **Question types** — What kinds of questions should it handle?
   Default: factual (25%), procedural (35%), troubleshooting (25%),
   comparison (15%). But adapt to the domain — a troubleshooting guide
   needs more troubleshooting questions, a reference manual needs more
   factual ones.
4. **Difficulty levels** — Default: basic (35%), intermediate (45%),
   advanced (20%).
5. **Which teacher model?** — Call `list_models` to discover available
   models from the AI Gateway. Present each as a numbered option.
   If no models are returned, direct the user to Settings → AI Gateway.
6. **How many samples?** — Suggest based on document size.
   ~50 for prototyping, ~500 for a small production model, ~3000+ for
   comprehensive coverage.
7. **Chunking granularity** — How many sentences per chunk?
   Default: 15 sentences. More = broader context, fewer = more focused.

## Building the Config

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

### columns — CREATE THESE FROM SCRATCH

Columns define the generation pipeline. Each column can reference prior
columns and seed data via `{{ variable_name }}`. Build columns that match
the user's requirements — don't copy the template verbatim.

**Sampler columns** — weighted random selection for variation axes:

```json
{
  "column_type": "sampler",
  "name": "<your_name>",
  "sampler_type": "category",
  "params": {
    "values": ["value1", "value2", "value3"],
    "weights": [0.4, 0.35, 0.25]
  }
}
```

Create samplers for the dimensions that matter for this user's task.
Common dimensions: difficulty, question_type, topic, user_role, scenario.
Use `subcategory` sampler type when you need value-dependent descriptions.

**LLM text columns** — LLM generates text from a prompt:

```json
{
  "column_type": "llm-text",
  "name": "<your_name>",
  "model_alias": "text",
  "system_prompt": "<domain-specific system prompt>",
  "prompt": "<prompt using {{ variables }} from prior columns and {{ text }} from seed>"
}
```

Write system prompts and prompts specifically for the user's domain.
Always include `{{ text }}` to reference the document chunk.

### processors — OUTPUT FORMAT

Use `schema_transform` to convert columns into SFT training format:

```json
"processors": [{
  "processor_type": "schema_transform",
  "name": "sft_format",
  "template": {
    "messages": [
      {"role": "system", "content": "<training system prompt>"},
      {"role": "user", "content": "{{ question }}"},
      {"role": "assistant", "content": "{{ answer }}"}
    ]
  }
}]
```

The system prompt here defines how the TRAINED MODEL should behave —
write it for the user's specific use case.

## Prompt Engineering Tips

**Question generation — enforce answerability:**
"The question MUST be fully answerable using ONLY the provided documentation
context."

**Answer generation — trust the context:**
"The question is answerable from the provided context, so read it carefully
and answer from it."

**Domain-specific:** Adapt both prompts to the user's domain. A medical QA
bot needs different prompt engineering than a DevOps troubleshooting assistant.

## After SDG — Training

Recommend OSFT training. Read `skills/training/knowledge-ingestion/osft/guide.md` for the
training config. The SDG job's output (stored in MLflow) becomes the
training job's `data_path` via parent job chaining.
