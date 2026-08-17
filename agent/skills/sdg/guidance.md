# SDG Skill Guidance

## Task Routing

Map the user's task pattern to the right approach:

| Pattern | Examples | Route |
|---------|----------|-------|
| Knowledge QA | FAQ bots, doc chat, support assistants | `knowledge-ingestion` guide |
| Classification | ticket routing, intent, sentiment, moderation | `classification` guide |
| Summarization | meeting notes, ticket digests, briefs | Custom config |
| Extraction | entity parsing, field extraction, structured output | Custom config |
| Transformation | rewriting, translation, style transfer | Custom config |
| Scoring | quality grading, relevance, moderation scores | Custom config |
| Code generation | SQL from NL, config generation, templating | Custom config |
| Routing | query dispatch, escalation, workflow branching | `classification` variant |

**If a sub-skill guide exists**, load it and follow its requirement-
gathering steps:

| Sub-Skill | Path |
|-----------|------|
| knowledge-ingestion | `skills/sdg/knowledge-ingestion/` |
| classification | `skills/sdg/classification/` |

**If no guide exists** (custom config), gather requirements yourself
and construct the config from the column primitives below.

## Custom Config Requirements

When building a custom config (no sub-skill guide), gather at minimum:
1. Source data — documents, or describe the input domain
2. Output structure — what should each training example look like?
3. Variety dimensions — what should vary across samples? (tone, difficulty, length, style)
4. Teacher model — which frontier model generates the data
5. Volume — how many samples

Ask these ONE AT A TIME via `present_options`.

## Building Custom Configs

When no sub-skill guide exists — summarization, extraction,
transformation, scoring, code generation, routing — you construct
the SDG config from column primitives. The Data Designer pipeline is
a DAG of columns: each column can reference prior columns and seed
data via `{{ variable_name }}`.

### Column Types

| Type | Purpose | When to Use |
|------|---------|-------------|
| `sampler` | Inject categorical variety | Difficulty, tone, style, length, category — any dimension that should vary across samples |
| `llm-text` | Free-form LLM generation | Questions, answers, summaries, rewrites — the workhorse for text output |
| `llm-structured` | JSON-schema-constrained LLM output | Entity extraction, structured fields, typed output — when the output must conform to a schema |
| `llm-judge` | LLM-as-judge scoring | Quality filtering, relevance grading, factuality checks — when you need to score or filter generated data |
| `llm-code` | Code generation | SQL, config files, scripts — when the output is code |
| `expression` | Python expressions over prior columns | Computed fields, string formatting, conditional logic |
| `validation` | Quality gates | Filter low-quality samples before they enter the training set |

### Config Patterns for Common Tasks

**Summarization** (meeting notes, ticket digests, document briefs):
- Samplers: `summary_style` ("executive brief", "bullet points",
  "technical abstract"), `length` ("1 paragraph", "3 sentences")
- LLM columns: generate summary from `{{ content }}` chunk, conditioned
  on style and length
- Processor: `schema_transform` → messages with system prompt for the
  summarizer model

**Extraction** (entities, fields, structured data from text):
- Samplers: `entity_type` or `field_set` if variety is needed
- LLM columns: use `llm-structured` with a JSON schema defining the
  output fields — enforces structure at generation time
- Processor: `schema_transform` → messages where assistant response
  is the structured JSON

**Transformation** (rewriting, translation, style transfer):
- Samplers: `target_style`, `formality_level`, `audience`
- LLM columns: generate source text (or use `{{ content }}`), then
  generate transformed version referencing the source
- Processor: `schema_transform` → messages with (input, transformed output)

**Scoring / Evaluation** (quality, relevance, moderation):
- Samplers: `content_category`, `expected_score_range`
- LLM columns: generate content to score, then use `llm-judge` to
  produce a score with rationale
- Processor: `schema_transform` → messages where assistant produces
  the score

**Routing** (query dispatch, escalation, workflow branching):
- This is a classification variant. Use the `classification` sub-skill
  guide with destination categories instead of topic labels.

**Code Generation** (SQL from NL, configs, templates):
- Samplers: `complexity` ("simple select", "multi-join", "aggregation")
- LLM columns: use `llm-code` for the code output, `llm-text` for the
  natural language description
- Processor: `schema_transform` → messages with (description, code)

### Constructing Any Config

Every SDG config needs these building blocks:

1. **At least one sampler** for variety — without it, every sample looks
   the same. Think about what dimensions should vary: difficulty, type,
   tone, length, category, complexity.
2. **One or more LLM columns** for generation — these do the actual work.
   Each column gets a `system_prompt` (who the LLM is) and a `prompt`
   (what to generate), referencing prior columns via `{{ name }}`.
3. **A processor** to format output — always `schema_transform` to
   produce `messages` format for TRL compatibility.

If the task uses documents, `{{ content }}` injects document chunks.
Always ground generation in the actual content — do NOT let the LLM
hallucinate from the topic alone.

**Prompt variable format** — ALWAYS place each variable on its own line
with a label:
```
Difficulty: {{ difficulty }}
Style: {{ summary_style }}
```
Do NOT embed variables inline in sentences. Separate lines make each
attribute more salient to the LLM.

## Teacher Model Selection

ONLY show models returned by the gateway. If no models are returned,
**stop the workflow** and tell the user to go to Settings → AI Gateway.

1. Call `list_models` to discover available models
2. Call `get_model_pricing` for EVERY model — try the most specific
   name part first, broaden if no results
3. Call `show_model_pricing` ONCE with all collected pricing data
4. Call `present_options` with each model as an option, pricing in the
   description. Use the endpoint `name` as the display label everywhere
5. Wait for the user to select — NEVER auto-select, even if only one

## Dataset Inspection

When the user asks about their datasets or wants to compare them:

1. Call `list_datasets` (filter by name or topic if specified)
2. Call `get_dataset_samples` to preview actual rows — show 2-3 samples

When an SDG job succeeds, call `get_dataset_samples` with the job's
`mlflow_run_id` so the user can verify quality before training.

## SDG Preview Flow

Call `validate_sdg_job` with `mode: "preview"` first for a ~10 sample
test run. Once the preview succeeds and the user is happy, call again
with `mode: "create"` for the full run. NEVER call with `mode: "create"`
more than once per conversation for the same job.

## SDG Confirmation

Before calling `validate_sdg_job`, call `get_model_pricing` with the
selected model name to show cost context.

## SDG Defaults (all configs)

Always include these in `model_configs` inference_parameters:

```json
"inference_parameters": {
  "temperature": 0.7,
  "max_parallel_requests": 32
}
```

`max_parallel_requests` controls how many LLM calls run concurrently.
32 is the default for fast generation. Do not omit this field.
