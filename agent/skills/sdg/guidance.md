# SDG Skill Guidance

Pick the sub-skill that best matches the user's task. Read its `guide.md`
for the requirement-gathering workflow and reference payload.

## Available Sub-Skills

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| knowledge-ingestion | `skills/sdg/knowledge-ingestion/` | FAQ bots, QA assistants, doc-grounded chat, RAG models |
| classification | `skills/sdg/classification/` | Ticket classifiers, intent routers, sentiment analysis, content moderation |
| task-distillation | `skills/sdg/task-distillation/` | RFE assessors, code reviewers, compliance checkers, rubric-based scoring |

## How to Choose

- **User has documents they want a model to answer questions about** -> `knowledge-ingestion`
- **User wants to sort/label/categorize text** -> `classification`
- **User wants to replicate a frontier model's scoring/assessment behavior** -> `task-distillation`

## Calling `create_sdg_job`

The tool has full Pydantic validation from Data Designer's own types.
Each sub-skill provides a reference payload -- use it as the base and
adapt fields to the user's requirements.

### Model Configs

Always use the gateway provider with these defaults:

```json
{
  "alias": "text",
  "model": "<from list_models>",
  "provider": "gateway",
  "skip_health_check": true,
  "inference_parameters": {
    "temperature": 0.7,
    "max_parallel_requests": 32
  }
}
```

`max_parallel_requests` controls concurrent LLM calls. 32 is the
default for fast generation. Do not omit this field.

### Columns

Columns define the generation pipeline. Each column can reference prior
columns and seed data via `{{ variable_name }}`. Evaluated in order.

**Sampler columns** -- every sampler value MUST include a description
after the name, separated by " - ":

```
"Factual - Understanding what something is or how components relate"
```

**LLM text columns** -- prompt variables MUST be on separate labeled
lines, not inline in a sentence:

```
Difficulty: {{ difficulty }}
Question type: {{ question_type }}
```

NOT: "Generate a {{ difficulty }} {{ question_type }} question"

### Processors

Use `schema_transform` to produce SFT `messages` format. Include extra
columns in the template for post-analysis -- the trainer ignores them
but they're preserved in the artifact for inspection.

### Prompt Engineering

- **Groundedness**: generated data trains a model. Answers that fabricate
  details teach hallucination. A short answer from context is better
  than a long answer with invented details. Constrain to source content.
- **Domain-specific**: adapt prompts to the user's domain. Reference the
  specific product/technology name, not generic "documentation".
- **Minimal user prompts**: context + variables only. No trailing
  instructions like "Provide a thorough answer" -- the system prompt
  is sufficient.
