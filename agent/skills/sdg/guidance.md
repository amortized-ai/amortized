# SDG Skill Guidance

Pick the sub-skill that best matches the user's task. Read its `guide.md`
for deep expertise before calling `create_sdg_job`.

## Available Sub-Skills

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| knowledge-ingestion | `skills/sdg/knowledge-ingestion/` | FAQ bots, QA assistants, doc-grounded chat, RAG models |
| classification | `skills/sdg/classification/` | Ticket classifiers, intent routers, sentiment analysis, content moderation |

## How to Choose

- **User has documents they want a model to answer questions about** → `knowledge-ingestion`
- **User wants to sort/label/categorize text** → `classification`

## After Loading the Sub-Skill

The sub-skill's `guide.md` will tell you:
- What questions to ask the user
- What parameters to pass to `create_sdg_job`
- How to write effective prompts for columns
- Key tradeoffs and decisions

Follow the guide's recommendations, but the user has the final call
on every parameter. Always confirm before submitting.

## SDG Defaults (all sub-skills)

Always include these in `model_configs` inference_parameters:

```json
"inference_parameters": {
  "temperature": 0.7,
  "max_parallel_requests": 32
}
```

`max_parallel_requests` controls how many LLM calls run concurrently.
32 is the default for fast generation. Do not omit this field.
