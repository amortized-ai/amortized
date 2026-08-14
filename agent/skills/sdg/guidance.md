# SDG Skill Guidance

Pick the sub-skill that best matches the user's task. Read its `guide.md`
for deep expertise before building the config.

## Available Sub-Skills

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| knowledge-ingestion | `skills/sdg/knowledge-ingestion/` | FAQ bots, QA assistants, doc-grounded chat, RAG models |
| classification | `skills/sdg/classification/` | Ticket classifiers, intent routers, sentiment analysis, content moderation |

## How to Choose

- **User has documents they want a model to answer questions about** → `knowledge-ingestion`
- **User wants to sort/label/categorize text** → `classification`

## Teacher Model Selection

ONLY show models returned by the gateway. If no models are returned,
**stop the workflow** and tell the user to go to Settings → AI Gateway.

1. Discover available models from the gateway
2. Look up pricing for EVERY model — try the most specific name part
   first, broaden if no results
3. Show a pricing comparison card with all collected pricing data
4. Present each model as an option with pricing in the description.
   Use the endpoint `name` as the display label everywhere
5. Wait for the user to select — NEVER auto-select, even if only one

## Dataset Inspection

When the user asks about their datasets or wants to compare them:

1. List available datasets (filter by name or topic if specified)
2. Preview actual rows — show 2-3 representative samples

When an SDG job succeeds, preview the generated data using the job's
`mlflow_run_id` so the user can verify quality before training.

## SDG Confirmation

Before submitting, look up pricing for the selected model to show
cost context.

## SDG Defaults

Always include in `model_configs` inference_parameters:

```json
"inference_parameters": {
  "temperature": 0.7,
  "max_parallel_requests": 32
}
```

## SDG Preview Flow

Call the validation tool with `mode: "preview"` first for a ~10 sample
test run. Once the preview succeeds and the user is happy, call again
with `mode: "create"` for the full run. NEVER call with `mode: "create"`
more than once per conversation for the same job.
