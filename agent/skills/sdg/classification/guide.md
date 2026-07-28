# Classification — SDG Guide

Use this guide when building ticket classifiers, intent routers, sentiment
analyzers, or content moderators.

## How This Works

You will **create a brand new Data Designer config** from scratch. The
classification pipeline generates labeled examples where each sample has
input text and a classification label.

## Requirement Gathering

Ask the user these questions (one at a time, with numbered options):

1. **What domain?** — What kind of content will this classifier handle?
   e.g., customer support tickets, user messages, content moderation.
2. **What categories?** — What labels should the classifier predict?
   Suggest 3-6 relevant labels based on the domain. Let the user
   customize or define their own.
3. **Urgency levels?** — Should the classifier also assign urgency?
   1) Yes, 3 levels — Low, Medium, High
   2) Yes, 4 levels — Low, Medium, High, Critical
   3) No urgency — Just classify by category
4. **Which teacher model?** — Call `list_models` to get the models
   configured on the AI Gateway. Present ONLY those models as options.
   Do NOT suggest models that aren't returned by `list_models` — they
   won't work. If no models are returned, stop and direct the user to
   Settings → AI Gateway.
5. **How many samples?** — Scale based on category count and desired
   coverage. Recommend at least 50 samples per category for basic
   coverage and 150+ per category for production quality.
   1) N×50 samples — Basic coverage across all categories
   2) N×100 samples — Good diversity, recommended
   3) N×150 samples — Best quality, most diverse examples
   (where N = number of categories × urgency levels)
6. **Distribution** — Should categories be balanced or weighted?
   Default: roughly balanced unless the real-world distribution is known.

## Building the Config

```json
{
  "type": "sdg",
  "config": {
    "num_records": 500,
    "model_configs": [{"alias": "text", "model": "<selected_model>", "provider": "gateway", "skip_health_check": true}],
    "columns": [
      {
        "column_type": "sampler",
        "name": "category",
        "sampler_type": "category",
        "params": {
          "values": ["<CATEGORY_1>", "<CATEGORY_2>", "<CATEGORY_3>"],
          "weights": [0.4, 0.35, 0.25]
        }
      },
      {
        "column_type": "llm-text",
        "name": "text",
        "model_alias": "text",
        "system_prompt": "<domain-specific prompt for generating realistic input text>",
        "prompt": "Generate a realistic {{ category }} example..."
      },
      {
        "column_type": "llm-text",
        "name": "label",
        "model_alias": "text",
        "system_prompt": "Classify the text. Output ONLY the label.",
        "prompt": "Text: {{ text }}\n\nClassify as one of: <categories>. Output ONLY the label."
      }
    ],
    "processors": [
      {
        "processor_type": "schema_transform",
        "name": "sft_format",
        "template": {
          "messages": [
            {"role": "system", "content": "<classification system prompt>"},
            {"role": "user", "content": "{{ text }}"},
            {"role": "assistant", "content": "{{ label }}"}
          ]
        }
      }
    ]
  }
}
```

Create columns, prompts, and categories based on the user's specific task.
Use the model name from `list_models` in the `model_configs`.
Submit via `create_job`.

## After SDG — Training

Recommend OSFT training. Read `skills/training/knowledge-ingestion/osft/guide.md`
for the training config. Chain via `parent_job_id`.
