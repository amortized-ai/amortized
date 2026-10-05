# Embedding Classifier — Data Generation Guide

Use this guide to generate data for a **compact embedding classifier / router**
(trained with `embedding_sft`), as opposed to a generative SFT classifier. The
output is labeled `(text, category)` examples — one row per example, no
`messages` transform.

## When to use this vs. the SFT classification guide

- **Embedding classifier** (this guide): trains a small sentence-transformers
  model so same-category texts cluster; classify by nearest class at inference.
  Fast, cheap, tiny model, great for routing/intent/topic. Data: `(text, category)`.
- **SFT classification** (`classification/guide.md`): fine-tunes an LLM to emit
  the label as text. Data: chat `messages`. Use when you need an LLM anyway.

## How this works

Gather requirements, then call `validate_sdg_job` (pull the exact field structure
from its schema — don't restate it here). The config samples a `category` per row
and generates a realistic `text` for it. Output parquet has `category` (string)
and `text` columns — feed it straight to training with `algorithm="embedding_sft"`,
`text_column="text"`, `label_column="category"` (embedding_sft label-encodes string
categories to integers automatically).

## Requirement gathering

Follow the workflow's one-question-per-turn rules. Gather, in order:

1. **Domain** — what content will the classifier handle? (e.g. support tickets,
   user intents, log lines). Folds into the `text` column's `system_prompt`.
2. **Categories** — the class labels (3–8 works well), lowercase snake_case. These
   become the `category` sampler's values.
3. **Seed examples (important — this sets the style)** — ask the user to paste a
   few **real** example messages for each category (2–5 each is plenty), **exactly
   as they actually appear** — including shorthand, abbreviations, informal
   phrasing, or typos. Fold them into the `text` column's `system_prompt` grouped
   by category. This grounds the generation so synthetic data matches the real
   distribution instead of the model's guess. Do NOT invent the style yourself. If
   the user genuinely has no examples, say quality will be lower and offer to
   proceed with the model's best guess (or to look at a real sample first).
4. **Teacher model** — per the workflow's `list_models` rules.
5. **Samples** — recommend ≥ 100 per category (`num_records = categories × 100`).
   Then also generate a second, smaller run (different seed, same categories) as a
   held-out **eval** dataset.

## Minimal worked example

The structure is standard `validate_sdg_job` (see the knowledge-ingestion guide
for the general shape); only two things are specific to this skill — a `category`
**sampler** and a single `llm-text` column with the seed examples folded into the
`system_prompt`. Note `processors: []` — unlike the other skills, there is **no
`schema_transform`**; we want raw `text,category`, not `messages`.

```json
{
  "num_records": 300,
  "columns": [
    {
      "column_type": "sampler",
      "name": "category",
      "sampler_type": "category",
      "params": { "values": ["<cat_a>", "<cat_b>", "<cat_c>"] }
    },
    {
      "column_type": "llm-text",
      "name": "text",
      "model_alias": "text",
      "system_prompt": "You generate realistic messages that users actually send in the <domain> setting. Match the real style, brevity, vocabulary, and phrasing of the example messages for each category below — including any shorthand, abbreviations, or typos. Do NOT make messages cleaner or longer than the examples, and do not name the category in the text. Output ONLY the message text.\n\nExample messages by category:\n- <cat_a>: \"<seed>\"; \"<seed>\"\n- <cat_b>: \"<seed>\"; \"<seed>\"\n- <cat_c>: \"<seed>\"; \"<seed>\"",
      "prompt": "Generate ONE new, distinct message for the category: {{ category }}. Match the register and phrasing of the '{{ category }}' examples above; vary the wording — do not copy any example verbatim."
    }
  ],
  "processors": []
}
```

This snippet is intentionally partial — `model_configs` (defining the `text`
alias → teacher model/provider) is omitted; add it per the teacher-model step.
`num_records` is illustrative (3 categories × 100); compute it from the real
category count.

Grounding the generation in the user's real examples (rather than a synthetic
"style" knob) is what makes the classifier learn the *actual* distribution — and
it's what lets fine-tuning beat the base model on the messy real phrasings.

## Downstream

- **Train**: `create_training_job` with `algorithm="embedding_sft"`,
  `model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"` (or another
  sentence-transformers model), `parent_job_id`/`data_run_id` = this dataset.
- **Eval**: `create_eval_job` with `eval_mode="classification"`,
  `training_job_id` = the training job, and the held-out eval dataset as the
  parent/`eval_data_run_id`. Reports accuracy / macro-F1 / confusion.
