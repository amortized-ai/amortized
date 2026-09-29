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

Gather requirements, then call `validate_sdg_job`. The config samples a
`category` per row and generates a realistic `text` for it. Output parquet has
`category` (string) and `text` columns — feed it straight to training with
`algorithm="embedding_sft"`, `text_column="text"`, `label_column="category"`
(embedding_sft label-encodes string categories to integers automatically).

## Requirement gathering (one question at a time, numbered options)

1. **Domain** — what content will the classifier handle? (e.g. support tickets,
   user intents, log lines). Substitute it for `[DOMAIN]` in the text column's
   `system_prompt`.
2. **Categories** — the class labels (3–8 works well). Put them in the `category`
   sampler's `params.values` (lowercase, snake_case). These become the classes.
3. **Teacher model** — call `list_models`; use a returned `name`/`provider` in
   `model_configs`. If none are returned, stop (no teacher configured).
4. **Samples** — recommend ≥ 100 per category (`num_records = categories × 100`).
   Generate a second, smaller run (different topic/seed, same categories) as a
   held-out **eval** dataset for the classification eval.

## Tool parameters

Call `validate_sdg_job` with the columns/model_configs/processors from
`reference-payload.json`, customizing:
- `category` sampler `params.values` → the user's categories (optionally add
  `weights` for a non-uniform distribution).
- `text` column `system_prompt` → replace `[DOMAIN]`.
- `model_configs[0].model`/`provider` → from `list_models`.
- `processors` → leave `[]` (no `messages` transform — we want raw `text,category`).

## Downstream

- **Train**: `create_training_job` with `algorithm="embedding_sft"`,
  `model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"` (or another
  sentence-transformers model), `parent_job_id`/`data_run_id` = this dataset.
- **Eval**: `create_eval_job` with `eval_mode="classification"`,
  `training_job_id` = the training job, and the held-out eval dataset as the
  parent/`eval_data_run_id`. Reports accuracy / macro-F1 / confusion.
