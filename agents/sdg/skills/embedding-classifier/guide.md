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

## Requirement gathering (STRICT one-question-per-turn)

Ask these **one at a time**. Send ONE question, then **STOP and wait for the
user's reply** before asking the next. Never write the user's answer yourself,
never simulate a "user:" turn, and never advance to later steps (or to preview /
`validate_sdg_job`) in the same message — output only the current question and
end your turn. Use the user's real answers; if they add a requirement (e.g. ESL
typos), acknowledge it and fold it into the prompt, then continue from where you
are — do not restart.

1. **Domain** — what content will the classifier handle? (e.g. support tickets,
   user intents, log lines). Substitute it for `[DOMAIN]` in the text column's
   `system_prompt`. **STOP — wait for the reply.**
2. **Categories** — the class labels (3–8 works well). Put them in the `category`
   sampler's `params.values` (lowercase, snake_case). These become the classes.
   **STOP — wait for the reply.**
3. **Teacher model** — call `list_models`; present ONLY returned models. In
   `model_configs`, set `model` to the model's exact **`name`** field verbatim
   (e.g. `gpt-oss`) and `provider` to its `provider` field (e.g. `gateway`) —
   do NOT use the `model_name` field or a provider-prefixed id like
   `openai/gpt-oss-120b`; the gateway resolves the short `name`, and a fuller id
   fails with "model could not be found". If none are returned, stop (no teacher
   configured). **STOP — wait for the reply.**
4. **Samples** — recommend ≥ 100 per category (`num_records = categories × 100`).
   **STOP — wait for the reply.** Only after the user answers, proceed to build
   the config and run the preview. Also generate a second, smaller run (different
   topic/seed, same categories) as a held-out **eval** dataset for the eval.

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
