---
name: training-embedding-classifier
description: >-
  Train a compact embedding classifier / router with embedding_sft (contrastive
  fine-tuning of a sentence-transformers model). For intent routing, ticket/topic
  classification, and sentiment — small, fast, and cheap to serve. The algorithm
  is fixed (no lora/qlora/osft choice) and there is no VRAM/model-size selection.
  Pairs with the sdg-embedding-classifier skill.
metadata:
  stage: training
---

# Embedding Classifier — Training Guide

Use this skill to train a **compact embedding classifier / router** with
`embedding_sft` (contrastive fine-tuning of a sentence-transformers model). Best
for intent routing, ticket/topic classification, and sentiment — small, fast, and
cheap to serve. Pair it with the **Embedding Classifier** SDG skill, which
produces `(text, category)` data.

This flow is different from the LLM training sub-skills (OSFT/SFT/LoRA):

- The algorithm is fixed to `embedding_sft` — there is **no method choice**
  (lora/qlora/osft/sft) to present.
- The base is a small **sentence-transformers** model, so **skip the VRAM cards**
  and student-model-size comparison — they don't apply. Just pick a base model.

## Requirement gathering

Follow the workflow's one-question-per-turn rules. Gather:

1. **Training data** — should come from a completed classification SDG job via
   `parent_job_id` (or a dataset `data_run_id`). If the orchestrator passed an SDG
   job ID, use it without asking. Set `label_column` to the category column name
   (`category` for the SDG skill) and `text_column` to the text column (`text`).
   String categories are label-encoded to integers automatically.
2. **Base model** — default `sentence-transformers/all-MiniLM-L6-v2` (small, fast,
   strong general-purpose). Only surface alternatives if the user asks (e.g. a
   multilingual or larger sentence-transformers model).
3. **Epochs** — default 25 for a few-hundred-example dataset; more for small
   datasets, fewer for large ones. Don't ask about learning rate / batch size /
   loss unless the user brings them up.

## Config

`validate_training_job` already exposes every field you need as a first-class
parameter — `algorithm`, `model_name_or_path`, `text_column`, `label_column`,
`num_train_epochs`, `parent_job_id`/`data_run_id`, and the optional
`loss_type` / `batch_sampler` / `warmup_ratio` / `seed`. Pull the structure from
that schema and set `algorithm="embedding_sft"`; sensible defaults apply
(`batch_all_triplet` loss, `group_by_label` sampler, lr 2e-5, batch 32) if you
leave the optional knobs unset. No VRAM card is needed for this sub-skill.

## Hold-out for evaluation

Classifier quality is measured by the **classification eval**, which needs a
**held-out** labeled set. Confirm data usage as usual: either generate a separate
small eval SDG run (same categories), or use `split_dataset` to hold out a portion
of this dataset. Keep the held-out dataset's run ID — the eval uses it.

## After training — evaluate

Once training succeeds, offer a **classification eval**: `create_eval_job` with
`eval_mode="classification"`, `training_job_id` = this job, the held-out dataset as
`parent_job_id`/`eval_data_run_id`, and `label_column` matching the data. It
reports accuracy / macro-F1 / confusion (no rubric or judge). eval_mode defaults to
classification automatically for embedding models, so a generative/judge eval is
never offered here.
