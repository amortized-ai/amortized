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
- It **trains on CPU by default** (the job requests 0 GPUs, schedules on any node,
  and doesn't touch the GPU quota) — so skip the VRAM cards. A GPU is **optional**:
  set `train_on_gpu=True` to reserve one GPU, which is ~100x faster. CPU is fine for
  small/quick runs; a GPU matters for many epochs or large datasets (rough rule:
  CPU trains ~25 samples/s, so samples × epochs ÷ 25 ≈ seconds on CPU — if that's
  more than a few minutes, offer the GPU opt-in). You must surface this CPU-vs-GPU
  choice when presenting the config — see "Config & device" below.

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
4. **Device (CPU vs GPU)** — defaults to CPU (`train_on_gpu=False`). This is not a
   separate up-front question; you surface it **when you present the config for
   confirmation** — see "Config & device" below. (Even when the user says "use
   defaults," you still state the device choice there.)

## Config & device

`validate_training_job` already exposes every field you need as a first-class
parameter — `algorithm`, `model_name_or_path`, `text_column`, `label_column`,
`num_train_epochs`, `parent_job_id`/`data_run_id`, and the optional
`loss_type` / `batch_sampler` / `warmup_ratio` / `seed`. Pull the structure from
that schema and set `algorithm="embedding_sft"`; sensible defaults apply
(`batch_all_triplet` loss, `group_by_label` sampler, lr 2e-5, batch 32) if you
leave the optional knobs unset. No VRAM card is needed for this sub-skill — the
VRAM/model-size estimator is for generative LLM training (sft/lora/qlora/osft) only.

**Always surface the CPU-vs-GPU choice when you present the config for confirmation —
never go to the confirm card silently on device.** Estimate the CPU runtime from the
training-set size: read `num_samples` from the SDG run / dataset (`num_samples` tag or
`num_samples_generated` metric) and compute `num_samples × epochs ÷ 25 ≈ seconds`
(CPU trains ~25 samples/s). Then, alongside the config card:

- **If the estimate exceeds ~a few minutes** (or the dataset is clearly large and the
  count is unavailable): state the approximate CPU time and **explicitly offer
  `train_on_gpu=True`** (~100x faster; reserves one GPU from the quota) as the faster
  alternative before the user confirms.
- **If it's quick/small:** just note it will train on CPU (no GPU needed).

Only set `train_on_gpu=True` after the user opts in; otherwise leave it unset (CPU
default). Don't guess a sample count or ask the user for one — if it's genuinely
unknown, present the CPU default and mention the GPU opt-in is available.

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
