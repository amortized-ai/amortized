---
name: training-classifier
description: >-
  Train a text classifier / decision model from (text, category) data. Default is a
  classification head (classifier_sft) — best accuracy + calibrated probabilities for a
  fixed label set (intent routing, triage, topic/sentiment). Offers an embedding router
  (embedding_sft) when the user wants embeddings / semantic routing / retrieval. No
  generative-LLM classifier. Pairs with the sdg-classification skill.
metadata:
  stage: training
---

# Classifier — Training Guide

Use this skill to train a **text classifier / decision model** from labeled
`(text, category)` data (from the **Classification** SDG skill). Two paths, same data:

- **Classifier head** (`algorithm="classifier"` → `classifier_sft`) — **the default.** A
  dedicated classification head (on a modern encoder like ModernBERT, or a decoder via
  LoRA). Best accuracy and **calibrated probabilities** for a *fixed, known* label set —
  e.g. ModernBERT scores ~0.94 on Banking77 (77 intents). Use this for the common case:
  "classify text into my N categories."
- **Embedding router** (`algorithm="embedding"` → `embedding_sft`) — contrastive
  sentence-transformers; classifies by nearest class and produces reusable embedding
  vectors. Pick it when the task *shape* calls for it (see "Path" below), not the default.

There is no generative-LLM classifier path — a dedicated head is faster, cheaper, and
more accurate.

This flow differs from the LLM sub-skills (OSFT/SFT/LoRA): there is **no VRAM card /
model-size comparison** — the VRAM estimator is for generative LLM training only.

## Requirement gathering

Follow the workflow's one-question-per-turn rules. Gather:

1. **Training data** — a completed classification SDG job via `parent_job_id` (or a dataset
   `data_run_id`). If the orchestrator passed an SDG job ID, use it without asking. Set one
   of the two, never both: to train on a split or an uploaded dataset, set only
   `data_run_id`. Set `text_column` (`text`) and `label_column` (`category` for the SDG skill). String
   categories are label-encoded to integers automatically.
2. **Path (head vs router)** — decide by the task *shape*, not by whether the user used
   ML jargon (a domain user rarely says "embeddings" or "open-set" for a task that
   actually needs the router). **Default to the head classifier.** Choose the **embedding
   router** when either is clearly true:
   - **Open/growing label set** — the categories will change or new ones get added over
     time without retraining (a head is tied to the labels it was trained on).
   - **Reusable embeddings / nearest-match** — the user needs the vectors themselves, or
     wants to match text to the closest existing item (retrieval, semantic search,
     dedup/similarity), not just a predicted label.

   If the request shows routing/matching intent but it's **ambiguous** whether the label
   set is fixed (e.g. "route tickets to the right team", "match questions to our FAQs"),
   do NOT silently guess — present a one-time choice with `present_options`:
   - "Fixed set of categories — classifier (recommended): best accuracy, calibrated
     confidence"
   - "Categories will grow, or I need embeddings for matching/retrieval — embedding router"

   A plainly fixed-label task ("classify into these N categories") needs no question —
   just use the head. Set `algorithm` from the resolved choice.
3. **Base model** — head default `answerdotai/ModernBERT-base` (fast, strong, 8k context);
   embedding default `sentence-transformers/all-MiniLM-L6-v2`. Only surface alternatives if
   asked (any HF encoder/decoder with a classification head works for the head path; custom
   architectures that ship an HF `auto_map` work via `trust_remote_code` + a pinned
   `revision`; a decoder backbone needs `use_peft=True`, which requires a GPU).
4. **Epochs** — head default ~10 for a few-thousand-example set (more for small, fewer for
   large); embedding default ~25. Don't ask about lr/batch/loss unless the user brings it up.
5. **Device (CPU vs GPU)** — defaults to CPU (`train_on_gpu=False`) for both paths. Not a
   separate up-front question; surface it when you present the config — see "Config & device".

## Config & device

`validate_training_job` exposes every field as a first-class parameter. Set
`algorithm="classifier"` (head) or `algorithm="embedding"` (router) plus
`model_name_or_path`, `text_column`, `label_column`, `num_train_epochs`,
`parent_job_id`/`data_run_id`. Head-only optional knobs: `task` (default
`sequence_classification`; `multiple_choice` for pick-one-of-K with an `options` column, or
`token_classification` with a `tokens` column), `use_peft` (decoder backbones), and
`trust_remote_code`/`revision` (custom architectures). No VRAM card is needed.

**The device note goes in the message that carries the confirm card — as plain text, not a
second card.** The confirmable config card is the one interactive element of that turn; do
NOT also raise a separate `present_options` device card. Add a one-line device note so the
user can confirm (CPU) or reply to switch to GPU.

To write that note, estimate the CPU runtime from the training-set size: read `num_samples`
from the SDG run / dataset (`num_samples` tag or `num_samples_generated` metric) and compute
`num_samples × epochs ÷ 25 ≈ seconds`:

- **If the estimate exceeds ~a few minutes** (or the dataset is clearly large and the count
  is unavailable): state the approximate CPU time and tell the user they can **reply to
  switch to `train_on_gpu=True`** (~100x faster; reserves one GPU) before confirming — e.g.
  *"This trains on CPU (~20 min). Reply if you'd rather run it on a GPU; otherwise Confirm."*
- **If it's quick/small:** just note it will train on CPU and let them confirm.

Set `train_on_gpu=True` (and re-present the card) only after the user opts in; otherwise
leave it unset. A decoder backbone with `use_peft=True` always needs a GPU regardless. Don't
guess a sample count or ask for one — if it's unknown, present the CPU default and mention
the GPU opt-in is available.

## Hold-out for evaluation

Classifier quality is measured by the **classification eval**, which needs a **held-out**
labeled set. Either generate a separate small eval SDG run (same categories), or use
`split_dataset` to hold out a portion. Keep the held-out dataset's run ID — the eval uses it.

## After training — evaluate

Once training succeeds, offer to **evaluate** the model. This is a **classification
eval** (not a rubric/judge eval) — the eval flow auto-detects the model type and scores
predicted labels directly: for a head classifier, **held-out accuracy / macro-F1 /
per-class F1 / confusion + calibrated ECE** (argmax); for an embedding router, a
nearest-centroid separability proxy. It needs a **held-out labeled `text,category`
set** with the same categories (a separate classification SDG run, or a split of the
training data) — NOT a `messages` dataset, and there is no rubric or judge to design.
