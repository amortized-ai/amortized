---
name: sdg-classification
description: >-
  Generate synthetic labeled training data for text classifiers. Use for
  ticket classifiers, intent routers, sentiment analysis, and content
  moderation, where each sample is input text plus a category (and optional
  urgency) label. Covers category/urgency design, sample counts, and SFT
  formatting.
metadata:
  stage: sdg
---

# Classification (Generative / LLM) — SDG Guide

> **Most classifier / router / sentiment / intent tasks should NOT use this
> guide — use the `sdg-embedding-classifier` skill instead** (the common case: a
> compact, fast embedding classifier trained with `embedding_sft` on flat
> `text,category` data). Use THIS guide *only* when the user explicitly wants a
> **generative LLM** classifier that emits the label as text (`messages`/SFT
> data). If you opened this by default, switch to `sdg-embedding-classifier` and
> just continue — do not tell the user you switched.

## How This Works

You will gather requirements and call `validate_sdg_job` with the
appropriate parameters. The classification pipeline generates labeled
examples where each sample has input text and a classification label.

Pull the config structure (columns, `model_configs`, processors) from the
`validate_sdg_job` schema — don't restate it here or copy another task's
template. This is the generative/`messages` variant, so the output goes through a
`schema_transform` processor into chat `messages` (unlike the embedding-classifier
skill, which emits flat `text,category` with `processors: []`).

## Requirement Gathering

Ask the user these questions (one at a time, with numbered options).
Work these steps in order, ONE question per turn. Each step's detailed
instructions arrive as you reach it — the titles below are the outline so
you always know where you are and what's next; do not run ahead of the
step whose detail is in front of you.

1. **Domain**
2. **Categories** (ask AFTER user picks domain)
3. **Urgency levels**
4. **Sample count**
5. **Teacher model**

## Tool Parameters

Call `validate_sdg_job` with these parameters. Customize columns,
prompts, and categories based on the user's specific task. Use the
model name from `list_models` in `model_configs`.

Key parameters for classification:
- `columns` — a category sampler, an LLM column to generate text, an LLM column to generate labels
- `model_configs` — `[{"alias": "text", "model": "<from-list_models>", "provider": "<from-list_models>", "skip_health_check": true}]`
- `processors` — schema_transform to produce SFT `messages` format
- `num_records` — based on category count (see sample count step)

## After SDG — Training

This generative classifier is a normal `messages`-based fine-tune. Recommend
OSFT/SFT training — load the `training-knowledge-ingestion-osft` skill (its
**Scope** note covers `messages`-based tasks; skip the document-specific steps,
since there are no documents here). Chain via `parent_job_id`.
