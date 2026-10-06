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
4. **Which teacher model?** — Call `list_models` to get the available
   teacher models (each has a `name` and a `provider`). Present ONLY those
   models as options. Do NOT suggest models that aren't returned by
   `list_models` — they won't work. If no models are returned, stop — no
   teacher-model provider is configured on the server.
5. **How many samples?** — Scale based on category count and desired
   coverage. Recommend at least 50 samples per category for basic
   coverage and 150+ per category for production quality.
   1) N×50 samples — Basic coverage across all categories
   2) N×100 samples — Good diversity, recommended
   3) N×150 samples — Best quality, most diverse examples
   (where N = number of categories × urgency levels)
6. **Distribution** — Should categories be balanced or weighted?
   Default: roughly balanced unless the real-world distribution is known.

## Tool Parameters

Call `validate_sdg_job` with these parameters. Customize columns,
prompts, and categories based on the user's specific task. Use the
model name from `list_models` in `model_configs`.

Key parameters for classification:
- `columns` — a category sampler, an LLM column to generate text, an LLM column to generate labels
- `model_configs` — `[{"alias": "text", "model": "<from list_models>", "provider": "<from list_models>", "skip_health_check": true}]`
- `processors` — schema_transform to produce SFT `messages` format
- `num_records` — based on category count (see sample count step)

Call `present_options` with step="sdg-domain" and these options:
- title: "Software/technical support", description: "Bug reports, feature requests, troubleshooting", value: "Software/technical support — Bug reports, feature requests, troubleshooting"
- title: "Billing & payments", description: "Invoices, refunds, subscription issues", value: "Billing & payments — Invoices, refunds, subscription issues"
- title: "Customer service", description: "Account access, onboarding, general inquiries", value: "Customer service — Account access, onboarding, general inquiries"
- title: "E-commerce", description: "Orders, shipping, returns, product questions", value: "E-commerce — Orders, shipping, returns, product questions"

STOP here. Do NOT continue to step 2 in this message.

### Step 2 — Categories (ask AFTER user picks domain)

Based on the domain they chose, suggest specific category labels:

"What categories should it classify into?"

For customer support, call `present_options` with step="sdg-categories" and these options:
- title: "Standard categories", description: "Billing, Technical, Account, General Inquiry", value: "Standard categories — Billing, Technical, Account, General Inquiry"
- title: "Detailed categories", description: "Billing, Technical, Account, Shipping, Returns, Product Questions", value: "Detailed categories — Billing, Technical, Account, Shipping, Returns, Product Questions"
- title: "Custom categories", description: "I'll define my own labels", value: "Custom categories — I'll define my own labels"

For other domains, suggest 3-4 relevant groupings.

STOP here. Do NOT continue to step 3 in this message.

### Step 3 — Urgency levels

"Should the classifier also assign an urgency level?"

Call `present_options` with step="sdg-urgency" and these options:
- title: "Yes, 3 levels", description: "Low, Medium, High", value: "Yes, 3 levels — Low, Medium, High"
- title: "Yes, 4 levels", description: "Low, Medium, High, Critical", value: "Yes, 4 levels — Low, Medium, High, Critical"
- title: "No urgency", description: "Just classify by category", value: "No urgency — Just classify by category"

### Step 4 — Sample count

"How many training examples should we generate?"

Call `present_options` with step="sdg-samples" and these options:
- title: "100 samples", description: "Quick prototype", value: "100 samples — Quick prototype"
- title: "500 samples", description: "Good coverage across categories", value: "500 samples — Good coverage across categories"
- title: "1000 samples", description: "Best model quality, more diverse examples", value: "1000 samples — Best model quality, more diverse examples"

### Step 5 — Teacher model

Call `list_models` to discover available teacher models (name + provider).
Call `present_options` with step="sdg-teacher-model" and each model as an option.
ALWAYS add as the last option:
- title: "Configure a model", description: "Configure a teacher-model provider key on the server", value: "Configure a model — Configure a teacher-model provider key on the server"

## After SDG — Training

This generative classifier is a normal `messages`-based fine-tune. Recommend
OSFT/SFT training — load the `training-knowledge-ingestion-osft` skill (its
**Scope** note covers `messages`-based tasks; skip the document-specific steps,
since there are no documents here). Chain via `parent_job_id`.
