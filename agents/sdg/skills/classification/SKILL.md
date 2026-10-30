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
- `model_configs` — `[{"alias": "text", "model": "<from-list_models>", "provider": "<from-list_models>", "skip_health_check": true}]`
- `processors` — schema_transform to produce SFT `messages` format
- `num_records` — based on category count (see sample count step)

### Step 1 — Domain <!--signal:domain-->

"What kind of content will this classifier handle?" e.g., customer support
tickets, user messages, content moderation.

Call `present_options` with step="domain" and these options:
- title: "Software/technical support", description: "Bug reports, feature requests, troubleshooting", value: "Software/technical support — Bug reports, feature requests, troubleshooting"
- title: "Billing & payments", description: "Invoices, refunds, subscription issues", value: "Billing & payments — Invoices, refunds, subscription issues"
- title: "Customer service", description: "Account access, onboarding, general inquiries", value: "Customer service — Account access, onboarding, general inquiries"
- title: "E-commerce", description: "Orders, shipping, returns, product questions", value: "E-commerce — Orders, shipping, returns, product questions"

STOP here. Do NOT continue to step 2 in this message.

### Step 2 — Categories (ask AFTER user picks domain) <!--signal:categories-->

Based on the domain they chose, suggest specific category labels:

"What categories should it classify into?"

For customer support, call `present_options` with step="categories" and these options:
- title: "Standard categories", description: "Billing, Technical, Account, General Inquiry", value: "Standard categories — Billing, Technical, Account, General Inquiry"
- title: "Detailed categories", description: "Billing, Technical, Account, Shipping, Returns, Product Questions", value: "Detailed categories — Billing, Technical, Account, Shipping, Returns, Product Questions"
- title: "Custom categories", description: "I'll define my own labels", value: "Custom categories — I'll define my own labels"

For other domains, suggest 3-4 relevant groupings.

STOP here. Do NOT continue to step 3 in this message.

### Step 3 — Urgency levels <!--signal:urgency-->

"Should the classifier also assign an urgency level?"

Call `present_options` with step="urgency" and these options:
- title: "Yes, 3 levels", description: "Low, Medium, High", value: "Yes, 3 levels — Low, Medium, High"
- title: "Yes, 4 levels", description: "Low, Medium, High, Critical", value: "Yes, 4 levels — Low, Medium, High, Critical"
- title: "No urgency", description: "Just classify by category", value: "No urgency — Just classify by category"

### Step 4 — Sample count <!--signal:samples-->

"How many training examples should we generate?"

Call `present_options` with step="samples" and these options:
- title: "100 samples", description: "Quick prototype", value: "100 samples — Quick prototype"
- title: "500 samples", description: "Good coverage across categories", value: "500 samples — Good coverage across categories"
- title: "1000 samples", description: "Best model quality, more diverse examples", value: "1000 samples — Best model quality, more diverse examples"

### Step 5 — Teacher model <!--signal:teacher-model-->

Call `list_models` to discover available teacher models (name + provider).
Call `present_options` with step="teacher-model" and each model as an option.
ALWAYS add as the last option:
- title: "Configure a model", description: "Configure a teacher-model provider key on the server", value: "Configure a model — Configure a teacher-model provider key on the server"

## Prompt review

**Each prompt gets its OWN review step.** This generative classifier composes TWO
prompts — the text-generation system prompt (Step 6) and the classifier system
prompt (Step 7, the SFT processor system message the trained model runs under).
Review them in SEPARATE turns, one per step: do NOT compose or `show_prompt` both
in the same turn. Do NOT ask the user to write them — generate domain-specific
defaults from the chosen domain and categories; the user reviews and adjusts each
via its `show_prompt` review card. Every prompt field in the config MUST be a
`<prompt:...>` ref returned by `show_prompt` — never pasted inline. (If your
pipeline uses more than one `llm-text` generator column, give EACH its own review
step — one prompt per turn.)

### Step 6 — Text-generation system prompt <!--signal:generation-prompt all:show_prompt,present_options-->

Compose the `llm-text` generation column's `system_prompt` — the role that writes
a realistic sample of text for the sampled category. Ground it in the domain and
categories gathered above; instruct it NOT to name the category in the generated
text. Render it with `show_prompt` (pass the complete prompt text), then ask the
user to approve or adjust it with `present_options`. Handle ONLY this one prompt
this turn — do not compose or `show_prompt` the classifier prompt (Step 7) yet.
Stay on this step through the render → approve loop (on any edit, re-render with
`show_prompt` AGAIN — each returns a NEW `prompt_ref`, use the latest) until the
user approves it as shown. Put the returned `<prompt:...>` ref into the generation
column's `system_prompt` — never paste the prompt text inline.

### Step 7 — Classifier system prompt <!--signal:classifier-prompt all:show_prompt,present_options-->

Compose the classifier `system_prompt` — the SFT processor system message that
defines how the TRAINED classifier behaves at inference time (classify the input
into exactly one of the categories; list the allowed labels). Render it with
`show_prompt` (pass the complete prompt text), then ask the user to approve or
adjust it with `present_options`. Handle ONLY this one prompt this turn — do not
re-show the Step 6 generation prompt, and do not bundle both prompts into one
turn. Stay on this step through the render → approve loop (on any edit, re-render
with `show_prompt` AGAIN — each returns a NEW `prompt_ref`, use the latest) until
the user approves it as shown. Put the returned `<prompt:...>` ref into the
schema_transform processor's system message — never paste the prompt text inline.

## After SDG — Training

This generative classifier is a normal `messages`-based fine-tune. Recommend
OSFT/SFT training — load the `training-knowledge-ingestion-osft` skill (its
**Scope** note covers `messages`-based tasks; skip the document-specific steps,
since there are no documents here). Chain via `parent_job_id`.
