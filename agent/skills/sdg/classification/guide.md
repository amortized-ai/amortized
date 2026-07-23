# Classification — SDG Guide

Use this guide for ticket classifiers, intent routers, sentiment analysis,
and content moderation tasks.

## Requirement Gathering

Ask the user these questions (one at a time, with numbered options):

1. **What domain?** — What kind of content will this classifier handle?
   1) Customer support tickets — Route tickets by topic and urgency
   2) User messages/intents — Classify user intents for chatbots or routing
   3) Content moderation — Flag content by category (spam, toxic, etc.)
   4) Something else — Describe your classification task

2. **What categories?** — Based on the domain, suggest specific labels.
   For customer support, suggest:
   1) Standard categories — Billing, Technical, Account, General Inquiry
   2) Detailed categories — Billing, Technical, Account, Shipping, Returns, Product Questions
   3) Custom categories — I'll define my own labels
   For other domains, suggest 3-4 relevant groupings.

3. **Urgency levels?** — Should the classifier also assign urgency?
   1) Yes, 3 levels — Low, Medium, High
   2) Yes, 4 levels — Low, Medium, High, Critical
   3) No urgency — Just classify by category

4. **How many samples?** — How many training examples to generate?
   1) 100 samples — Quick prototype
   2) 500 samples — Good coverage across categories
   3) 1000 samples — Best model quality, more diverse examples

5. **Which teacher model?** — Call `list_models` to discover available
   models from the AI Gateway. Present each as a numbered option.
   ALWAYS add as the last option:
   N) Configure a model — Set up an AI Gateway endpoint in Settings

## Recipe Selection

- Customer support tickets → `examples/ticket-classifier/synth`
- Intent routing → `examples/intent-router/synth`
- Content moderation → `examples/content-moderator/synth`
- Other → use `get_recipes` to find a match, or fall back to
  `templates/sdg/classification.yaml`

## Job Submission

Use `submit_recipe_job` with these overrides:
- `num_samples`: user's chosen count
- `model`: the `name` field from the selected gateway endpoint
  (e.g. `openai/gpt-4o-mini`)
- `task_description`: a detailed description including ALL categories
  and urgency levels the user selected

## Recommended Training Method

After SDG completes, recommend **LoRA SFT** for classification tasks.
It's fast, memory-efficient, and well-suited for label prediction.
