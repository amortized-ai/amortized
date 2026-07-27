# Classification — SDG Guide

Use this guide for ticket classifiers, intent routers, sentiment analysis,
and content moderation tasks.

## Requirement Gathering

CRITICAL: Ask these questions **one at a time, in separate messages**.
Each question MUST be its own message. Do NOT combine two questions into
one message. Do NOT ask about categories in the same message as domain.
Wait for the user to respond before asking the next question.

### Step 1 — Domain (ALWAYS ask this first, even if obvious)

Your first response after the user describes their task must ONLY ask
what domain/industry their classifier is for. Do NOT mention categories,
urgency, or samples yet.

Say ONE short sentence acknowledging their goal, then ask:

"What type of support tickets will this handle?"

1) Software/technical support — Bug reports, feature requests, troubleshooting
2) Billing & payments — Invoices, refunds, subscription issues
3) Customer service — Account access, onboarding, general inquiries
4) E-commerce — Orders, shipping, returns, product questions

STOP here. Do NOT continue to step 2 in this message.

### Step 2 — Categories (ask AFTER user picks domain)

Based on the domain they chose, suggest specific category labels:

"What categories should it classify into?"

For customer support, suggest:
1) Standard categories — Billing, Technical, Account, General Inquiry
2) Detailed categories — Billing, Technical, Account, Shipping, Returns, Product Questions
3) Custom categories — I'll define my own labels

For other domains, suggest 3-4 relevant groupings.

STOP here. Do NOT continue to step 3 in this message.

### Step 3 — Urgency levels

"Should the classifier also assign an urgency level?"

1) Yes, 3 levels — Low, Medium, High
2) Yes, 4 levels — Low, Medium, High, Critical
3) No urgency — Just classify by category

### Step 4 — Sample count

"How many training examples should we generate?"

1) 100 samples — Quick prototype
2) 500 samples — Good coverage across categories
3) 1000 samples — Best model quality, more diverse examples

### Step 5 — Teacher model

Call `list_models` to discover available models from the AI Gateway.
Present each as a numbered option. ALWAYS add as the last option:
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
