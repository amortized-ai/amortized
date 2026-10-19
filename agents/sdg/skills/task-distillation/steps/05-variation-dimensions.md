### Step 5 — Variation dimensions

Infer axes of diversity from the user's examples:
- **Domain/topic** — ask the user what content domains or categories
  the task covers (e.g., product areas, ticket types, document topics)
- **Quality/difficulty profile** — target levels with weights
- **Structure** — format variations (formal, terse, detailed)

Each becomes a sampler column. Only ask if ambiguous.

When you ask the user about this step, call present_options with step: "variation-dimensions".
