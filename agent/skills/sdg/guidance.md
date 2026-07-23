# SDG Skill Guidance

Pick the sub-skill that best matches the user's task. Read its `guide.md`
for deep expertise before configuring the SDG job.

## Available Sub-Skills

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| knowledge-ingestion | `skills/sdg/knowledge-ingestion/` | FAQ bots, QA assistants, doc-grounded chat, RAG models, knowledge base assistants |
| classification | `skills/sdg/classification/` | Ticket classifiers, intent routers, sentiment analysis, content moderation |
| extraction | `skills/sdg/extraction/` | Entity extraction, structured field extraction from unstructured text |
| summarization | `skills/sdg/summarization/` | Document summarization, conversation summarization, abstractive/extractive summaries |

## How to Choose

- **User has documents/docs they want a model to answer questions about** → `knowledge-ingestion`
- **User wants to sort/label/categorize text** → `classification`
- **User wants to pull specific fields out of text** → `extraction`
- **User wants to condense or summarize text** → `summarization`

If none match, use `get_recipes` to browse the full recipe catalog and
proceed with the general workflow.

## After Loading the Sub-Skill

The sub-skill's `guide.md` will tell you:
- What questions to ask the user (domain-specific requirement gathering)
- What recipe or config template to use as a starting point
- What defaults to apply and what to ask about
- Key tradeoffs and decisions the user should make
- Pitfalls to avoid

Follow the guide's recommendations, but remember: the user has the final
call on every knob. Always confirm before submitting.
