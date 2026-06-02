---
description: List available SDG (synthetic data generation) flows
invoke-on-demand: true
---

List all available SDG flows that can be used for synthetic data generation.

## API call

```bash
curl http://localhost:8000/api/v1/flows
```

Returns a list of flows with `id`, `name`, `description`, and `category`.

## Flow categories
- **knowledge_infusion** — Q&A generation, summaries, knowledge extraction
- **evaluation** — RAG evaluation, answer quality assessment
- **agentic** — MCP distillation, agent behavior datasets
- **red_team** — Adversarial prompt generation
- **text_analysis** — Classification, sentiment, text transformation
- **code_evaluation** — Code quality, bug detection datasets

Use the flow `id` when submitting an SDG job via the submit-sdg-job skill.
