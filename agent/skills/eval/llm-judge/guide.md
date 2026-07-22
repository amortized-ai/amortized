# LLM Judge — Eval Guide

*This guide is a placeholder. It will be populated with expert knowledge
for LLM-as-judge evaluation.*

In the meantime, use the recipe template at `templates/eval/llm-judge`
and call `list_models` to select a judge model.

Key considerations:
- Use a stronger model than the student as the judge
- Include specific evaluation criteria in the judge prompt
- Common criteria: accuracy, completeness, groundedness, format compliance
