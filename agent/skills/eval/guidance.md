# Eval Skill Guidance

Pick the evaluation approach that best matches what the user wants to
measure. Read its `guide.md` for deep expertise before configuring the
eval job.

## Available Sub-Skills

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| llm-judge | `skills/eval/llm-judge/` | General-purpose evaluation using an LLM as judge. Works for most task types |

## How to Choose

- **Most tasks** → `llm-judge` (versatile, works across task types)
- **Specific metric needed** → check `get_recipes` for specialized eval
  templates (exact-match, regex, classification-accuracy, etc.)

## After Loading the Sub-Skill

The sub-skill's `guide.md` will tell you:
- How to configure the judge model and evaluation criteria
- What metrics to measure and how to interpret results
- Config template to use as a starting point

Always call `estimate_eval_cost` with the sample count and judge model
before presenting the confirmation table.
