# LLM Judge Evaluation Guide

Use an LLM as a judge to evaluate model outputs against ground truth.

## Requirement Gathering

Ask the user these questions (one at a time, with numbered options):

1. **Which judge model?** — Call `list_models` to discover available
   models from the AI Gateway. Present each as a numbered option.
   ALWAYS add as the last option:
   N) Configure a model — Set up an AI Gateway endpoint in Settings

2. **How many samples to evaluate?** — How many test examples?
   1) 50 samples — Quick check
   2) 100 samples — Standard evaluation
   3) All samples — Evaluate the entire test set

## Job Submission

Use the eval recipe matching the task type. Always set `parent_job_id`
to the training job ID.
