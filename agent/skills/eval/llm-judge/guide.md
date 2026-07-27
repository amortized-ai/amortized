# LLM Judge Evaluation Guide

Use an LLM as a judge to evaluate model outputs against ground truth.

## Requirement Gathering

Ask the user these questions **one at a time, in separate messages**.
Do NOT skip ahead or combine questions. Wait for the user's answer to
each question before asking the next one.

1. **Which evaluation method?** — What should the judge evaluate?
   1) Classification accuracy — Compare predicted labels against ground truth
   2) Response quality — Rate generated text for relevance, coherence, completeness
   3) Both — Run accuracy and quality evaluations

2. **Which judge model?** — Call `list_models` to discover available
   models from the AI Gateway. Present each as a numbered option.
   ALWAYS add as the last option:
   N) Configure a model — Set up an AI Gateway endpoint in Settings

By default, evaluate **all samples** in the test set. Do not ask the
user how many samples to evaluate — just use the full dataset.

3. **Confirm plan** — After the user picks a judge model, show a
   confirmation table and call `estimate_eval_cost` to get a cost
   estimate. Show:

   | Setting        | Value                |
   |----------------|----------------------|
   | Method         | (selected method)    |
   | Judge Model    | (selected model)     |
   | Samples        | All                  |
   | Est. Cost      | $X.XX                |
   | Savings        | XX% vs manual review |

   Then ask:
   > Ready to run the evaluation? (yes / change something)

   Do NOT submit the job until the user confirms. This step is MANDATORY.

## Job Submission

Use the eval recipe matching the task type.

**CRITICAL:** You MUST pass `parent_job_id` set to the training job ID
from this conversation. Without it, the eval job has no model or dataset
to evaluate and will fail. The training job ID is in the conversation
history from when you submitted the training job.
