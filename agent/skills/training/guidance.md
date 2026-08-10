# Training Skill Guidance

Pick the sub-skill that best matches the user's task. Read its `guide.md`
for the requirement-gathering workflow and reference payload.

## Available Sub-Skills

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| knowledge-ingestion/osft | `skills/training/knowledge-ingestion/osft/` | Knowledge ingestion, FAQ bots, doc-grounded QA, classification |
| task-distillation/osft | `skills/training/task-distillation/osft/` | RFE assessors, code reviewers, rubric-based scoring |

## How to Choose

- **Knowledge ingestion or classification** -> OSFT (default, recommended)
- **Task distillation (scoring/assessment)** -> OSFT with task-distillation hyperparameters

## Calling `create_training_job`

The tool has full Pydantic validation. Each sub-skill provides a
reference payload -- use it as the base and adapt to the user's model
choice, dataset size, and compute.

### Job Chaining

Set `parent_job_id` to the SDG job ID. The worker automatically resolves
the SDG job's MLflow artifact URI for `data_path`. No manual data path
configuration needed.

### Data Format

Training expects JSONL with a `messages` column -- this is exactly what
the SDG `schema_transform` processor produces.
