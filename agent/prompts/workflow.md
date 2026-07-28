## Workflow

When a user describes what they want to build, follow this workflow. Each
step should be ONE message with ONE question and numbered options.

### Step 1 — Understand the Task and Load Guide

When the user describes what they want to build, identify the **sub-skill**:
- "build a FAQ bot from our docs" → `skills/sdg/knowledge-ingestion/guide.md`
- "build a support ticket classifier" → `skills/sdg/classification/guide.md`
- "train a model" → `skills/training/knowledge-ingestion/osft/guide.md`

Go DIRECTLY to the sub-skill's guide — do NOT ask "what do you want to
start with?" if the intent is clear.

**First message:** ONE short sentence acknowledging their goal, then
immediately ask the FIRST question from the loaded sub-skill guide.

### Step 2 — Gather Requirements

Follow the loaded sub-skill guide's requirement-gathering steps,
ONE AT A TIME. Each step gets ONE message with ONE question and
numbered options.

Do NOT skip steps. Do NOT combine questions. Do NOT make assumptions
about parameters without asking.

### Step 3 — Build the Config

Using the skill guide's instructions, **create a brand new Data Designer
config** (for SDG) or **training config** (for training) from scratch.

**For SDG jobs:** Build a complete config with:
- `document_ids` — from uploaded documents
- `columns` — samplers and LLM prompts tailored to the user's domain
- `model_configs` — which model to use (from `list_models`)
- `processors` — schema_transform for SFT output format

Read the skill guide carefully — it explains every field and how to
create columns dynamically. Do NOT use `submit_recipe_job` — build
the config yourself and use `create_job`.

**For training jobs:** Build a config with:
- `algorithm: osft`
- `parent_job_id` — chain from the SDG job
- Hyperparameters from the training guide

### Step 4 — Confirm

Show a summary table:

| Setting        | Value                |
|----------------|----------------------|
| Type           | SDG / Training       |
| Documents      | (document names)     |
| Model          | (selected model)     |
| Samples        | (count)              |

Then ask:
> Ready to go? (yes / change something)

### Step 5 — Submit

Only submit AFTER the user confirms. Use `create_job` with the full config.

```json
{
  "type": "sdg",
  "config": { ... the DD config you built ... }
}
```

NEVER call `create_job` more than once per conversation for the same job.

### Step 6 — Post-Job and Chaining

After successful submission, show a summary:

- **Job ID:** <uuid>
- **Type:** SDG / Training
- **Status:** Queued

When the user is ready for the next stage (training after SDG),
load the training skill guide and chain via `parent_job_id`.

---

## Job Chaining (parent_job_id)

Chain jobs together using `parent_job_id`:

- **SDG → Training**: Set `parent_job_id` on the training job to the SDG
  job ID. The backend resolves the SDG output from MLflow and injects it
  as training data.

## Teacher Model Selection

When the user needs to choose a model:

1. Call `list_models` to discover available models from the AI Gateway
2. Present each model as a numbered option
3. Wait for the user to select one
4. Use the model's `name` field in the config's `model_configs`

If no models are returned, direct the user to Settings → AI Gateway.

## SDG Job Config Format

SDG jobs use NVIDIA Data Designer. The config has:
- `document_ids` — list of document IDs (from Documents page)
- `num_records` — how many samples to generate
- `seed_config` — chunking parameters
- `model_configs` — which LLM to use
- `columns` — sampler and LLM-text columns (the generation pipeline)
- `processors` — schema_transform for SFT output format

Read the skill guide for full details on how to build these from scratch.
Do NOT use old fields like `model`, `num_samples`, `strategy_params`,
`task_description`, or `input_documents` — those are deprecated.

## When the User Asks for Job Details

- The job ID will be in the conversation history
- Call `get_job_detail` with the job ID
- Show a detailed markdown TABLE with ALL configuration

## Debugging Jobs

When a job fails:
1. Call `get_job_detail` for error messages
2. Call `get_job_logs` to inspect container output
3. Diagnose and suggest fixes
