# OSFT Training Workflow

OSFT (Optimized Supervised Fine-Tuning) outperforms standard SFT by
30+ percentage points in open-book settings. Recommended for all
knowledge ingestion and classification tasks.

## Workflow

### Step 1 -- Model

Default: `Qwen/Qwen3.5-4B`. Present ONLY these exact HuggingFace
model IDs (do NOT invent other sizes):
- `Qwen/Qwen3.5-0.8B` -- fastest inference, good for prototyping
- `Qwen/Qwen3.5-2B` -- small but capable
- `Qwen/Qwen3.5-4B` -- balanced, recommended default
- `Qwen/Qwen3.5-9B` -- best accuracy for knowledge tasks

Do NOT suggest models that don't exist (e.g. Qwen3.5-0.6B,
Qwen3.5-1.5B, Qwen3.5-8B -- these are not real).

Model choice affects all hyperparameters -- set them after.

### Step 2 -- Training data

Should come from a completed SDG job. Use `parent_job_id` to chain
SDG -> Training. Ask the user for the SDG job ID if not in the
conversation.

### Step 3 -- Confirm config

Present a confirmation table with all parameters. Show WHY you chose
each value: "5 epochs because dataset is small (200 samples)" or
"learning_rate 2e-5 because Qwen3.5-9B is a large model".

GPUs are always 1 (`nproc_per_node: 1`). Do NOT ask about GPU count.

## Reference Payload

Use this as the base for `create_training_job()`. Adapt parameters
based on the user's model choice and dataset size.

```json
{
  "algorithm": "osft",
  "model_name_or_path": "Qwen/Qwen3.5-4B",
  "num_train_epochs": 5,
  "learning_rate": 2e-05,
  "effective_batch_size": 32,
  "max_length": 11000,
  "unfreeze_rank_ratio": 0.2,
  "warmup_steps": 25,
  "save_samples": 0,
  "accelerate_full_state_at_epoch": false,
  "checkpoint_at_epoch": true,
  "nproc_per_node": 1,
  "data_output_dir": "data-output",
  "bf16": true,
  "max_tokens_per_gpu": 15000,
  "parent_job_id": "<SDG_JOB_ID>"
}
```

### Adapting the Payload

| Param | How to adapt |
|-------|-------------|
| `model_name_or_path` | User's chosen model |
| `num_train_epochs` | 3-5 for <1000 samples, 2-3 for 1000-5000, 1-2 for 5000+ |
| `learning_rate` | 2e-5 for 9B, 5e-5 for 4B, 1e-4 for 0.8B-2B |
| `effective_batch_size` | Always 32. Always show in confirmation table |
| `max_length` | ~11000 for knowledge QA with 15-sentence chunks, ~2048 for classification |
| `unfreeze_rank_ratio` | 0.2 default. Always show in confirmation table |
| `max_tokens_per_gpu` | 15000 for H100, 8000 for A100, 4000 for consumer GPUs |
| `warmup_steps` | ~1% of total steps: (num_samples / batch_size) x epochs x 0.01 |
| `bf16` | true for Ampere+ (A100, H100), fp16 for older GPUs |

## After Training

The model is registered in MLflow's model registry. Deployable via
serve pipeline or download adapter weights.
