# OSFT Training Workflow -- Task Distillation

OSFT for task distillation tasks (scoring, assessment, structured
evaluation). Same algorithm as knowledge ingestion but with different
hyperparameters: longer sequences, lower learning rate, fewer epochs.

## Workflow

### Step 1 -- Model

Default: `Qwen/Qwen3.5-9B`. Present ONLY these exact HuggingFace
model IDs (do NOT invent other sizes):
- `Qwen/Qwen3.5-4B` -- lighter, faster inference
- `Qwen/Qwen3.5-9B` -- recommended for structured assessment tasks

Do NOT suggest models that don't exist (e.g. Qwen3.5-0.6B,
Qwen3.5-1.5B, Qwen3.5-8B -- these are not real).

### Step 2 -- Training data

Should come from a completed SDG job. Use `parent_job_id` to chain
SDG -> Training. Ask the user for the SDG job ID if not in the
conversation.

### Step 3 -- Confirm config

Present a confirmation table with all parameters. Show WHY you chose
each value: "3 epochs because task distillation data is
information-dense" or "learning_rate 5e-6 to prevent overfitting on
structured output".

GPUs default to 4 (`nproc_per_node: 4`) for task distillation due to
longer sequences.

## Reference Payload

Use this as the base for `create_training_job()`. Adapt parameters
based on the user's model choice and dataset size.

```json
{
  "algorithm": "osft",
  "model_name_or_path": "Qwen/Qwen3.5-9B",
  "num_train_epochs": 3,
  "learning_rate": 5e-06,
  "effective_batch_size": 32,
  "max_length": 16384,
  "unfreeze_rank_ratio": 0.25,
  "warmup_steps": 0,
  "save_samples": 0,
  "accelerate_full_state_at_epoch": true,
  "checkpoint_at_epoch": true,
  "nproc_per_node": 4,
  "data_output_dir": "data-output",
  "bf16": true,
  "max_tokens_per_gpu": 16384,
  "parent_job_id": "<SDG_JOB_ID>"
}
```

### Adapting the Payload

| Param | How to adapt |
|-------|-------------|
| `model_name_or_path` | User's chosen model |
| `num_train_epochs` | 3 default. 2 for >2000 samples, 4-5 for <500 samples |
| `learning_rate` | 5e-6 for 9B (default). 1e-5 for 4B |
| `effective_batch_size` | 32. Smaller than knowledge ingestion due to longer sequences |
| `max_length` | 16384 to fit rubric system prompt + input + full assessment |
| `unfreeze_rank_ratio` | 0.25 default. Slightly higher than knowledge ingestion (0.2) |
| `max_tokens_per_gpu` | 16384 for H100, 8192 for A100 |
| `warmup_steps` | 0 with cosine LR scheduler |
| `nproc_per_node` | 4 default. Adjust based on available GPUs |
| `bf16` | true for Ampere+ (A100, H100), fp16 for older GPUs |

## After Training

The model is registered in MLflow's model registry. Deployable via
serve pipeline or download adapter weights.
