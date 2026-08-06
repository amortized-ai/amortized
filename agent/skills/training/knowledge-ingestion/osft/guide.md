# OSFT Training Guide

OSFT (Optimized Supervised Fine-Tuning) is the recommended training method
for knowledge ingestion. It outperforms standard SFT by 30+ percentage
points in open-book settings.

## How This Works

You will **create a brand new training config** based on the user's
requirements. The reference template at
`skills/training/knowledge-ingestion/osft/training-config-template.json` shows the config
structure — study it to understand the format, but create a fresh config
tailored to the user's compute setup, model choice, and data.

Every parameter is adjustable:
- Model selection based on available compute
- Learning rate and epochs based on dataset size
- GPU count and batch size based on hardware
- Sequence length based on the SDG output

## Requirement Gathering

Ask the user:

1. **What model?** — Default: `Qwen/Qwen3-8B`. Explain tradeoffs:
   - 0.6B–1.5B: fast inference, lower accuracy, good for prototyping
   - 4B: balanced, single GPU fine-tuning possible
   - 8B: best accuracy for knowledge tasks
   The model choice affects all other hyperparameters — set them after.
2. **GPUs** — Always use 1 GPU (`nproc_per_node: 1`). Do NOT ask the
   user how many GPUs they have or offer GPU count options.
3. **Training data** — Should come from a completed SDG job. Use
   `parent_job_id` to chain SDG → Training automatically. Ask the user
   for the SDG job ID if not already in the conversation.

## Building the Config

```json
{
  "type": "training",
  "config": {
    "algorithm": "osft",
    "model_name_or_path": "<model>",
    "data_path": "<resolved from parent SDG job or specified directly>",
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
    "output_dir": ""
  },
  "parent_job_id": "<SDG_JOB_ID>"
}
```

### Key Parameters — Adjust Based on Data and Compute

Do NOT use static defaults blindly. Adapt each parameter to the user's
dataset size, model choice, and GPU count. Explain your reasoning when
presenting the confirmation table.

| Param | How to set |
|-------|-----------|
| `model_name_or_path` | User's chosen model |
| `num_train_epochs` | 3–5 for <1000 samples, 2–3 for 1000–5000, 1–2 for 5000+. More data needs fewer epochs to avoid overfitting |
| `learning_rate` | 2e-5 for 8B models, 5e-5 for 4B, 1e-4 for 0.6B–1.5B. Larger models need lower LR |
| `effective_batch_size` | Always 32 (1 GPU × 32 per-GPU batch). **Always show in confirmation table.** |
| `max_length` | Must fit the longest context + question + answer from SDG. For knowledge QA with 15-sentence chunks: ~11000. For classification: ~2048 |
| `unfreeze_rank_ratio` | 0.2 is the OSFT default — fraction of weights trainable. **Always show in confirmation table** even when using the default. This is the key OSFT-specific parameter. |
| `nproc_per_node` | Always 1 |
| `max_tokens_per_gpu` | 15000 for H100 (80GB), 8000 for A100 (40GB), 4000 for consumer GPUs. Reduce if OOM |
| `warmup_steps` | ~1% of total steps. total_steps = (num_samples / effective_batch_size) × num_epochs |
| `bf16` | true for Ampere+ GPUs (A100, H100). Use fp16 for older GPUs |

When presenting the confirmation table, show WHY you chose each value:
"5 epochs because the dataset is small (200 samples)" or "learning_rate
2e-5 because Qwen3-8B is a large model".

### Job Chaining

Set `parent_job_id` to the SDG job ID. The worker automatically resolves
the SDG job's MLflow artifact URI and sets `data_path` to point at the
generated dataset. No manual data path configuration needed.

## Data Format

Training expects JSONL with a `messages` column — this is exactly what
the SDG `schema_transform` processor produces:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Compute Requirements

- All training runs use 1 GPU
- Checkpoint saved at each epoch

## After Training

The trained model is registered in MLflow's model registry. The user can
deploy it via the serve pipeline or download the adapter weights.
