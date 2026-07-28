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

1. **What model?** — Default: `Qwen/Qwen3-8B`. Smaller models (0.6B, 1.5B)
   for prototyping, larger (8B) for production.
2. **How many GPUs?** — Default: 8. Adjust `nproc_per_node` and
   `effective_batch_size` accordingly.
3. **Training data** — Should come from a completed SDG job. Use
   `parent_job_id` to chain SDG → Training automatically. Or specify
   `data_path` directly if data is already in S3/MLflow.

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
    "effective_batch_size": 256,
    "max_length": 11000,
    "unfreeze_rank_ratio": 0.2,
    "warmup_steps": 25,
    "save_samples": 0,
    "accelerate_full_state_at_epoch": false,
    "checkpoint_at_epoch": true,
    "nproc_per_node": 8,
    "data_output_dir": "data-output",
    "bf16": true,
    "max_tokens_per_gpu": 15000,
    "output_dir": ""
  },
  "parent_job_id": "<SDG_JOB_ID>"
}
```

### Key Parameters — Adjust Per User

| Param | Default | When to change |
|-------|---------|---------------|
| `model_name_or_path` | `Qwen/Qwen3-8B` | User wants a different base model |
| `num_train_epochs` | 5 | More epochs for small datasets, fewer for large |
| `learning_rate` | 2e-5 | Lower for larger models, higher for smaller |
| `effective_batch_size` | 256 | Scale with GPU count: 32 per GPU × num GPUs |
| `max_length` | 11000 | Must fit context + Q + A from the SDG data |
| `unfreeze_rank_ratio` | 0.2 | OSFT-specific: fraction of weights trainable |
| `nproc_per_node` | 8 | Match user's available GPUs |
| `max_tokens_per_gpu` | 15000 | Reduce if OOM, increase if GPU has more VRAM |
| `warmup_steps` | 25 | Scale with dataset size |
| `bf16` | true | Use fp16 if bf16 not supported |

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

- **8 GPUs** recommended for 8B models
- ~30 minutes for 3000 samples with 8× H100s
- Checkpoint saved at each epoch

## After Training

The trained model is registered in MLflow's model registry. The user can
deploy it via the serve pipeline or download the adapter weights.
