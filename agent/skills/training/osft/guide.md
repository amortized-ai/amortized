# OSFT Training Guide

OSFT (Optimized Supervised Fine-Tuning) is the default training method for
knowledge ingestion and document-grounded QA tasks. It outperforms standard
SFT by 30+ percentage points in open-book settings.

## When to Use OSFT

- Document-grounded QA, FAQ bots, knowledge assistants
- RAG-deployed models
- Any task where the model needs to faithfully answer from provided context
- Use unless there is a specific reason to choose another method

## Key Hyperparameters

| Param | Recommended | Notes |
|-------|-------------|-------|
| `algorithm` | `osft` | |
| `unfreeze_rank_ratio` | 0.2 | Fraction of weights trainable in OSFT |
| `num_train_epochs` | 5 | |
| `learning_rate` | 2e-5 | Standard for OSFT with 8B models, adjust based on data scale |
| `effective_batch_size` | 256 | Total batch across all GPUs |
| `max_length` / `max_seq_len` | 11000 | Must accommodate context + question + answer |
| `max_tokens_per_gpu` | 15000 | Memory budget per GPU for token batching |
| `warmup_steps` | 25 | Short warmup sufficient for ~3000 samples |
| `bf16` | true | Mixed precision training |

## Model Selection

- **Qwen/Qwen3-8B** — recommended default for OSFT knowledge tasks
- Adjust `learning_rate` based on model size and data scale

## Data Format

Training Hub expects JSONL with a `messages` column:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Extra columns (topic, difficulty, etc.) are ignored by the trainer but
useful for stratified evaluation.

## Compute Requirements

- **8 GPUs** recommended for OSFT with 8B models
- `nproc_per_node: 8`
- `checkpoint_at_epoch: true` for saving checkpoints

## Config Template

The config template at `skills/training/osft/training-config-template.json`
shows the golden parameterization for an OSFT training job. Customize
`model_name_or_path`, `data_path`, and `output_dir` based on the user's
setup. The hyperparameters are tuned defaults — only adjust if the user
has specific requirements.
