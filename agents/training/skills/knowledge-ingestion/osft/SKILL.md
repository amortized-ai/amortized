---
name: training-knowledge-ingestion-osft
description: >-
  Fine-tune a small model with OSFT (Optimized Supervised Fine-Tuning).
  Recommended for knowledge ingestion, where it outperforms standard SFT by 30+
  points in open-book settings. Trains a messages-format dataset, from an SDG job
  (chained via parent_job_id) or an uploaded dataset / split (data_run_id).
  Covers model selection, hyperparameter tuning by dataset size and compute,
  OSFT's unfreeze_rank_ratio, and dataset chaining. Bundles the supported-model
  list and a config template.
metadata:
  stage: training
---

# OSFT Training Guide

OSFT (Optimized Supervised Fine-Tuning) is the recommended training method
for knowledge ingestion. It outperforms standard SFT by 30+ percentage
points in open-book settings.

> **Scope.** This guide is written around document-grounded QA, but OSFT/SFT also
> covers any plain **`messages`-based** fine-tune — including a **generative (LLM)
> classifier** trained on `sdg-classification`-skill data. For those non-document
> tasks, skip the document-specific steps below (uploads, chunk/coverage math) and
> take the training data from the SDG `parent_job_id` as usual; the model,
> epochs, and learning-rate guidance still apply. (Note: most classifier tasks
> want the **`sdg-embedding-classifier`** sub-skill instead — this generative path is
> the rare case where the user explicitly wants an LLM to emit the label.)

## How This Works

You will gather requirements and call `validate_training_job` with the
appropriate parameters. The tool validates all fields and rejects
missing required params (e.g. `unfreeze_rank_ratio` for OSFT).

Every parameter is adjustable:
- Model selection based on available compute
- Learning rate and epochs based on dataset size
- GPU count and batch size based on hardware
- Sequence length based on the SDG output

## Requirement Gathering

Ask the user:

1. **What model?** — Read `supported_models.json` (bundled with this
   skill) for the list of verified models. **Recommend a model from
   this list first** — every model in it has been tested end-to-end
   with compatible architecture and configuration. Present the options
   with their sizes and descriptions.

   **If the user wants a model that is NOT in the list**, follow this
   flow:

   a. **Validate the model.** Check two things:
      - **Valid HuggingFace model ID.** The model must have a valid
        HuggingFace identifier (e.g. `org/model-name`) that can be
        resolved and downloaded.
      - **Compatible model architecture.** The model's architecture
        must be suitable for the intended task. For example, a text
        generation task requires a generative (decoder) model — an
        encoder-only model designed for embeddings or classification
        would not be appropriate. If the model passes both checks,
        accept it and proceed.

   b. **If the model looks risky** (fails either check, or the
      architecture cannot be determined), warn the user and suggest a
      similar model from the supported list. If the user insists,
      allow it — state clearly: "This model is not in our verified
      list and may not be compatible. We will proceed with your
      choice, but if the job fails, consider switching to one of the
      supported models." Then continue normally.

   The model choice affects all downstream hyperparameters — gather
   them after.

2. **GPUs** — Estimate the number of GPUs needed based on the model
   size and available VRAM. Default to 1 GPU (`nproc_per_node: 1`)
   for models that fit in a single GPU's memory. For larger models
   (e.g. 14B+ parameters), recommend additional GPUs as needed and
   explain the reasoning.
3. **Training data** — Should come from a completed SDG job. Use
   `parent_job_id` to chain SDG → Training automatically. Ask the user
   for the SDG job ID if not already in the conversation.

## Tool Parameters

Call `validate_training_job` with these parameters:

```json
{
  "algorithm": "osft",
  "model_name_or_path": "<model>",
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
  "parent_job_id": "<SDG_JOB_ID>",
  "topic": "<1-5 word topic>"
}
```

### Key Parameters — Adjust Based on Model Size, Data, and Compute

Do NOT use static defaults blindly. Adapt each parameter to the user's
dataset size, model choice, and GPU count. Explain your reasoning when
presenting the confirmation table.

| Param | How to set |
|-------|-----------|
| `model_name_or_path` | User's chosen model |
| `num_train_epochs` | 3–5 for <1000 samples, 2–3 for 1000–5000, 1–2 for 5000+. More data needs fewer epochs to avoid overfitting |
| `learning_rate` | Scale inversely with model size. Guideline: ≤2B → 6e-5, 3B–4B → 5e-5, 7B–9B → 2e-5, 14B+ → 1e-5. Larger models are more sensitive to high learning rates |
| `effective_batch_size` | 32 (nproc_per_node × per-GPU batch). **Always show in confirmation table.** |
| `max_length` | Must fit the longest context + question + answer from SDG. For knowledge QA with 15-sentence chunks: ~11000. For simple classification (short label output): ~2048. For task distillation (scoring, assessment, rubric-based evaluation — output is long-form feedback): **~16000**. Check which SDG skill produced the data to decide. Validate by checking the longest example in the SDG dataset. |
| `unfreeze_rank_ratio` | 0.2 is the OSFT default — fraction of weights trainable. **Always show in confirmation table** even when using the default. This is the key OSFT-specific parameter. |
| `nproc_per_node` | Default 1. Increase for large models that exceed single-GPU VRAM |
| `max_tokens_per_gpu` | 15000 for H100 (80GB), 8000 for A100 (40GB), 4000 for consumer GPUs. Reduce if OOM |
| `warmup_steps` | ~1% of total steps. total_steps = (num_samples / effective_batch_size) × num_epochs |
| `bf16` | true for Ampere+ GPUs (A100, H100). Use fp16 for older GPUs |
| `topic` | **Required.** 1–5 word description of what the model does (e.g. "support ticket classification"). Used for tracking in the Models page. Infer from the dataset/task if not explicitly stated. |

When presenting the confirmation table, show WHY you chose each value:
"5 epochs because the dataset is small (200 samples)" or "learning_rate
2e-5 because Qwen3.5-9B is a large model".

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

- Default to 1 GPU; increase based on model size and VRAM estimates
- Checkpoint saved at each epoch

## After Training

The trained model is registered in MLflow's model registry. The user can
deploy it via the serve pipeline or download the adapter weights.
