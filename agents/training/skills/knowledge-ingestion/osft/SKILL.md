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
   this list first** — every model in it has been tested with Training
   Hub (correct chat template, compatible architecture). Present the
   options with their sizes and descriptions.

   **If the user wants a model that is NOT in the list**, follow this
   flow:

   a. **Validate the model.** Assess whether it is likely compatible
      with Training Hub. Training Hub loads models via
      `AutoModelForCausalLM.from_pretrained()` and tokenizers via
      `AutoTokenizer.from_pretrained()`. A model must satisfy these
      requirements:
      - **From a known model family.** Training Hub has validated
        these architectures: Qwen, Qwen3, Qwen3.5, Llama, Granite,
        Phi, Gemma, Mistral, DeepSeek, and Nemotron. Models from
        these families are very likely to work. Models from other
        families may work but have not been tested.
      - **Is a causal language model.** Training Hub loads with
        `AutoModelForCausalLM` — encoder-only models (e.g. BERT) or
        multimodal models will fail at loading.
      - **Has a tokenizer with a `chat_template`.** Training Hub
        calls `apply_chat_template()` to format the `messages`
        training data into token IDs — this applies to OSFT, SFT,
        and LoRA. If the tokenizer lacks a `chat_template` field,
        data preprocessing fails. Instruct/chat variants include
        this; raw base models typically do not.
      - **Has a valid HuggingFace model ID** (`org/model-name`) so
        `from_pretrained()` can resolve it.

      If the model meets these criteria — known family, causal LM,
      instruct/chat variant with a chat template — accept it and
      proceed. Let the user know it is not in the pre-verified list
      but appears compatible.

   b. **If the model looks risky** (unknown model family, raw base
      model without a chat template, encoder-only or multimodal
      architecture, or an unrecognized source), warn the user. Explain
      that Training Hub has only validated certain model families and
      requires a causal LM architecture with a functional tokenizer —
      if the model is outside these, training may fail at startup.
      Suggest a similar model from the supported list as an
      alternative.

   c. **If the user insists** on using a model you warned about, allow
      it — but state clearly: "This model is not in our verified list
      and may lack a chat template. Training could fail at startup if
      Training Hub cannot format the data. We'll proceed with your
      choice, but if the job fails, consider switching to one of the
      supported models." Then continue with the rest of the flow
      normally.

   The model choice affects all downstream hyperparameters — gather
   them after.

2. **GPUs** — Always use 1 GPU (`nproc_per_node: 1`). Do NOT ask the
   user how many GPUs they have or offer GPU count options.
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
| `effective_batch_size` | Always 32 (1 GPU × 32 per-GPU batch). **Always show in confirmation table.** |
| `max_length` | Must fit the longest context + question + answer from SDG. For knowledge QA with 15-sentence chunks: ~11000. For simple classification (short label output): ~2048. For task distillation (scoring, assessment, rubric-based evaluation — output is long-form feedback): **~16000**. Check which SDG skill produced the data to decide. Validate by checking the longest example in the SDG dataset. |
| `unfreeze_rank_ratio` | 0.2 is the OSFT default — fraction of weights trainable. **Always show in confirmation table** even when using the default. This is the key OSFT-specific parameter. |
| `nproc_per_node` | Always 1 |
| `max_tokens_per_gpu` | Scale by GPU memory and model size. H100 (80GB): 15000 for ≤9B, 10000 for 14B, 4000 for 30B+. A100 (40GB): 8000 for ≤4B, 5000 for 7B–9B, 3000 for 14B+. Consumer GPUs (24GB): 4000 for ≤4B, 2000 for 7B–9B. Reduce if OOM |
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

- All training runs use 1 GPU
- Checkpoint saved at each epoch

## After Training

The trained model is registered in MLflow's model registry. The user can
deploy it via the serve pipeline or download the adapter weights.
