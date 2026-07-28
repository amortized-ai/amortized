# Content Moderator

> **Note:** The CLI recipe files (`synth.yaml`, `train.yaml`, `eval.yaml`)
> were removed in the Data Designer migration. Use **Amortized Studio → Chat**
> to build this pipeline interactively — tell Morty "build a content moderator"
> and it will guide you through SDG and training.

Fine-tune a tiny model for binary content moderation (safe vs unsafe),
replacing expensive frontier model calls on the moderation hot path.

## What You'll Build

A task model that takes user-generated content and classifies it as:
- **safe** — appropriate content following community guidelines
- **unsafe** — content violating guidelines (harassment, hate speech, misinformation, spam, self-harm)

This is the simplest possible task model — binary classification with a single output token.

## Prerequisites

- Amortized server running (`amortized up`)
- Compute backend configured (`amortized config`)
- API key for an LLM provider (OpenAI, Anthropic, etc.) set as env var
- ~25 minutes total (5 min synth, 10 min training, 2 min eval)

## Pipeline

### Step 1: Generate training data (200 labeled content samples)

```bash
amortized submit examples/content-moderator/synth.yaml --confirm
```

Generates 200 content samples across 4 content types (comments, reviews,
messages, posts) with a 60/40 safe/unsafe split and 5 violation categories.

### Step 2: Fine-tune with LoRA SFT

```bash
amortized submit examples/content-moderator/train.yaml \
  --set config.data_path=<sdg-output-path> --confirm
```

Trains Qwen3-0.6B with LoRA. Binary classification trains fast — ~10 minutes on a single GPU.

### Step 3: Serve the fine-tuned model

```bash
amortized submit recipes/serve/adapter.yaml \
  --set config.model=Qwen/Qwen3-0.6B \
  --set config.adapter=<training-output-path> --confirm
```

### Step 4: Evaluate

```bash
amortized submit examples/content-moderator/eval.yaml \
  --set config.dataset=<test-data-path> \
  --set config.model_endpoint=<serve-url> --confirm
```
