# Ticket Classifier

Fine-tune a small model to classify customer support tickets by urgency and topic,
replacing expensive frontier model calls.

## What You'll Build

A task model that takes a customer support ticket and outputs:
- **Urgency**: low, medium, high, critical
- **Topic**: orders, shipping, returns, payments, product_questions, account_issues

## Prerequisites

- Amortized server running (`amortized up`)
- Compute backend configured (`amortized config`)
- API key for an LLM provider (OpenAI, Anthropic, etc.) set as env var
- ~30 minutes total (5 min synth, 15 min training, 5 min eval)

## Pipeline

### Step 1: Generate training data (100 labeled tickets)

```bash
amortized submit examples/ticket-classifier/synth.yaml --confirm
```

This generates 100 realistic customer support tickets with controlled
urgency/topic distributions, formatted as SFT training conversations.

### Step 2: Fine-tune with LoRA SFT

```bash
amortized submit examples/ticket-classifier/train.yaml \
  --set config.data_path=<sdg-output-path> --confirm
```

Trains a Qwen 2.5 1.5B model with LoRA. Takes ~15 minutes on a single GPU.

### Step 3: Serve the fine-tuned model

```bash
amortized submit recipes/serve/adapter.yaml \
  --set config.model=Qwen/Qwen2.5-1.5B-Instruct \
  --set config.adapter=<training-output-path> --confirm
```

### Step 4: Evaluate

```bash
amortized submit examples/ticket-classifier/eval.yaml \
  --set config.dataset=<test-data-path> \
  --set config.model_endpoint=<serve-url> --confirm
```
