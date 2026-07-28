# Intent Router

> **Note:** The CLI recipe files (`synth.yaml`, `train.yaml`, `eval.yaml`)
> were removed in the Data Designer migration. Use **Amortized Studio → Chat**
> to build this pipeline interactively — tell Morty "build an intent router"
> and it will guide you through SDG and training.

Fine-tune the smallest possible model to classify customer messages by intent,
replacing a frontier model on the hot path of every API request.

## What You'll Build

A task model that takes a customer message and outputs one of 8 intent labels:
- **billing_inquiry** — charges, invoices, payment methods
- **technical_support** — bugs, errors, how-to
- **account_management** — password reset, profile updates
- **order_status** — tracking, delivery ETA
- **refund_request** — returns, disputes
- **feature_request** — suggestions, improvements
- **complaint** — dissatisfaction, escalation
- **general_inquiry** — pricing, availability, policies

## Prerequisites

- Amortized server running (`amortized up`)
- Compute backend configured (`amortized config`)
- API key for an LLM provider (OpenAI, Anthropic, etc.) set as env var
- ~20 minutes total (5 min synth, 10 min training, 2 min eval)

## Pipeline

### Step 1: Generate training data (200 labeled messages)

```bash
amortized submit examples/intent-router/synth.yaml --confirm
```

Generates 200 realistic customer messages across 8 intents and 4 tones,
formatted as SFT training conversations.

### Step 2: Fine-tune with LoRA SFT

```bash
amortized submit examples/intent-router/train.yaml \
  --set config.data_path=<sdg-output-path> --confirm
```

Trains a Qwen3-0.6B model with LoRA. Uses the smallest model for minimal
inference latency — intent routing is on the hot path.

### Step 3: Serve the fine-tuned model

```bash
amortized submit recipes/serve/adapter.yaml \
  --set config.model=Qwen/Qwen3-0.6B \
  --set config.adapter=<training-output-path> --confirm
```

### Step 4: Evaluate

```bash
amortized submit examples/intent-router/eval.yaml \
  --set config.dataset=<test-data-path> \
  --set config.model_endpoint=<serve-url> --confirm
```
