# Entity Extractor

Fine-tune a model to extract structured entities (people, organizations,
dates, products) from unstructured text, replacing expensive frontier model calls.

## What You'll Build

A task model that takes unstructured text and outputs a JSON array of extracted entities
with "entity" and "type" fields. Supports 6 entity types:
- **Person**: Names of people
- **Organization**: Company and organization names
- **Location**: Places, addresses, geographic entities
- **Date**: Dates, times, durations
- **Product**: Product names, model numbers
- **Monetary**: Prices, amounts, currencies

## Prerequisites

- Amortized server running (`amortized up`)
- Compute backend configured (`amortized config`)
- API key for an LLM provider (OpenAI, Anthropic, etc.) set as env var
- ~35 minutes total (5 min synth, 20 min training, 5 min eval)

## Pipeline

### Step 1: Generate training data (100 labeled examples)

```bash
amortized submit examples/entity-extractor/synth.yaml --confirm
```

This generates 100 text samples across 4 text types (email, article, report,
social media) with controlled entity type distributions and extracted entity labels.

### Step 2: Fine-tune with LoRA SFT

```bash
amortized submit examples/entity-extractor/train.yaml \
  --set config.data_path=<sdg-output-path> --confirm
```

Trains a Qwen 2.5 1.5B model with LoRA. Uses gradient checkpointing for
longer sequences (max_length: 4096). Takes ~20 minutes on a single GPU.

### Step 3: Serve the fine-tuned model

```bash
amortized submit recipes/serve/adapter.yaml \
  --set config.model=Qwen/Qwen2.5-1.5B-Instruct \
  --set config.adapter=<training-output-path> --confirm
```

### Step 4: Evaluate

```bash
amortized submit examples/entity-extractor/eval.yaml \
  --set config.dataset=<test-data-path> \
  --set config.model_endpoint=<serve-url> --confirm
```
