# Summarizer

Fine-tune a model to condense customer conversations into structured summaries,
replacing verbose frontier model outputs with consistent, formatted summaries.

## What You'll Build

A task model that takes a multi-turn conversation and outputs a structured summary:
- **Issue** — one sentence describing the customer's problem
- **Resolution** — one sentence describing the outcome (or "Unresolved")
- **Action Items** — bullet list of follow-ups (or "None")
- **Sentiment** — positive, neutral, or negative

## Prerequisites

- Amortized server running (`amortized up`)
- Compute backend configured (`amortized config`)
- API key for an LLM provider (OpenAI, Anthropic, etc.) set as env var
- ~35 minutes total (5 min synth, 20 min training, 5 min eval)

## Pipeline

### Step 1: Generate training data (100 conversations + summaries)

```bash
amortized submit examples/summarizer/synth.yaml --confirm
```

Generates 100 realistic business conversations across 4 domains (support, sales,
HR, legal) with varying lengths (2-15 turns), each paired with a structured summary.

### Step 2: Fine-tune with LoRA SFT

```bash
amortized submit examples/summarizer/train.yaml \
  --set config.data_path=<sdg-output-path> --confirm
```

Trains a Qwen 2.5 1.5B model with LoRA. Uses `gradient_checkpointing` and
`max_length: 4096` to handle longer conversation inputs.

### Step 3: Serve the fine-tuned model

```bash
amortized submit recipes/serve/adapter.yaml \
  --set config.model=Qwen/Qwen2.5-1.5B-Instruct \
  --set config.adapter=<training-output-path> --confirm
```

### Step 4: Evaluate

```bash
amortized submit examples/summarizer/eval.yaml \
  --set config.test_data_path=<test-data-path> \
  --set config.model_endpoint=<serve-url> --confirm
```

## Expected Results

| Metric | Base Model | Fine-tuned |
|--------|-----------|------------|
| Structure compliance | ~50% | ~95%+ |
| Judge pass rate | ~60% | ~90%+ |
| Per-field accuracy | ~55% | ~85%+ |

Base models tend to produce verbose, unstructured summaries. The fine-tuned model
consistently outputs the exact 4-field format.

## Customization

- **More data**: Increase `num_samples` in `synth.yaml` to 500 for better quality
- **Different domains**: Edit `possible_values` for the `domain` attribute
- **Longer conversations**: Adjust the `length` distribution in `synth.yaml`
- **Different output format**: Modify the system prompt in `synth.yaml` to change the summary structure
- **Larger model**: Change to `Qwen/Qwen3-4B` in `train.yaml` for higher quality summaries

## GPU Requirements

| Stage | GPU | VRAM | Time |
|-------|-----|------|------|
| Synth | None (API calls) | 0 | ~5 min |
| Training | 1x GPU | 8 GB+ | ~20 min |
| Serving | 1x GPU | 4 GB+ | — |
| Eval | None (API calls) | 0 | ~5 min |
