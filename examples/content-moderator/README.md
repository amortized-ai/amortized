# Content Moderator

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
  --set config.test_data_path=<test-data-path> \
  --set config.model_endpoint=<serve-url> --confirm
```

## Expected Results

| Metric | Base Model | Fine-tuned |
|--------|-----------|------------|
| Overall accuracy | ~75% | ~95%+ |
| Precision | ~70% | ~93%+ |
| Recall | ~65% | ~96%+ |
| F1 | ~67% | ~94%+ |

**Critical metric: false negative rate.** Missing unsafe content is far worse than
over-flagging safe content. The fine-tuned model targets <5% false negative rate.

## Customization

- **More data**: Increase `num_samples` in `synth.yaml` to 500-1000 for better recall
- **Different model**: Change `model_name_or_path` in `train.yaml` (Qwen3-0.6B is sufficient for binary classification)
- **Custom violation types**: Edit the `violation_type` possible values in `synth.yaml`
- **Adjust safe/unsafe ratio**: Change `sample_rate` on `safety_label` to match your expected distribution

## GPU Requirements

| Stage | GPU | VRAM | Time |
|-------|-----|------|------|
| Synth | None (API calls) | 0 | ~5 min |
| Training | 1x GPU | 8 GB+ | ~10 min |
| Serving | 1x GPU | 2 GB+ | — |
| Eval | None (API calls) | 0 | ~2 min |
