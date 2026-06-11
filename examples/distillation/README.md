# Distillation

Compress a larger teacher model into a tiny student model using Generalized
Knowledge Distillation (GKD) — no synthetic data generation step needed.

## What You'll Build

A student model (Qwen3-0.6B) that learns to mimic a teacher model (Qwen3-4B)
on ticket classification. The student reaches ~90% of teacher accuracy at
1/7th the size and inference cost.

## When to Use Distillation vs SFT

| Approach | When to use |
|----------|-------------|
| **SFT** (ticket-classifier example) | No existing model — train from labeled data |
| **Distillation** (this example) | Already have a working larger model you want to compress |

Distillation transfers the teacher's "soft" knowledge (probability distributions
over outputs), not just hard labels. This often produces better students than
training on the same labeled data with SFT alone.

## Prerequisites

- Amortized server running (`amortized up`)
- Compute backend configured (`amortized config`)
- API key for an LLM provider (for synth data generation)
- ~45 minutes total (5 min synth, 30 min training, 5 min eval)
- GPU with 16 GB+ VRAM (both teacher and student must fit)

## Pipeline

### Step 1: Generate training data

```bash
amortized submit examples/distillation/synth.yaml --confirm
```

Same ticket data as the ticket-classifier example. In a true distillation
workflow, you'd skip this and use unlabeled data — the teacher generates
the labels. Here we use labeled data so you can compare student vs teacher
vs ground truth.

### Step 2: Distill with GKD

```bash
amortized submit examples/distillation/train.yaml \
  --set config.data_path=<sdg-output-path> --confirm
```

Runs GKD: the teacher (Qwen3-4B) generates soft targets, and the student
(Qwen3-0.6B) learns to match the teacher's output distribution.

### Step 3: Serve the student model

```bash
amortized submit recipes/serve/adapter.yaml \
  --set config.model=Qwen/Qwen3-0.6B \
  --set config.adapter=<training-output-path> --confirm
```

### Step 4: Evaluate student vs teacher

```bash
amortized submit examples/distillation/eval.yaml \
  --set config.dataset=<test-data-path> \
  --set config.model_endpoint=<serve-url> --confirm
```

Run the same eval against both the teacher and student endpoints to compare.
