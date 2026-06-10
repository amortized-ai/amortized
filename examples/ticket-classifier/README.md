# Ticket Classifier

Fine-tune a small model to classify customer support tickets by urgency and topic,
replacing expensive frontier model calls.

## Pipeline

```bash
# 1. Generate training data (100 labeled tickets)
amortized submit sdg --recipe examples/ticket-classifier/synth --confirm

# 2. Fine-tune with LoRA SFT
amortized submit training --recipe examples/ticket-classifier/train \
  --model Qwen/Qwen2.5-1.5B-Instruct --data <sdg-artifact-id> --confirm

# 3. Serve the fine-tuned model
amortized submit serve --recipe serve/adapter \
  --model Qwen/Qwen2.5-1.5B-Instruct --adapter <model-artifact-id> --confirm

# 4. Evaluate
amortized submit eval --recipe examples/ticket-classifier/eval \
  --data <sdg-artifact-id> --serve <serve-job-id> --confirm
```

## Expected Results

- Base model: ~60% urgency accuracy, ~80% topic accuracy
- Fine-tuned model: ~90%+ on both with sufficient training data
