# Entity Extractor

Fine-tune a model to extract structured entities (people, organizations,
dates, products) from unstructured text.

## Pipeline

```bash
# 1. Generate extraction training data
amortized submit sdg --recipe examples/entity-extractor/synth --confirm

# 2. Fine-tune with LoRA SFT
amortized submit training --recipe examples/entity-extractor/train \
  --model Qwen/Qwen2.5-1.5B-Instruct --data <sdg-artifact-id> --confirm

# 3. Serve and evaluate
amortized submit serve --recipe serve/adapter \
  --model Qwen/Qwen2.5-1.5B-Instruct --adapter <model-artifact-id> --confirm
amortized submit eval --recipe examples/entity-extractor/eval \
  --data <sdg-artifact-id> --serve <serve-job-id> --confirm
```
