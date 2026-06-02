---
description: Submit a LoRA SFT training job to the Amortized runtime API
invoke-on-demand: true
---

Submit a LoRA SFT training job by calling the Amortized runtime API.

## Required parameters
- `model_path` — HuggingFace model ID (e.g. `Qwen/Qwen2.5-1.5B-Instruct`)
- `data_path` — path to training data JSONL file
- `ckpt_output_dir` — directory for checkpoints and outputs

## Optional parameters
- `learning_rate` (default: 2e-4)
- `num_epochs` (default: 3)
- `lora_r` (default: 16)
- `lora_alpha` (default: 32)
- `load_in_4bit` (default: false) — enable QLoRA for reduced VRAM
- `micro_batch_size` (default: 2)
- `max_seq_len` (default: 2048)

## API call

```bash
curl -X POST http://localhost:8000/api/v1/jobs/training \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "data_path": "./data.jsonl",
    "ckpt_output_dir": "./outputs",
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "lora_r": 16,
    "lora_alpha": 32
  }'
```

The response contains a `job_id` that can be used to check status and metrics.
