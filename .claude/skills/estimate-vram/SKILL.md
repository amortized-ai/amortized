---
description: Estimate GPU VRAM requirements before starting a training job
invoke-on-demand: true
---

Estimate how much GPU VRAM a LoRA or QLoRA training job will require.

## Parameters
- `model_path` (required) — HuggingFace model ID
- `lora_r` (default: 16) — LoRA rank
- `batch_size` (default: 2) — micro batch size
- `max_seq_len` (default: 2048) — maximum sequence length
- `load_in_4bit` (default: false) — use QLoRA 4-bit quantization

## API call

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_r": 16,
    "batch_size": 2,
    "max_seq_len": 2048,
    "load_in_4bit": false
  }'
```

Returns `estimated_vram_gb`. If this exceeds available GPU memory, suggest:
- Reducing `batch_size` or `max_seq_len`
- Enabling `load_in_4bit` (QLoRA) to significantly reduce VRAM
- Using a smaller model
