# LoRA SFT Training Guide

Parameter-efficient fine-tuning using Low-Rank Adaptation. Good default
for classification, extraction, and summarization tasks.

## Requirement Gathering

Ask the user these questions **one at a time, in separate messages**.
Do NOT skip ahead or combine questions. Wait for the user's answer to
each question before asking the next one.

Do NOT show the confirmation table until the user has answered BOTH
questions below. Ask each in its own message.

1. **Which student model?** — Call `estimate_training_cost` with the
   sample count to get cost estimates for all models. Then present:
   1) Qwen3 0.6B — Fastest, cheapest, great for simple classification
   2) Qwen 2.5 1.5B — Good balance of speed and accuracy
   3) Qwen3 4B — Higher accuracy, needs more VRAM
   4) Llama 3.1 8B — Largest, best accuracy, needs QLoRA for memory

   Wait for the user's response before continuing.

2. **Which training method?** — Call `estimate_training_method_cost`
   with the chosen model and sample count. Then present:
   1) LoRA SFT — Recommended, fastest and cheapest
   2) QLoRA SFT — Lower memory, slightly slower
   3) Full SFT — Best quality, most expensive

   Wait for the user's response before showing the confirmation table.

## Config Defaults

Use these defaults (do NOT ask the user about these unless they bring
them up):
- num_train_epochs: 3
- per_device_train_batch_size: 2
- max_length: 2048
- learning_rate: 0.0002
- lora_r: 16
- lora_alpha: 32

**WARNING:** TRL parameter names differ from common conventions:
- `num_train_epochs` NOT `num_epochs`
- `per_device_train_batch_size` NOT `batch_size`
- `max_length` NOT `max_seq_len`

## Recipe Selection

Use the model-specific recipe from `templates/training/models/`:
- Qwen3 0.6B → `templates/training/models/qwen3-0.6b-lora`
- Qwen 2.5 1.5B → `templates/training/models/qwen2.5-1.5b-lora`
- Qwen3 4B → `templates/training/models/qwen3-4b-lora`
- Llama 3.1 8B → `templates/training/models/llama-3.1-8b-qlora`

Always set `parent_job_id` to the SDG job ID to chain the jobs.
