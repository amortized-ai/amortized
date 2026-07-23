# LoRA SFT Training Guide

*This guide is a placeholder. It will be populated with expert knowledge
for LoRA SFT fine-tuning.*

In the meantime, use the recipe template at `templates/training/lora-sft`
and the model presets in `templates/training/models/`.

Key parameters (TRL CLI):
- `model_name_or_path`: HuggingFace model ID
- `num_train_epochs`: Number of epochs (not `num_epochs`)
- `per_device_train_batch_size`: Batch size per GPU (not `batch_size`)
- `max_length`: Max sequence length (not `max_seq_len`)
- Defaults: lr=2e-4, epochs=3, batch=2, max_len=2048, lora_r=16, lora_alpha=32
