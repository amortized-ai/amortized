# QLoRA Training Guide

*This guide is a placeholder. It will be populated with expert knowledge
for QLoRA (quantized LoRA) fine-tuning.*

QLoRA is required for large models (8B+) when GPU memory is limited.
Uses 4-bit quantization (`load_in_4bit=true`) to reduce memory footprint.

In the meantime, use the recipe template at `templates/training/lora-sft`
with `load_in_4bit: true` added to the config.
