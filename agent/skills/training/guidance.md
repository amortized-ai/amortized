# Training Skill Guidance

Pick the training method that best matches the user's task and data.
Read its `guide.md` for deep expertise before configuring the training job.

## Available Sub-Skills

| Sub-Skill | Path | Best For |
|-----------|------|----------|
| osft | `skills/training/osft/` | Default choice. Outperforms standard SFT significantly, especially in open-book/RAG settings |
| lora-sft | `skills/training/lora-sft/` | Parameter-efficient fine-tuning. Good general-purpose default when OSFT is not applicable |
| qlora | `skills/training/qlora/` | Quantized LoRA. Required for large models (8B+) when GPU memory is limited |

## How to Choose

- **Document-grounded QA, knowledge tasks, RAG models** → `osft` (30+ pp improvement over standard SFT in open-book settings)
- **General classification, extraction, summarization** → `lora-sft` (well-understood, reliable)
- **Large models (8B+) with limited GPU memory** → `qlora` (4-bit quantization reduces memory)
- **SDG skill guide recommended a specific method** → follow that recommendation, but let the user choose

If the SDG sub-skill guide recommends a training method, mention that
recommendation when presenting options to the user.

## Training Flow — Separate Steps

Ask these questions ONE AT A TIME, each in its own message:

1. **Which student model?** — Present model options with sizes (0.6B, 1.5B,
   4B, 8B). Call `estimate_training_cost` to show cost comparison.
2. **Which training method?** — Present LoRA SFT, QLoRA, Full SFT. Call
   `estimate_training_method_cost` with the chosen model to show method
   cost comparison.
3. **Confirm plan** — Show confirmation table with all settings.

Do NOT combine model selection and method selection into one message.
Do NOT auto-select a training method — always let the user choose.

## After Loading the Sub-Skill

The sub-skill's `guide.md` will tell you:
- Recommended hyperparameters and why
- Model selection guidance (which student models work best with this method)
- Compute requirements (GPUs, VRAM, estimated time)
- Config template to use as a starting point
